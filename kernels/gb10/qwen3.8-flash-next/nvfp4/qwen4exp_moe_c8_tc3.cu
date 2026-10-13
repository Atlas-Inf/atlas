// SPDX-License-Identifier: AGPL-3.0-only
//
// TC MoE v3 (ATLAS_QWEN4EXP_MOE_TC=1, the default TC kernels; v2 behind
// ATLAS_QWEN4EXP_MOE_TC_V2=1): qwen4exp_moe_c8_tc.cu's gate/up + SiLU and
// down in ONE persistent launch, the same bytes. Contract (b) as there:
// every output runs the same MMA sequence (qwen4exp_moe_tc.cuh's k order,
// gate/up's 4 K quarters summed in order, scale2 in the epilogue), so a row's
// outputs do not depend on the other rows of the launch.
//
// Why: the TC kernels are DRAM-latency bound (few loads in flight per SM, a
// grid boundary and its tail between gate/up and down, an expert of more
// than 16 rows read once per 16). Here:
//   - a unit is ALL of an expert's rows (up to 64: 1-4 m16 tiles reusing the
//     same weight registers), so every expert's weights are read once;
//   - CTAs are persistent and claim work items in order from a counter, each
//     item's weights (a warp: 5 x 16 bytes a lane + 5 scale pairs) loaded
//     into registers one item AHEAD of its math (register double buffer);
//   - items: Z (zero a remote expert's rows), G (gate/up of a unit's 8
//     columns, 80 a unit), D (down of a unit's 64 columns, 40 a unit). A D
//     item issues its weight loads, then waits for its unit's 80 G items
//     (a counter, release/acquire); items are claimed in order and every
//     G precedes every D, so a waited-on item is always held by a running
//     CTA (no deadlock whatever the residency).
// Activations pass between G and D as BF16 (down rounds them to BF16 first
// anyway: the same bytes as v2's FP32 + round).
//
// MEASURED (preliminary: a GB10 shared with other tenants; one
// EP2 rank's 256-expert pool, real C8 routing (REALBIN, a layer a wave),
// us plan + gate/up + down a layer, min of 11 interleaved passes, v2 -> v3):
// 1 row 72.6 -> 69.2, 16 rows 465.3 -> 410.5, 32 rows 730.9 -> 628.1
// (204 -> 237 GB/s of unique expert bytes), 36 rows 802.0 -> 682.6.
// L2-resident (POOL=3) 32 rows: 305 -> 315 -- the win is DRAM overlap.
// Tried: claims two items ahead, a shared-memory table of the units'
// weight pointers, red.release for the counters (each 0 to 2% slower);
// 1 CTA a SM (math-bound: 487 us L2-resident).
//
// Workspace (qwen4exp_moe_c8_tc3_plan): ws[0] units, ws[1] the claim
// counter, ws[2] local units L (local units first, heaviest first; then
// remote), units at C8_WS_UNITS, rows at C8_WS_ROWS, the G-done counters
// at TC3_WS_DONE.

#include "qwen4exp_moe_c8.cuh"
#include "qwen4exp_moe_tc.cuh"

#define TC3_RMAX 64u
#define TC3_WS_DONE (C8_WS_ROWS + C8_SLOTS_MAX)
#define TC3_UMAX 529u  // 512 experts + 1 shared + splits of > 64 entries
#define TC3_G 80u      // G items a unit: 640 / 8
#define TC3_D 40u      // D items a unit: 2560 / 64
// The shared unit takes bucket `rows` among the local 1..TC3_RMAX.
static_assert(C8_ROWS_MAX <= TC3_RMAX, "shared unit bucket");

// v3 plan: v2's counting sort with units of up to TC3_RMAX entries, local
// units (gate pointer non-null, and the shared unit) before remote ones.
extern "C" __global__ void __launch_bounds__(1024) qwen4exp_moe_c8_tc3_plan(
    const unsigned int* __restrict__ expert_indices, const unsigned long long* __restrict__ gate_packed_ptrs,
    unsigned int* __restrict__ ws, unsigned int top_k, unsigned int rows
) {
    constexpr unsigned NEXP = 512, NB = TC3_RMAX + 2;  // buckets: local 1..64 by n, 65 remote
    __shared__ unsigned s_cnt[NEXP], s_off[NEXP], s_bucket[NB], s_base[NB], s_wsum[16];
    const unsigned slots = rows * top_k;
    // rows >= 1: bucket 0, where no rows would put the shared unit, has no base.
    if (rows == 0 || slots > C8_SLOTS_MAX || rows > C8_ROWS_MAX || blockDim.x != 1024) __trap();
    for (unsigned i = threadIdx.x; i < NEXP; i += blockDim.x) s_cnt[i] = 0;
    if (threadIdx.x < NB) s_bucket[threadIdx.x] = 0;
    __syncthreads();
    unsigned e = 0, at = 0;
    if (threadIdx.x < slots) {
        e = expert_indices[threadIdx.x];
        if (e >= NEXP) __trap();
        at = atomicAdd(&s_cnt[e], 1u);
    }
    __syncthreads();
    const unsigned lane = threadIdx.x & 31u, w = threadIdx.x >> 5;
    unsigned v = 0, incl = 0;
    if (threadIdx.x < NEXP) {
        v = s_cnt[threadIdx.x];
        incl = v;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const unsigned n = __shfl_up_sync(0xFFFFFFFFu, incl, o);
            if (lane >= (unsigned)o) incl += n;
        }
        if (lane == 31) s_wsum[w] = incl;
    }
    __syncthreads();
    if (threadIdx.x < NEXP) {
        unsigned base = 0;
        for (unsigned i = 0; i < w; i++) base += s_wsum[i];
        s_off[threadIdx.x] = base + incl - v;
    }
    __syncthreads();
    if (threadIdx.x < slots) ws[C8_WS_ROWS + s_off[e] + at] = threadIdx.x;
    const bool is_e = threadIdx.x < NEXP, is_sh = threadIdx.x == NEXP;
    const unsigned total = is_e ? s_cnt[threadIdx.x] : 0u;
    const bool remote = is_e && total && gate_packed_ptrs[threadIdx.x] == 0;
    const auto bucket = [&](unsigned n) { return remote ? TC3_RMAX + 1 : n; };
    if (is_e)
        for (unsigned k = 0; k < total; k += TC3_RMAX) atomicAdd(&s_bucket[bucket(min(TC3_RMAX, total - k))], 1u);
    if (is_sh) atomicAdd(&s_bucket[rows], 1u);
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned b = 0;
        for (unsigned n = TC3_RMAX; n >= 1; n--) { s_base[n] = b; b += s_bucket[n]; s_bucket[n] = 0; }
        ws[2] = b;  // local units
        s_base[TC3_RMAX + 1] = b;
        b += s_bucket[TC3_RMAX + 1];
        s_bucket[TC3_RMAX + 1] = 0;
        ws[0] = b;
        ws[1] = 0;
    }
    __syncthreads();
    if (is_e)
        for (unsigned k = 0; k < total; k += TC3_RMAX) {
            const unsigned n = min(TC3_RMAX, total - k), bk = bucket(n);
            unsigned* u = ws + C8_WS_UNITS + 3 * (s_base[bk] + atomicAdd(&s_bucket[bk], 1u));
            u[0] = threadIdx.x; u[1] = s_off[threadIdx.x] + k; u[2] = n;
        }
    if (is_sh) {
        unsigned* u = ws + C8_WS_UNITS + 3 * (s_base[rows] + atomicAdd(&s_bucket[rows], 1u));
        u[0] = C8_SHARED; u[1] = 0; u[2] = rows;
    }
    for (unsigned i = threadIdx.x; i < TC3_UMAX; i += blockDim.x) ws[TC3_WS_DONE + i] = 0;
}

// A warp's weights for one item: 5 super-blocks of its n8 tile, 16 bytes and
// a scale pair a lane each (qwen4exp_moe_tc.cuh's tc_tile layout).
struct Tc3Frag {
    uint4 w[5];
    unsigned short e[5];
};

struct Tc3Smem {
    unsigned unit[TC3_UMAX][3];  // the plan's units: expert, start, n
    unsigned rows[C8_SLOTS_MAX];  // the plan's rows region
    float part[4][2][16][8];
    unsigned short gu[2][16][8];
    unsigned item[2];
};

__device__ __forceinline__ void tc3_load(Tc3Frag& f, const unsigned char* b, const unsigned char* s) {
#pragma unroll
    for (unsigned sb = 0; sb < 5; sb++) {
        f.w[sb] = __ldcs((const uint4*)(b + 64 * sb));
        f.e[sb] = __ldcs((const unsigned short*)(s + 8 * sb));
    }
}

// The item's chain for one m16 tile: super-blocks [0, 5) of the warp's
// fragment at real k = k0 + 128 sb + 32 t + 8 j (act gives the A pairs).
template <typename Act>
__device__ __forceinline__ void tc3_chain(const Tc3Frag& f, unsigned k0, unsigned t, float (&acc)[4], Act&& act) {
#pragma unroll
    for (unsigned sb = 0; sb < 5; sb++) {
        const unsigned q[4] = {f.w[sb].x, f.w[sb].y, f.w[sb].z, f.w[sb].w};
#pragma unroll
        for (unsigned j = 0; j < 4; j++) {
            unsigned a[4], a8[4];
            act(k0 + 128 * sb + 32 * t + 8 * j, a, a8);
            tc_block(acc, q[j], (unsigned char)(f.e[sb] >> (8 * (j >> 1))), a, a8);
        }
    }
}

#define TC3_PASS                                                               \
    A, gate_packed_ptrs, gate_scale_ptrs, gate_scale2_vals, up_packed_ptrs,    \
    up_scale_ptrs, up_scale2_vals, down_packed_ptrs, down_scale_ptrs,          \
    down_scale2_vals, sh_gate_packed, sh_gate_scale, sh_gate_s2, sh_up_packed, \
    sh_up_scale, sh_up_s2, sh_down_packed, sh_down_scale, sh_down_s2, ws, act, \
    C, sh_down_out, top_k, rows
#define TC3_ARGS                                                               \
    const __nv_bfloat16* __restrict__ A,                                       \
    const unsigned long long* __restrict__ gate_packed_ptrs,                   \
    const unsigned long long* __restrict__ gate_scale_ptrs,                    \
    const float* __restrict__ gate_scale2_vals,                                \
    const unsigned long long* __restrict__ up_packed_ptrs,                     \
    const unsigned long long* __restrict__ up_scale_ptrs,                      \
    const float* __restrict__ up_scale2_vals,                                  \
    const unsigned long long* __restrict__ down_packed_ptrs,                   \
    const unsigned long long* __restrict__ down_scale_ptrs,                    \
    const float* __restrict__ down_scale2_vals,                                \
    const unsigned char* __restrict__ sh_gate_packed,                          \
    const unsigned char* __restrict__ sh_gate_scale, float sh_gate_s2,         \
    const unsigned char* __restrict__ sh_up_packed,                            \
    const unsigned char* __restrict__ sh_up_scale, float sh_up_s2,             \
    const unsigned char* __restrict__ sh_down_packed,                          \
    const unsigned char* __restrict__ sh_down_scale, float sh_down_s2,         \
    unsigned int* __restrict__ ws, __nv_bfloat16* __restrict__ act,            \
    __nv_bfloat16* __restrict__ C, __nv_bfloat16* __restrict__ sh_down_out,    \
    unsigned int top_k, unsigned int rows

// The item's kind and unit: Z items [0, R), G [R, R + 80 L), D after.
struct Tc3Item {
    unsigned kind, u, x;  // kind 0 Z, 1 G, 2 D, 3 none
};
__device__ __forceinline__ Tc3Item tc3_item(unsigned i, unsigned L, unsigned R) {
    if (i < R) return {0u, L + i, 0u};
    i -= R;
    if (i < TC3_G * L) return {1u, i / TC3_G, i % TC3_G};
    i -= TC3_G * L;
    if (i < TC3_D * L) return {2u, i / TC3_D, i % TC3_D};
    return {3u, 0u, 0u};
}

template <bool CLAMP>
__device__ __forceinline__ void tc3_body(TC3_ARGS) {
    // 8 warps: 4 K quarters x {gate, up}; D's 64 columns as 8 n8 tiles.
    if (blockDim.x != 256) __trap();
    __shared__ Tc3Smem sm;
    const unsigned slots = rows * top_k;
    const unsigned U = ws[0], L = ws[2], R = U - L;
    // The unit's weight `which` (0/1 gate packed/scale, 2/3 up, 4/5 down)
    // and scale2 (0 gate, 1 up, 2 down), from the tables.
    const auto wt = [&](unsigned e, unsigned which) -> const unsigned char* {
        const bool sh = e == C8_SHARED;
        switch (which) {
            case 0: return sh ? sh_gate_packed : (const unsigned char*)gate_packed_ptrs[e];
            case 1: return sh ? sh_gate_scale : (const unsigned char*)gate_scale_ptrs[e];
            case 2: return sh ? sh_up_packed : (const unsigned char*)up_packed_ptrs[e];
            case 3: return sh ? sh_up_scale : (const unsigned char*)up_scale_ptrs[e];
            case 4: return sh ? sh_down_packed : (const unsigned char*)down_packed_ptrs[e];
            default: return sh ? sh_down_scale : (const unsigned char*)down_scale_ptrs[e];
        }
    };
    const auto s2t = [&](unsigned e, unsigned which) {
        if (e == C8_SHARED) return which == 0 ? sh_gate_s2 : which == 1 ? sh_up_s2 : sh_down_s2;
        return (which == 0 ? gate_scale2_vals : which == 1 ? up_scale2_vals : down_scale2_vals)[e];
    };
    for (unsigned i = threadIdx.x; i < 3 * U; i += blockDim.x) (&sm.unit[0][0])[i] = ws[C8_WS_UNITS + i];
    for (unsigned i = threadIdx.x; i < slots; i += blockDim.x) sm.rows[i] = ws[C8_WS_ROWS + i];
    const unsigned lane = threadIdx.x & 31u, warp = threadIdx.x >> 5, g = lane >> 2, t = lane & 3u;
    const unsigned proj = warp & 1u, kq = warp >> 1;
    unsigned* claim = ws + 1;
    unsigned* done = ws + TC3_WS_DONE;
    const auto wp = [&](unsigned u, unsigned which) { return wt(sm.unit[u][0], which); };
    const auto s2 = [&](unsigned u, unsigned which) { return s2t(sm.unit[u][0], which); };
    // The unit's row i: its slot q (routed) or token (shared).
    const auto slot = [&](unsigned u, unsigned i) {
        return sm.unit[u][0] == C8_SHARED ? i : sm.rows[sm.unit[u][1] + i];
    };
    // Issue the item's weight loads into f (nothing for Z / none).
    const auto fetch = [&](const Tc3Item& it, Tc3Frag& f) {
        if (it.kind == 1) {
            const unsigned n = 8 * it.x + g;
            tc3_load(f, wp(it.u, 2 * proj) + (size_t)n * (C8_H / 2) + 16 * t + 64 * 5 * kq,
                     wp(it.u, 2 * proj + 1) + (size_t)n * (C8_H / 16) + 2 * t + 8 * 5 * kq);
        } else if (it.kind == 2) {
            const unsigned n = 64 * it.x + 8 * warp + g;
            tc3_load(f, wp(it.u, 4) + (size_t)n * (C8_I / 2) + 16 * t, wp(it.u, 5) + (size_t)n * (C8_I / 16) + 2 * t);
        }
    };
    // A remote expert's rows: zeros.
    const auto run_z = [&](const Tc3Item& it) {
        const unsigned n = sm.unit[it.u][2];
        if (C)
            for (unsigned i = threadIdx.x; i < n * (C8_H / 8); i += blockDim.x)
                *(uint4*)(C + (size_t)slot(it.u, i / (C8_H / 8)) * C8_H + 8 * (i % (C8_H / 8))) = make_uint4(0, 0, 0, 0);
    };
    // gate/up + SiLU of the unit's columns [8x, 8x + 8), m16 tile by tile;
    // then the unit's G count, released after every thread's act stores.
    const auto run_g = [&](const Tc3Item& it, const Tc3Frag& f) {
        const unsigned n = sm.unit[it.u][2], n0 = 8 * it.x;
        const bool sh = sm.unit[it.u][0] == C8_SHARED;
        for (unsigned m0 = 0; m0 < n; m0 += 16) {
            const unsigned mt = min(16u, n - m0);
            const bool r0 = g < mt, r1 = g + 8 < mt;
            const unsigned q0 = slot(it.u, m0 + (r0 ? g : 0)), q1 = slot(it.u, m0 + (r1 ? g + 8 : 0));
            const __nv_bfloat16* A0 = A + (size_t)(sh ? q0 : q0 / top_k) * C8_H;
            const __nv_bfloat16* A1 = A + (size_t)(sh ? q1 : q1 / top_k) * C8_H;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            tc3_chain(f, kq * (C8_H / 4), t, acc, [&](unsigned k, unsigned (&a)[4], unsigned (&a8)[4]) {
                const uint4 z = make_uint4(0, 0, 0, 0);
                tc_a8(r0 ? *(const uint4*)(A0 + k) : z, a);
                tc_a8(r1 ? *(const uint4*)(A1 + k) : z, a8);
            });
            sm.part[kq][proj][g][2 * t] = acc[0]; sm.part[kq][proj][g][2 * t + 1] = acc[1];
            sm.part[kq][proj][g + 8][2 * t] = acc[2]; sm.part[kq][proj][g + 8][2 * t + 1] = acc[3];
            __syncthreads();
            {
                const unsigned p = threadIdx.x >> 7, m = (threadIdx.x >> 3) & 15u, j = threadIdx.x & 7u;
                const float v = ((sm.part[0][p][m][j] + sm.part[1][p][m][j]) + sm.part[2][p][m][j]) + sm.part[3][p][m][j];
                sm.gu[p][m][j] = __bfloat16_as_ushort(__float2bfloat16(v * s2(it.u, p)));
            }
            __syncthreads();
            if (threadIdx.x < mt * 8) {
                const unsigned m = threadIdx.x >> 3, j = threadIdx.x & 7u, q = slot(it.u, m0 + m);
                act[(size_t)(sh ? slots + q : q) * C8_I + n0 + j] =
                    __float2bfloat16(tc_silu_up<CLAMP>(sm.gu[0][m][j], sm.gu[1][m][j], !sh));
            }
        }
        __threadfence();
        __syncthreads();
        if (threadIdx.x == 0) atomicAdd(done + it.u, 1u);
    };
    // down of the unit's columns [64x, 64x + 64): warp w the n8 tile 8w,
    // once the unit's 80 G items are done (acquire).
    const auto run_d = [&](const Tc3Item& it, const Tc3Frag& f) {
        if (threadIdx.x == 0) {
            unsigned c;
            while (true) {
                asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(c) : "l"(done + it.u) : "memory");
                if (c >= TC3_G) break;
                __nanosleep(100);
            }
        }
        __syncthreads();
        const unsigned n = sm.unit[it.u][2];
        const bool sh = sm.unit[it.u][0] == C8_SHARED;
        const float sc = s2(it.u, 2);
        __nv_bfloat16* out = sh ? sh_down_out : C;
        const unsigned col = 64 * it.x + 8 * warp + 2 * t;
        for (unsigned m0 = 0; m0 < n; m0 += 16) {
            const unsigned mt = min(16u, n - m0);
            const bool r0 = g < mt, r1 = g + 8 < mt;
            const unsigned q0 = slot(it.u, m0 + (r0 ? g : 0)), q1 = slot(it.u, m0 + (r1 ? g + 8 : 0));
            const __nv_bfloat16* X0 = act + (size_t)(sh ? slots + q0 : q0) * C8_I;
            const __nv_bfloat16* X1 = act + (size_t)(sh ? slots + q1 : q1) * C8_I;
            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            tc3_chain(f, 0, t, acc, [&](unsigned k, unsigned (&a)[4], unsigned (&a8)[4]) {
                const uint4 z = make_uint4(0, 0, 0, 0);
                tc_a8(r0 ? *(const uint4*)(X0 + k) : z, a);
                tc_a8(r1 ? *(const uint4*)(X1 + k) : z, a8);
            });
            if (r0) *(unsigned*)(out + (size_t)q0 * C8_H + col) = tc_bf16x2(acc[0] * sc, acc[1] * sc);
            if (r1) *(unsigned*)(out + (size_t)q1 * C8_H + col) = tc_bf16x2(acc[2] * sc, acc[3] * sc);
        }
    };
    const auto run = [&](const Tc3Item& it, const Tc3Frag& f) {
        if (it.kind == 0) run_z(it);
        else if (it.kind == 1) run_g(it, f);
        else run_d(it, f);
    };
    // Claims in order; weights one item ahead (registers f0 / f1).
    const auto next = [&](unsigned b) {
        if (threadIdx.x == 0) sm.item[b] = atomicAdd(claim, 1u);
        __syncthreads();
        return tc3_item(sm.item[b], L, R);
    };
    Tc3Frag f0, f1;
    Tc3Item cur = next(0);
    fetch(cur, f0);
    while (cur.kind != 3) {
        Tc3Item nx = next(1);
        fetch(nx, f1);
        run(cur, f0);
        __syncthreads();
        if (nx.kind == 3) break;
        cur = next(0);
        fetch(cur, f0);
        run(nx, f1);
        __syncthreads();
    }
}

extern "C" __global__ void __launch_bounds__(256, 2) qwen4exp_moe_c8_tc3(TC3_ARGS) {
    tc3_body<true>(TC3_PASS);
}
extern "C" __global__ void __launch_bounds__(256, 2) qwen4exp_moe_c8_tc3_nc(TC3_ARGS) {
    tc3_body<false>(TC3_PASS);
}
