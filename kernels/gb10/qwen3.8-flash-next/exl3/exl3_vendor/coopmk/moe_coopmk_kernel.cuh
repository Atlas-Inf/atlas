// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from vcruz305/exllamav3 @ 047ce72 (fork of turboderp-org/exllamav3, MIT,
// Copyright (c) 2025 Turboderp); see ../NOTICE.md.
// Source: exllamav3_ext/quant/exl3_moe_coopmk_kernel.cuh (vcruz305 9aa5329/047ce72). Modifications:
// coopmk_a_kernel / coopmk_b_kernel are __device__ __forceinline__ bodies instead of __global__
// templates (Atlas resolves kernels by unmangled name, so exl3_coopmk.cu wraps each instance in an
// extern "C" __global__ with the same __launch_bounds__(THREADS, MINB)); the bsz > 1 rotation
// pre-kernel (COOPMK_DEFINE_ROT) is vendored as the __device__ body coopmk_rot_body, wrapped by
// exl3_coopmk_rot in exl3_coopmk.cu; include path. Kernel bodies are verbatim.
#pragma once

// Per-expert runtime-K variant of the fused decode MoE kernels (exl3_moe_coop_kernel.cuh) for
// layers whose experts are quantized at different integer bitrates (mixed-K packs).
//
// Everything that does not depend on K is the uniform kernel's own code: run building, slot
// info, split-k, completion counters, the chunk epilogues (rotation, svh, activation, down suh,
// fixed-order reduction) and the per-expert pointer tables. The only change is the k-loop: each
// block works on exactly one expert run, reads that expert's bitrate from a device table and
// switches to the matching gemv_tile<K> instantiation. The switch is block-uniform, so there is
// no divergence, and no host synchronization is needed: the tables live on the device.
//
// KMASK selects the bitrates a kernel instance contains (bit K set = K compiled in). A block
// whose expert's K is outside the mask exits at once, so a layer can be covered either by one
// all-K launch per stage or by a few launches over disjoint K ranges (smaller register
// footprint per instance). Blocks of a K the instance does not contain do no work and touch no
// counters, so launches over disjoint masks compose into exactly the uniform kernel's result.
//
// With a single-K layer and a mask containing that K, the arithmetic is identical to the
// uniform kernel's (same gemv_tile body, same launch geometry), so outputs are bit-identical.

#include "moe_coop_kernel.cuh"

namespace exl3_coopmk_ns {

using namespace exl3_moe_coop_ns;

// Bitrate masks (bit K = K bits per weight). REG: the widths the uniform kernel decodes in
// registers on sm_8x/sm_12x (tile_reg); STG: the staged (shared-memory) widths
constexpr uint32_t KM_ALL = 0x1FEu;                                   // K 1..8
constexpr uint32_t KM_REG = (1u << 2) | (1u << 3) | (1u << 4);         // K 2, 3, 4
constexpr uint32_t KM_STG = KM_ALL & ~KM_REG;                          // K 1, 5, 6, 7, 8

// Dynamic shared memory: the staging area is always reserved (any mask may contain a staged K)
__host__ __device__ constexpr int smem_a_bytes_mk(int Hi) { return Hi * 2 + smem_red_bytes() + WK * STAGE_WORDS * 4; }
__host__ __device__ constexpr int smem_b_bytes_mk() { return smem_red_bytes() + WK * STAGE_WORDS * 4 + smem_part_bytes(); }

// Runtime-K GEMV tile: dispatch on the block's expert bitrate
template <int cb, bool WIDE, uint32_t KMASK>
__device__ __forceinline__ void gemv_tile_rk
(
    int kb,
    const uint32_t* __restrict__ B32,
    const half2* __restrict__ A2,
    size_t a_stride2,
    const int* __restrict__ rows,
    int nrows,
    void* __restrict__ C,
    size_t c_stride,
    bool c_f32,
    int k_begin,
    int k_end,
    int ntiles,
    int group,
    float* __restrict__ sh_red,
    uint32_t* __restrict__ sh_stage
)
{
    #define COOPMK_CASE(K) \
        case K: \
            if constexpr (((KMASK >> K) & 1u) != 0u) \
                gemv_tile<K, cb, WIDE, false>(B32, A2, a_stride2, rows, nrows, C, c_stride, c_f32, \
                                              k_begin, k_end, ntiles, group, sh_red, sh_stage); \
            break;
    switch (kb)
    {
        COOPMK_CASE(1)
        COOPMK_CASE(2)
        COOPMK_CASE(3)
        COOPMK_CASE(4)
        COOPMK_CASE(5)
        COOPMK_CASE(6)
        COOPMK_CASE(7)
        COOPMK_CASE(8)
        default: break;
    }
    #undef COOPMK_CASE
}

__device__ __forceinline__ bool in_mask(uint32_t mask, int kb)
{
    return kb >= 1 && kb <= 8 && ((mask >> kb) & 1u);
}

// ---------------------------------------------------------------------------------------------
// Rotation pre-kernel (bsz > 1): upstream coopmk_rot_kernel, verbatim body. Block 0 also builds
// the run table (p.runs) that kernels A and B read when p.a_global is set.

__device__ __forceinline__ void coopmk_rot_body(const MoeCoopParams p)
{
    __shared__ int16_t sh_order[MAX_SLOTS];
    __shared__ int16_t sh_local[MAX_SLOTS];
    __shared__ int sh_scan[WK];
    __shared__ int sh_res[4];
    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int chunks = p.Hi / 128;
    const int slots = p.bsz * p.topk;
    if (blockIdx.x == 0) build_runs(p, sh_order, sh_local, sh_scan, sh_res);
    const int item = blockIdx.x * (THREADS / 32) + warp;
    if (item >= slots * chunks * nproj) return;
    const int s = item / (chunks * nproj);
    const int rem = item % (chunks * nproj);
    const int c = rem / nproj;
    const bool is_gate = p.gated && (rem % nproj) == 0;
    const SlotInfo si = slot_info(p, s);
    if (!si.active)
    {
        if (rem % nproj == 0 && s % p.topk == 0 && token_active_slots(p, si.row) == 0)
            for (int oc = c; oc < p.Ho / 128; oc += chunks)
                write_empty_row_chunk(p, si.row, oc, lane);
        return;
    }
    const half* suh = (const half*) (is_gate ? p.g_suh : p.u_suh)[si.local];
    half* dst = (is_gate ? p.had_g : p.had_u) + (size_t) s * p.Hi;
    rotate_chunk(p, si.row, suh, c, dst, lane);
}

// ---------------------------------------------------------------------------------------------
// Kernel A (gate/up): exl3_moe_coop_a_kernel with the k-loop dispatched on the block's
// projection bitrate (k_tab row 0 gate, row 1 up; gate and up may differ)

template <int cb, bool WIDE, uint32_t KMASK, int MINB>
__device__ __forceinline__
void coopmk_a_kernel(const MoeCoopParams p, const int* __restrict__ k_tab)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    extern __shared__ uint32_t smem_dyn[];
    half* sh_A = (half*) smem_dyn;
    float* sh_red = (float*) (smem_dyn + p.Hi / 2);
    uint32_t* sh_stage = smem_dyn + p.Hi / 2 + WK * ROWS * COLS;
    __shared__ int sh_flag;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int nproj = p.gated ? 2 : 1;
    const int ng = p.I / TCOLS;

    // Reset the down stage's counters for this call (its previous use finished in stream order).
    // Every A launch of a split plan does this; all of them complete before any B launch
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_b_len; i += gridDim.x * THREADS)
        p.ctr_b[i] = 0;

    const int item = blockIdx.x;
    const bool is_gate = p.gated && (item % nproj) == 0;
    const int rem = item / nproj;
    const int ks = rem % p.ksplit_a;
    const int rem2 = rem / p.ksplit_a;
    const int run_idx = rem2 / ng;
    const int group = rem2 % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res))
    {
        // bsz 1: empty token rows (idempotent when several A launches cover the layer)
        if (!p.a_global && run_idx < p.bsz * p.topk && !is_gate && ks == 0 && run_idx % p.topk == 0)
        {
            const int row = run_idx / p.topk;
            if (token_active_slots(p, row) == 0)
                for (int oc = group + warp * ng; oc < p.Ho / 128; oc += ng * WK)
                    write_empty_row_chunk(p, row, oc, lane);
        }
        return;
    }
    const int local = slot_info(p, sh_rows[0]).local;
    const int kb = __ldg(k_tab + (is_gate ? 0 : p.n_local) + local);   // k_tab: [K_gate; K_up; K_down]
    if (!in_mask(KMASK, kb)) return;          // block-uniform: another launch covers this expert
    const int kslices_all = p.Hi / 16;
    const int k_begin = kslices_all * ks / p.ksplit_a;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_a;

    const half* suh = (const half*) (is_gate ? p.g_suh : p.u_suh)[local];
    const half2* A2;
    size_t a_stride2;
    if (p.a_global)
    {
        A2 = (const half2*) (is_gate ? p.had_g : p.had_u);
        a_stride2 = (size_t) p.Hi / 2;
    }
    else
    {
        const SlotInfo si = slot_info(p, sh_rows[0]);
        for (int c = warp; c < p.Hi / 128; c += WK)
            rotate_chunk(p, si.row, suh, c, sh_A, lane);
        __syncthreads();
        A2 = (const half2*) sh_A;
        a_stride2 = 0;
    }

    {
        const uint32_t* B32 = (const uint32_t*) (is_gate ? p.g_trellis : p.u_trellis)[local];
        const size_t part = (size_t) ks * (p.bsz * p.topk) * p.I * (p.gu_f32 ? 4 : 2);
        void* C = (void*) (((char*) (is_gate ? p.gu_g : p.gu_u)) + part);
        gemv_tile_rk<cb, WIDE, KMASK>(kb, B32, A2, a_stride2, sh_rows, nrows, C, p.I, p.gu_f32, k_begin, k_end, p.I / 16, group, sh_red, sh_stage);
    }

    for (int r = 0; r < nrows; ++r)
    for (int j = 0; j < CPB; ++j)
    {
        const int s = sh_rows[r];
        const int chunk = (group * CPB + j) / GPC;
        if (!arrive_last(p.ctr_a + (size_t) s * (p.I / 128) + chunk, GPC * nproj * p.ksplit_a, &sh_flag)) continue;
        if (p.dbg & 1) continue;
        if (warp == 0)
        {
            const int col = chunk * 128 + lane * 4;
            const size_t off = (size_t) s * p.I + col;
            const size_t pstride = (size_t) (p.bsz * p.topk) * p.I;

            auto load_sum = [&] (const void* buf, float& v0, float& v1, float& v2, float& v3)
            {
                v0 = v1 = v2 = v3 = 0.0f;
                for (int q = 0; q < p.ksplit_a; ++q)
                {
                    float t0, t1, t2, t3;
                    if (p.gu_f32) load_f4_cg(((const float*) buf) + q * pstride + off, t0, t1, t2, t3);
                    else          load_h4_cg(((const half*) buf) + q * pstride + off, t0, t1, t2, t3);
                    v0 += t0; v1 += t1; v2 += t2; v3 += t3;
                }
            };
            float u0, u1, u2, u3;
            load_sum(p.gu_u, u0, u1, u2, u3);
            had128(u0, u1, u2, u3, lane);
            scale_h4(((const half*) p.u_svh[local]) + col, u0, u1, u2, u3);
            if (p.u_bias) add_h4(((const half*) p.u_bias[local]) + col, u0, u1, u2, u3);

            float g0 = u0, g1 = u1, g2 = u2, g3 = u3;
            if (p.gated)
            {
                load_sum(p.gu_g, g0, g1, g2, g3);
                had128(g0, g1, g2, g3, lane);
                scale_h4(((const half*) p.g_svh[local]) + col, g0, g1, g2, g3);
                if (p.g_bias) add_h4(((const half*) p.g_bias[local]) + col, g0, g1, g2, g3);
            }

            float a0 = act_gate(p.act, p.gated, g0, u0, p.act_limit);
            float a1 = act_gate(p.act, p.gated, g1, u1, p.act_limit);
            float a2 = act_gate(p.act, p.gated, g2, u2, p.act_limit);
            float a3 = act_gate(p.act, p.gated, g3, u3, p.act_limit);

            scale_h4(((const half*) p.d_suh[local]) + col, a0, a1, a2, a3);
            had128(a0, a1, a2, a3, lane);
            store_h4(p.act_out + off, a0, a1, a2, a3);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// Kernel B (down): exl3_moe_coop_b_kernel with the k-loop dispatched on k_tab row 2

template <int cb, bool WIDE, uint32_t KMASK, int MINB>
__device__ __forceinline__
void coopmk_b_kernel(const MoeCoopParams p, const int* __restrict__ k_tab)
{
    constexpr int TCOLS = tile_cols<WIDE>();
    constexpr int GPC = tile_gpc<WIDE>();
    constexpr int CPB = tile_cpb<WIDE>();
    extern __shared__ uint32_t smem_dyn[];
    float* sh_red = (float*) smem_dyn;
    uint32_t* sh_stage = smem_dyn + WK * ROWS * COLS;
    float* sh_part = (float*) (smem_dyn + WK * ROWS * COLS + WK * STAGE_WORDS);
    __shared__ int sh_flag;
    __shared__ int sh_n_active;
    __shared__ float sh_gate;
    __shared__ int sh_res[4];
    __shared__ int sh_rows[ROWS];

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int ng = p.Ho / TCOLS;

    // Reset the gate/up stage's counters for the next call (every A launch has finished)
    for (int i = blockIdx.x * THREADS + threadIdx.x; i < p.ctr_a_len; i += gridDim.x * THREADS)
        p.ctr_a[i] = 0;

    const int item = blockIdx.x;
    const int ks = item % p.ksplit_b;
    const int rem = item / p.ksplit_b;
    const int run_idx = rem / ng;
    const int group = rem % ng;
    int nrows = 0;
    if (!read_run(p, run_idx, sh_rows, nrows, sh_res)) return;
    const int local = slot_info(p, sh_rows[0]).local;
    const int kb = __ldg(k_tab + 2 * p.n_local + local);
    if (!in_mask(KMASK, kb)) return;          // block-uniform: another launch covers this expert
    const int kslices_all = p.I / 16;
    const int k_begin = kslices_all * ks / p.ksplit_b;
    const int k_end = kslices_all * (ks + 1) / p.ksplit_b;

    {
        const uint32_t* B32 = (const uint32_t*) p.d_trellis[local];
        float* C = p.d_out + (size_t) ks * (p.bsz * p.topk) * p.Ho;
        gemv_tile_rk<cb, WIDE, KMASK>(kb, B32, (const half2*) p.act_out, (size_t) p.I / 2, sh_rows, nrows, C, p.Ho, true,
                                      k_begin, k_end, p.Ho / 16, group, sh_red, sh_stage);
    }

    for (int r = 0; r < nrows; ++r)
    for (int j = 0; j < CPB; ++j)
    {
        const int s = sh_rows[r];
        const int row = s / p.topk;
        const int chunk = (group * CPB + j) / GPC;

        if (threadIdx.x == 0)
        {
            int n = 0;
            for (int k = 0; k < p.topk; ++k)
                n += slot_info(p, row * p.topk + k).active ? 1 : 0;
            sh_n_active = n;
        }
        __syncthreads();
        if (!arrive_last(p.ctr_b + (size_t) row * (p.Ho / 128) + chunk, sh_n_active * GPC * p.ksplit_b, &sh_flag)) continue;
        if (p.dbg & 2) continue;

        const int col = chunk * 128 + lane * 4;

        if (p.sh_gate_w)
        {
            const half* xr = p.x + (size_t) row * p.x_stride;
            float dot = 0.0f;
            for (int i = threadIdx.x * 2; i < p.H; i += THREADS * 2)
            {
                half2 xv = *((const half2*) (xr + i));
                half2 wv = *((const half2*) (p.sh_gate_w + i));
                dot += __low2float(xv) * __low2float(wv) + __high2float(xv) * __high2float(wv);
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                dot += __shfl_xor_sync(0xffffffffu, dot, o);
            if (lane == 0) sh_part[warp] = dot;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                float tt = 0.0f;
                for (int w = 0; w < WK; ++w) tt += sh_part[w];
                sh_gate = 1.0f / (1.0f + __expf(-tt));
            }
            __syncthreads();
        }

        float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
        for (int k = warp; k < p.topk; k += WK)
        {
            const int sk = row * p.topk + k;
            const SlotInfo sj = slot_info(p, sk);
            if (!sj.active) continue;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            for (int q = 0; q < p.ksplit_b; ++q)
            {
                float t0, t1, t2, t3;
                load_f4_cg(p.d_out + ((size_t) q * (p.bsz * p.topk) + sk) * p.Ho + col, t0, t1, t2, t3);
                v0 += t0; v1 += t1; v2 += t2; v3 += t3;
            }
            had128(v0, v1, v2, v3, lane);
            scale_h4(((const half*) p.d_svh[sj.local]) + col, v0, v1, v2, v3);
            if (p.d_bias) add_h4(((const half*) p.d_bias[sj.local]) + col, v0, v1, v2, v3);
            o0 += sj.w * v0;
            o1 += sj.w * v1;
            o2 += sj.w * v2;
            o3 += sj.w * v3;
        }
        float* part = sh_part + warp * 128 + lane * 4;
        part[0] = o0; part[1] = o1; part[2] = o2; part[3] = o3;
        __syncthreads();
        if (warp == 0)
        {
            o0 = 0.0f; o1 = 0.0f; o2 = 0.0f; o3 = 0.0f;
            #pragma unroll
            for (int w = 0; w < WK; ++w)
            {
                const float* q = sh_part + w * 128 + lane * 4;
                o0 += q[0]; o1 += q[1]; o2 += q[2]; o3 += q[3];
            }

            if (p.sh_out)
            {
                const float gv = p.sh_gate_w ? sh_gate : 1.0f;
                const float* sh = p.sh_out + (size_t) row * p.H + col;
                if (col + 0 < p.H) o0 += gv * sh[0];
                if (col + 1 < p.H) o1 += gv * sh[1];
                if (col + 2 < p.H) o2 += gv * sh[2];
                if (col + 3 < p.H) o3 += gv * sh[3];
            }

            float* dst = p.out + (size_t) row * p.out_stride + col;
            if (col + 3 < p.H_out)
                *((float4*) dst) = make_float4(o0, o1, o2, o3);
            else
            {
                if (col + 0 < p.H_out) dst[0] = o0;
                if (col + 1 < p.H_out) dst[1] = o1;
                if (col + 2 < p.H_out) dst[2] = o2;
            }
        }
        __syncthreads();
    }
}

}  // namespace exl3_coopmk_ns
