// SPDX-License-Identifier: AGPL-3.0-only
// Ling MLA prompt attention over one key tile of expanded rows
// (ling_latent_expand): K = 32 x 128 NoPE channels then the shared 64-channel
// RoPE tail, V = 32 x 128 channels, both [keys][32 * 128 + 64] BF16. Queries
// are q_full rows, [rows][32][192]; outputs are compact, [rows][32][128].
//
// 128 query rows a CTA, laid out as prefill_paged_rows128.cu (SparkQwen):
// every warp owns 16 rows end to end (Q fragments, S, softmax, O), P goes from
// the S accumulators straight into the PV fragments, K and V are double
// buffered and read by ldmatrix, one barrier per key tile. QK^T is 192 wide;
// PV is 128 wide, where the paged BR=64 kernel this replaces padded V to 192.
//
// Per row the key tiles, the order of every MMA, the online-softmax updates
// and every rounding are the BR=64 kernel's, so the bytes are its bytes
// (tests/ling/mla_prefill.cu). A taller tile only adds key tiles fully masked
// for its earlier rows, which leave m, l and O unchanged.
//
// Grid (32, ceil(q_len / 128)), block 256, CTAs ordered head-major so a
// head's query tiles reuse its keys in L2. ling_mla_prefill_rows128 writes
// BF16 output; the _part entry writes FP32 output and each row's softmax
// (max, sum) for ling_prompt_tile_merge.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace ling_prefill {
constexpr unsigned HEADS = 32, NOPE = 128, QK = 192, VD = 128;
constexpr unsigned ROW = HEADS * NOPE + 64;   // expanded row width
constexpr unsigned BC = 32;                   // key tile
constexpr unsigned BR = 128;                  // query rows a CTA
constexpr unsigned KP = QK + 8, VP = VD + 8;  // smem pitches: 16-byte rows, conflict-free ldmatrix
constexpr unsigned NT_O = VD / 8;             // output n-tiles a warp
constexpr unsigned K_CHUNKS = BC * QK / 8, V_CHUNKS = BC * VD / 8;
static_assert(K_CHUNKS % 256 == 0 && V_CHUNKS % 256 == 0, "whole rounds of 256 threads");

__device__ __forceinline__ void cp16(void* dst, const void* src) {
    const unsigned s = (unsigned)__cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group 0;"); }
__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
    const unsigned a = (unsigned)__cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3},[%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
// The BR=64 kernel's bf16 pair -> fp16 pair, through float.
__device__ __forceinline__ uint32_t bf16w_to_f16x2(uint32_t w) {
    const __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(&w);
    __half2 h2 = __floats2half2_rn(__bfloat162float(__low2bfloat16(v)), __bfloat162float(__high2bfloat16(v)));
    return *reinterpret_cast<const uint32_t*>(&h2);
}
__device__ __forceinline__ uint32_t f16x2(float lo, float hi) {
    return (uint32_t)__half_as_ushort(__float2half_rn(lo)) | ((uint32_t)__half_as_ushort(__float2half_rn(hi)) << 16);
}

// Key tile [kv_s, kv_s + BC) of `head` into smem; rows past kv_l read as zero.
__device__ __forceinline__ void load_k(__nv_bfloat16 (*dst)[KP], const __nv_bfloat16* k, unsigned kv_s,
                                       unsigned kv_l, unsigned head, unsigned tid) {
#pragma unroll
    for (unsigned j = 0; j < K_CHUNKS / 256; j++) {
        const unsigned i = tid + 256 * j, row = i / (QK / 8), col = (i % (QK / 8)) * 8, p = kv_s + row;
        if (p < kv_l)
            cp16(&dst[row][col],
                 k + (unsigned long long)p * ROW + (col < NOPE ? head * NOPE + col : HEADS * NOPE + col - NOPE));
        else
            *reinterpret_cast<uint4*>(&dst[row][col]) = make_uint4(0, 0, 0, 0);
    }
}
__device__ __forceinline__ void load_v(__nv_bfloat16 (*dst)[VP], const __nv_bfloat16* v, unsigned kv_s,
                                       unsigned kv_l, unsigned head, unsigned tid) {
#pragma unroll
    for (unsigned j = 0; j < V_CHUNKS / 256; j++) {
        const unsigned i = tid + 256 * j, row = i / (VD / 8), col = (i % (VD / 8)) * 8, p = kv_s + row;
        if (p < kv_l)
            cp16(&dst[row][col], v + (unsigned long long)p * ROW + head * NOPE + col);
        else
            *reinterpret_cast<uint4*>(&dst[row][col]) = make_uint4(0, 0, 0, 0);
    }
}
// V to FP16 in place, over exactly the chunks load_v gave this thread (its
// own cp.async writes are visible to it after its own wait).
__device__ __forceinline__ void v_to_f16(__nv_bfloat16 (*v)[VP], unsigned tid) {
#pragma unroll
    for (unsigned j = 0; j < V_CHUNKS / 256; j++) {
        const unsigned i = tid + 256 * j;
        uint4* c = reinterpret_cast<uint4*>(&v[i / (VD / 8)][(i % (VD / 8)) * 8]);
        uint4 w = *c;
        w.x = bf16w_to_f16x2(w.x); w.y = bf16w_to_f16x2(w.y);
        w.z = bf16w_to_f16x2(w.z); w.w = bf16w_to_f16x2(w.w);
        *c = w;
    }
}

template <bool PART>
__device__ __forceinline__ void body(const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K,
                                     const __nv_bfloat16* __restrict__ V, void* __restrict__ O,
                                     float* __restrict__ ml, unsigned q_len, unsigned kv_len, unsigned q_offset,
                                     float inv_sqrt_d) {
    const unsigned tile = blockIdx.y * gridDim.x + blockIdx.x;
    const unsigned head = tile / gridDim.y, q_block = tile % gridDim.y;
    const unsigned q_start = q_block * BR;
    if (head >= HEADS || q_start >= q_len) return;
    const unsigned q_tile_len = min(q_start + BR, q_len) - q_start;
    const unsigned tid = threadIdx.x, warp = tid / 32, lane = tid % 32, grp = lane >> 2, tq = lane & 3;

    __shared__ __align__(16) __nv_bfloat16 sk[2][BC][KP];
    __shared__ __align__(16) __nv_bfloat16 sv[2][BC][VP];

    // The tile's causal key extent, as the BR=64 kernel derives it.
    const unsigned num_kv_blocks = min((kv_len + BC - 1) / BC, (q_offset + q_start + q_tile_len - 1) / BC + 1);
    const unsigned wm = warp * 16, row0 = wm + grp, row1 = row0 + 8;
    const bool active = wm < q_tile_len;

    if (num_kv_blocks > 0) {
        load_k(sk[0], K, 0, kv_len, head, tid);
        load_v(sv[0], V, 0, kv_len, head, tid);
    }
    cp_commit();

    // Q A-fragments of the warp's 16 rows (rows past the tile read as 0).
    uint32_t qa[QK / 16][4];
    {
        const bool v0 = row0 < q_tile_len, v1 = row1 < q_tile_len;
        const __nv_bfloat16* q0 = Q + (unsigned long long)(q_start + row0) * HEADS * QK + head * QK + 2 * tq;
        const __nv_bfloat16* q1 = Q + (unsigned long long)(q_start + row1) * HEADS * QK + head * QK + 2 * tq;
#pragma unroll
        for (unsigned ks = 0; ks < QK / 16; ks++) {
            qa[ks][0] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + ks * 16) : 0u;
            qa[ks][1] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + ks * 16) : 0u;
            qa[ks][2] = v0 ? *reinterpret_cast<const uint32_t*>(q0 + ks * 16 + 8) : 0u;
            qa[ks][3] = v1 ? *reinterpret_cast<const uint32_t*>(q1 + ks * 16 + 8) : 0u;
        }
    }
    cp_wait();
    v_to_f16(sv[0], tid);

    float acc_o[NT_O][4];
#pragma unroll
    for (unsigned i = 0; i < NT_O; i++) acc_o[i][0] = acc_o[i][1] = acc_o[i][2] = acc_o[i][3] = 0.0f;
    float m_r0 = -1e30f, m_r1 = -1e30f, l_r0 = 0.0f, l_r1 = 0.0f;
    const unsigned qr0 = q_offset + q_start + row0, qr1 = q_offset + q_start + row1;

    for (unsigned kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        const unsigned buf = kv_block & 1, kv_start = kv_block * BC;
        const unsigned kv_tile_len = min(kv_start + BC, kv_len) - kv_start;
        __syncthreads();   // tile kv_block is in buf (V as FP16); buf ^ 1 is free
        const bool next = kv_block + 1 < num_kv_blocks;
        if (next) {
            load_k(sk[buf ^ 1], K, kv_start + BC, kv_len, head, tid);
            load_v(sv[buf ^ 1], V, kv_start + BC, kv_len, head, tid);
        }
        cp_commit();
        if (active) {
            // S = Q K^T, 16 rows x 32 keys. ldmatrix x4 j holds keys 8 (j >> 1)..
            // and dims 8 (j & 1).. of an n-tile pair: b0 / b1 of n-tiles 2 np, 2 np + 1.
            float acc_s[4][4];
#pragma unroll
            for (unsigned i = 0; i < 4; i++) acc_s[i][0] = acc_s[i][1] = acc_s[i][2] = acc_s[i][3] = 0.0f;
            const unsigned lrow = (lane & 7) + ((lane >> 4) << 3), lcol = ((lane >> 3) & 1) * 8;
#pragma unroll
            for (unsigned ks = 0; ks < QK / 16; ks++) {
                uint32_t kf[2][4];
#pragma unroll
                for (unsigned np = 0; np < 2; np++) ldsm_x4(kf[np], &sk[buf][np * 16 + lrow][ks * 16 + lcol]);
#pragma unroll
                for (unsigned nt = 0; nt < 4; nt++) {
                    const uint32_t b0 = kf[nt >> 1][(nt & 1) * 2], b1 = kf[nt >> 1][(nt & 1) * 2 + 1];
                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]), "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(qa[ks][0]), "r"(qa[ks][1]), "r"(qa[ks][2]), "r"(qa[ks][3]), "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]), "f"(acc_s[nt][2]), "f"(acc_s[nt][3]));
                }
            }

            // Causal, key-count and row-count masks; a tile none can touch skips them.
            const bool full = kv_start + BC <= kv_len && wm + 16 <= q_tile_len &&
                              kv_start + BC - 1 <= q_offset + q_start + wm;
#pragma unroll
            for (unsigned nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d; acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d; acc_s[nt][3] *= inv_sqrt_d;
                if (full) continue;
                const unsigned c0 = nt * 8 + tq * 2, c1 = c0 + 1;
                if (kv_start + c0 > qr0) acc_s[nt][0] = -1e30f;
                if (kv_start + c1 > qr0) acc_s[nt][1] = -1e30f;
                if (kv_start + c0 > qr1) acc_s[nt][2] = -1e30f;
                if (kv_start + c1 > qr1) acc_s[nt][3] = -1e30f;
                if (c0 >= kv_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (c1 >= kv_tile_len) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_tile_len) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_tile_len) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            // Online softmax, the BR=64 kernel's row for row.
            float rmax0 = -1e30f, rmax1 = -1e30f;
#pragma unroll
            for (unsigned nt = 0; nt < 4; nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));
            const float mn0 = fmaxf(m_r0, rmax0);
            if (mn0 != m_r0) {
                const float eo0 = __expf(m_r0 - mn0);
                l_r0 *= eo0;
#pragma unroll
                for (unsigned i = 0; i < NT_O; i++) { acc_o[i][0] *= eo0; acc_o[i][1] *= eo0; }
                m_r0 = mn0;
            }
            const float mn1 = fmaxf(m_r1, rmax1);
            if (mn1 != m_r1) {
                const float eo1 = __expf(m_r1 - mn1);
                l_r1 *= eo1;
#pragma unroll
                for (unsigned i = 0; i < NT_O; i++) { acc_o[i][2] *= eo1; acc_o[i][3] *= eo1; }
                m_r1 = mn1;
            }
            float sum0 = 0, sum1 = 0;
            uint32_t pa[4][2];   // P as FP16 pairs: [n-tile][row0 | row1]
#pragma unroll
            for (unsigned nt = 0; nt < 4; nt++) {
                const float p00 = __expf(acc_s[nt][0] - m_r0), p01 = __expf(acc_s[nt][1] - m_r0);
                const float p10 = __expf(acc_s[nt][2] - m_r1), p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;
                pa[nt][0] = f16x2(p00, p01);
                pa[nt][1] = f16x2(p10, p11);
            }
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1); sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1); sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            // O += P V in FP16. ldmatrix.trans x4 j holds keys 16 ks + 8 (j & 1)..
            // and dims 8 (2 np + (j >> 1)).. : b0 / b1 of n-tiles 2 np and 2 np + 1.
            const unsigned vcol = (lane >> 4) * 8;
#pragma unroll
            for (unsigned ks = 0; ks < 2; ks++) {
                const uint32_t a0 = pa[2 * ks][0], a1 = pa[2 * ks][1], a2 = pa[2 * ks + 1][0], a3 = pa[2 * ks + 1][1];
                const unsigned vrow = ks * 16 + ((lane >> 3) & 1) * 8 + (lane & 7);
#pragma unroll
                for (unsigned np = 0; np < NT_O / 2; np++) {
                    uint32_t vf[4];
                    ldsm_x4_t(vf, &sv[buf][vrow][np * 16 + vcol]);
#pragma unroll
                    for (unsigned h = 0; h < 2; h++) {
                        const unsigned nt = 2 * np + h;
                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                            "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
                            : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]), "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(vf[2 * h]), "r"(vf[2 * h + 1]),
                              "f"(acc_o[nt][0]), "f"(acc_o[nt][1]), "f"(acc_o[nt][2]), "f"(acc_o[nt][3]));
                    }
                }
            }
        }
        if (next) {   // the next tile's V, converted before the barrier that publishes it
            cp_wait();
            v_to_f16(sv[buf ^ 1], tid);
        }
    }

    if (!active) return;
    const float il0 = (l_r0 > 0) ? (1.f / l_r0) : 0, il1 = (l_r1 > 0) ? (1.f / l_r1) : 0;
    const unsigned at0 = (q_start + row0) * HEADS + head, at1 = (q_start + row1) * HEADS + head;
#pragma unroll
    for (unsigned nt = 0; nt < NT_O; nt++) {
        const unsigned c0 = nt * 8 + tq * 2;
        if constexpr (PART) {
            float* o = static_cast<float*>(O);
            if (row0 < q_tile_len)
                *reinterpret_cast<float2*>(&o[at0 * VD + c0]) = make_float2(acc_o[nt][0] * il0, acc_o[nt][1] * il0);
            if (row1 < q_tile_len)
                *reinterpret_cast<float2*>(&o[at1 * VD + c0]) = make_float2(acc_o[nt][2] * il1, acc_o[nt][3] * il1);
        } else {
            __nv_bfloat16* o = static_cast<__nv_bfloat16*>(O);
            if (row0 < q_tile_len)
                *reinterpret_cast<__nv_bfloat162*>(&o[at0 * VD + c0]) =
                    __halves2bfloat162(__float2bfloat16(acc_o[nt][0] * il0), __float2bfloat16(acc_o[nt][1] * il0));
            if (row1 < q_tile_len)
                *reinterpret_cast<__nv_bfloat162*>(&o[at1 * VD + c0]) =
                    __halves2bfloat162(__float2bfloat16(acc_o[nt][2] * il1), __float2bfloat16(acc_o[nt][3] * il1));
        }
    }
    if (PART && tq == 0) {
        if (row0 < q_tile_len) { ml[at0 * 2] = m_r0; ml[at0 * 2 + 1] = l_r0; }
        if (row1 < q_tile_len) { ml[at1 * 2] = m_r1; ml[at1 * 2 + 1] = l_r1; }
    }
}
}  // namespace ling_prefill

extern "C" __global__ void __launch_bounds__(256, 1) ling_mla_prefill_rows128(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
    __nv_bfloat16* __restrict__ O, const unsigned q_len, const unsigned kv_len, const unsigned q_offset,
    const float inv_sqrt_d) {
    ling_prefill::body<false>(Q, K, V, O, nullptr, q_len, kv_len, q_offset, inv_sqrt_d);
}

extern "C" __global__ void __launch_bounds__(256, 1) ling_mla_prefill_rows128_part(
    const __nv_bfloat16* __restrict__ Q, const __nv_bfloat16* __restrict__ K, const __nv_bfloat16* __restrict__ V,
    float* __restrict__ O, float* __restrict__ ml, const unsigned q_len, const unsigned kv_len,
    const unsigned q_offset, const float inv_sqrt_d) {
    ling_prefill::body<true>(Q, K, V, O, ml, q_len, kv_len, q_offset, inv_sqrt_d);
}
