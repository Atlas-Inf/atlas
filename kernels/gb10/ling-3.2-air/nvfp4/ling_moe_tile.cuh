// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cuda_bf16.h>
#include "../../common/atlas_pdl.cuh"
#include "ling_moe_quant.cuh"

// Ling-3.2-air decode-path routed-expert NVFP4 tiles.
//
// The routed experts of a decode/verify batch are tiny (top-8 of 512, one to
// a few rows per expert), so the tile shape is the GLM verify-decode one the
// Atlas GB10 path proved out (glm_moe_decode_m16.cuh +
// glm_moe_decode_stream.cuh): a 16-row M16 slab (MI slabs per CTA for the
// _m32 twins) over K in 128-element stages, MMAs
// mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64 with
// e2m1 operands and ue4m3 per-16 group scales carried by the instruction,
// one 256-column shared weight tile per stage. Weights are staged with
// streaming loads (`__ldcs`) — the GLM stream-target measurements (7-16%
// faster than cp.async on the M16 decode tiles) — and 16-byte chunks are
// XOR-swizzled exactly as the GLM tiles, so the ldmatrix patterns are
// bank-conflict-free. Two stage buffers keep the whole tile under 42 KB of
// shared memory (conservative; _m32 included) with two CTAs per SM.
//
// Weight layout contract (the GLM native-NVFP4 one, loader-repacked from the
// checkpoint): packed bytes `[K/2, N]` (byte (k/2, n) = the two e2m1 codes of
// k, k+1 at output channel n) and group scales `[K/16, N]` (e4m3 bytes).
// Activation layout: packed `[rows, K/2]`, scales `[rows, K/16]`.
//
// Requires K % 128 == 0 and N % 128 == 0 (gate/up) or N % 256 == 0 (down);
// the host asserts the Ling shapes (K 2560 / 768, N 768 / 2560).

#define LMOE_THREADS 256
#define LMOE_KS 128                 // K per stage
#define LMOE_KP (LMOE_KS / 2)        // packed bytes per stage row (64)
#define LMOE_AP (LMOE_KP + 16)       // activation row pitch (16B aligned)
#define LMOE_NT 256                  // shared weight-tile columns
#define LMOE_NG (LMOE_KS / 16)       // scale groups per stage (8)
#define LMOE_NBUF 2                  // stage buffers

// 16-byte column chunk c: whether it reads the up matrix, and its column.
template <bool GATE_UP>
__device__ __forceinline__ bool lm_chunk_up(unsigned int c) {
    return GATE_UP && c >= LMOE_NT / 32;
}
template <bool GATE_UP>
__device__ __forceinline__ unsigned int lm_chunk_col(unsigned int cta_n, unsigned int c) {
    return cta_n + ((GATE_UP ? c % (LMOE_NT / 32) : c) << 4);
}
// Warp `warp`'s nb-th 16-byte chunk of the tile: gate chunk `warp` + the up
// chunk of the same columns (gate/up), or chunks 2*warp + nb (down).
template <bool GATE_UP>
__device__ __forceinline__ unsigned int lm_warp_chunk(unsigned int warp_id, int nb) {
    return GATE_UP ? nb * (LMOE_NT / 32) + warp_id : warp_id * 2 + nb;
}

__device__ __forceinline__ void lm_ldsm_x4(unsigned int (&d)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}
__device__ __forceinline__ void lm_ldsm_t2(unsigned int (&d)[4], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];"
                 : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]) : "r"(addr));
}
__device__ __forceinline__ void lm_ldsm_t1(unsigned int (&d)[2], unsigned int addr) {
    asm volatile("ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {%0,%1}, [%2];"
                 : "=r"(d[0]), "=r"(d[1]) : "r"(addr));
}
__device__ __forceinline__ unsigned int lm_cvta(const void* p) {
    return (unsigned int)__cvta_generic_to_shared(p);
}

// Warp `warp_id`'s MMAs of one K128 stage over one 16-row slab: the GLM
// pqd_mma_stage fragment and instruction sequence (its operands and scales
// exactly, k64 slices in K order). `sA` is the slab's stage rows
// ([16][LMOE_AP]), `sAs` its group scales ([16][LMOE_NG]).
template <bool GATE_UP>
__device__ __forceinline__ void lm_mma_stage(
    float acc[4][4], const unsigned char* sA, const unsigned char* sAs,
    const unsigned char* sB, const unsigned char* sS,
    unsigned int warp_id, unsigned int lane_id
) {
    const unsigned int kp_l = lane_id, g_l = lane_id & 7;
    const unsigned int sfa_m = (lane_id & 1) * 8 + (lane_id >> 2);
    const unsigned short tid_a = 0, bid_a = 0, bid_b = 0;
    unsigned int sfb[4];
    #pragma unroll
    for (int nb = 0; nb < 2; ++nb) {
        const unsigned int c = lm_warp_chunk<GATE_UP>(warp_id, nb);
        unsigned int d[2];
        lm_ldsm_t1(d, lm_cvta(&sS[g_l * LMOE_NT + ((c ^ g_l) << 4)]));
        sfb[nb * 2] = d[0];
        sfb[nb * 2 + 1] = d[1];
    }
    #pragma unroll
    for (int sl = 0; sl < 2; ++sl) {
        const unsigned int ko = sl * 32;
        const unsigned short tid_b = (unsigned short)sl;
        unsigned int a[4];
        {
            const unsigned int j = lane_id >> 3, r = lane_id & 7;
            lm_ldsm_x4(a, lm_cvta(&sA[(r + (j & 1) * 8) * LMOE_AP + ko + (j >> 1) * 16]));
        }
        const unsigned int sfa = *(const unsigned int*)&sAs[sfa_m * LMOE_NG + sl * 4];
        #pragma unroll
        for (int nb = 0; nb < 2; ++nb) {
            const unsigned int c = lm_warp_chunk<GATE_UP>(warp_id, nb);
            const unsigned int kp = ko + kp_l;
            unsigned int b[4];
            lm_ldsm_t2(b, lm_cvta(&sB[kp * LMOE_NT + ((c ^ (kp & 7)) << 4)]));
            #pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int nt = nb * 2 + h;
                asm volatile(
                    "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X."
                    "m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                    "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
                    "{%14},{%15,%16},{%17},{%18,%19};"
                    : "=f"(acc[nt][0]), "=f"(acc[nt][1]), "=f"(acc[nt][2]), "=f"(acc[nt][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[h]), "r"(b[2 + h]),
                      "f"(acc[nt][0]), "f"(acc[nt][1]), "f"(acc[nt][2]), "f"(acc[nt][3]),
                      "r"(sfa), "h"(bid_a), "h"(tid_a), "r"(sfb[nt]), "h"(bid_b), "h"(tid_b));
            }
        }
    }
}

// Gate/up epilogue: the routed-expert swiglu with the Bailing per-layer limit
// and the NVFP4 re-quantization of the down input. Column layout: this warp
// owns intermediate channels [cta_n + warp*16, +16); v[i] is channel
// cta_n + warp*16 + 8*(i/2) + 2*tid + i%2, rows cta_m + group_id + half*8.
__device__ __forceinline__ void lm_epilogue_gate_up(
    const float acc[4][4], unsigned int warp_id, unsigned int lane_id,
    unsigned int cta_m, unsigned int M_eff, unsigned int cta_n,
    float scale2_gate, float scale2_up, float limit, float a2g,
    unsigned char* __restrict__ out_packed, unsigned char* __restrict__ out_scale,
    unsigned int N
) {
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;
    const unsigned int col = cta_n + warp_id * 16;
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        const bool valid = (unsigned int)(group_id + half * 8) < M_eff;
        const unsigned int row = cta_m + group_id + half * 8;
        float v[4], group_max = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const int j = i >> 1, e = half * 2 + (i & 1);
            const float g = __bfloat162float(__float2bfloat16(acc[j][e] * scale2_gate));
            const float u = __bfloat162float(__float2bfloat16(acc[j + 2][e] * scale2_up));
            const float s = ling_silu_bf16(g, limit);
            v[i] = ling_gate_up_mul(s, u, limit);
            group_max = fmaxf(group_max, fabsf(v[i]));
        }
        group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 1));
        group_max = fmaxf(group_max, __shfl_xor_sync(0xffffffffu, group_max, 2));
        float inv;
        const unsigned char sc = ling_group_scale(group_max, &inv, a2g);
        if (!valid) continue;
        if (tid == 0)
            out_scale[(unsigned long long)row * (N / 16) + col / 16] = sc;
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const unsigned int q0 = ling_e2m1_quantize(v[hh * 2] * inv);
            const unsigned int q1 = ling_e2m1_quantize(v[hh * 2 + 1] * inv);
            out_packed[(unsigned long long)row * (N / 2) + (col + hh * 8) / 2 + tid] =
                (unsigned char)((q1 << 4) | q0);
        }
    }
}

// Down epilogue: BF16 C rows of tile columns cta_n + warp*32 + nt*8 (+2*tid).
__device__ __forceinline__ void lm_epilogue_down(
    const float acc[4][4], unsigned int warp_id, unsigned int lane_id,
    unsigned int cta_m, unsigned int M_eff, unsigned int cta_n,
    float scale2, __nv_bfloat16* __restrict__ C, unsigned int N
) {
    const unsigned int group_id = lane_id >> 2, tid = lane_id & 3;
    const bool r0v = (unsigned int)group_id < M_eff, r1v = (unsigned int)(group_id + 8) < M_eff;
    const unsigned int r0 = cta_m + group_id, r1 = r0 + 8;
    #pragma unroll
    for (int nt = 0; nt < 4; nt++) {
        const unsigned int c0 = cta_n + warp_id * 32 + nt * 8 + tid * 2;
        if (r0v)
            *(__nv_bfloat162*)&C[(unsigned long long)r0 * N + c0] = __floats2bfloat162_rn(
                acc[nt][0] * scale2, acc[nt][1] * scale2);
        if (r1v)
            *(__nv_bfloat162*)&C[(unsigned long long)r1 * N + c0] = __floats2bfloat162_rn(
                acc[nt][2] * scale2, acc[nt][3] * scale2);
    }
}

// The tables this CTA reads: one projection (down) or gate + up.
struct LmGateUp {
    const unsigned long long* __restrict__ up_packed_ptrs;
    const unsigned long long* __restrict__ up_scale_ptrs;
    const float* __restrict__ up_scale2_vals;
    const float* __restrict__ a2_gscale;   // [E]: 1 / input_global_scale(down), or null
    unsigned char* __restrict__ out_packed;
    unsigned char* __restrict__ out_scale;
};

// One work item: expert `expert` (global), row tile `mt` (LMOE_M*MI rows).
// Grid (column tiles, bound): blockIdx.y indexes the worklist (word 0 the
// item count, items from word 1 as (expert, mt) words). Streaming-staged
// K128 pipeline over the whole K.
template <bool GATE_UP, int MI, int LMOE_M = 16>
__device__ __forceinline__ void lm_impl(
    const unsigned char* __restrict__ a_packed,
    const unsigned char* __restrict__ a_scale,
    const unsigned long long* __restrict__ b_packed_ptrs,
    const unsigned long long* __restrict__ b_scale_ptrs,
    const float* __restrict__ scale2_vals,
    const int* __restrict__ expert_offsets,
    const int* __restrict__ sorted_token_ids,
    unsigned int num_experts, unsigned int N, unsigned int K,
    const unsigned int* __restrict__ worklist,
    float swiglu_limit,   // routed per-layer limit, 0 = none (gate/up)
    const LmGateUp& up,
    __nv_bfloat16* __restrict__ C
) {
    atlas_pdl_enter();
    // Shape guards (the host asserts the Ling shapes; these keep a wiring
    // mistake from reading out of bounds instead of failing loudly far away
    // downstream).
    if (K % LMOE_KS != 0u || N == 0u) return;
    if (GATE_UP ? (N % 128u != 0u) : (N % LMOE_NT != 0u)) return;
    if (blockIdx.y >= worklist[0]) return;
    const unsigned int expert = worklist[1 + blockIdx.y * 2];
    const unsigned int mt = worklist[2 + blockIdx.y * 2];
    if (expert >= num_experts) return;
    const unsigned int cta_m = (unsigned int)expert_offsets[expert] + mt * (LMOE_M * MI);
    const int avail = expert_offsets[expert + 1] - (int)cta_m;
    if (avail <= 0) return;
    const unsigned int M_eff =
        (unsigned int)avail < (unsigned int)(LMOE_M * MI) ? (unsigned int)avail : LMOE_M * MI;
    const unsigned char* B_expert = (const unsigned char*)b_packed_ptrs[expert];
    const unsigned char* S_expert = (const unsigned char*)b_scale_ptrs[expert];
    const unsigned char* U_expert = nullptr, *US_expert = nullptr;
    if (GATE_UP) {
        U_expert = (const unsigned char*)up.up_packed_ptrs[expert];
        US_expert = (const unsigned char*)up.up_scale_ptrs[expert];
    }
    if (B_expert == nullptr || (GATE_UP && (U_expert == nullptr || US_expert == nullptr))) return;

    const unsigned int t = threadIdx.x;
    const unsigned int warp_id = t / 32u, lane_id = t % 32u;
    const unsigned int cta_n = blockIdx.x * (GATE_UP ? LMOE_NT / 2 : LMOE_NT);

    __shared__ __align__(16) unsigned char sA[LMOE_NBUF][LMOE_M * MI][LMOE_AP];
    __shared__ __align__(16) unsigned char sAs[LMOE_NBUF][LMOE_M * MI][LMOE_NG];
    __shared__ __align__(16) unsigned char sB[LMOE_NBUF][LMOE_KP][LMOE_NT];
    __shared__ __align__(16) unsigned char sS[LMOE_NBUF][LMOE_NG][LMOE_NT];
    __shared__ int sTok[LMOE_M * MI];

    if (t < LMOE_M * MI) {
        const bool live = t < M_eff;
        sTok[t] = (sorted_token_ids && live) ? sorted_token_ids[cta_m + t]
                                             : (int)(cta_m + t);
    }
    __syncthreads();

    // One K stage: pieces of 16 bytes per thread. Weight and scale pieces
    // stream (`__ldcs`, each expert byte read once from DRAM); the packed
    // activation rows load through L2 (`__ldcg`, gathered by token or by
    // sorted row). Padding rows zero-fill.
    constexpr unsigned int NC = LMOE_NT / 16;
    constexpr unsigned int NB = LMOE_KP * NC;
    constexpr unsigned int NS = LMOE_NG * NC;
    constexpr unsigned int AP = LMOE_KP / 16;
    constexpr unsigned int NA = LMOE_M * MI * AP;
    constexpr int PIECES = (NB + NS + NA + LMOE_THREADS - 1) / LMOE_THREADS > 5
        ? (NB + NS + NA + LMOE_THREADS - 1) / LMOE_THREADS : 5;
    auto load = [&](int buf, unsigned int kb) {
        const uint4 zero = make_uint4(0, 0, 0, 0);
        uint4 v[PIECES];
        uint4* dst[PIECES];
        #pragma unroll
        for (int r = 0; r < PIECES; ++r) {
            const unsigned int p = t + r * LMOE_THREADS;
            const unsigned char* src = nullptr;
            v[r] = zero;
            dst[r] = nullptr;
            if (p < NB + NS + NA) {
                if (p < NB) {
                    const unsigned int kp = p / NC, c = p % NC;
                    dst[r] = (uint4*)&sB[buf][kp][(c ^ (kp & 7)) << 4];
                    src = (lm_chunk_up<GATE_UP>(c) ? U_expert : B_expert)
                        + (unsigned long long)(kb / 2 + kp) * N + lm_chunk_col<GATE_UP>(cta_n, c);
                    v[r] = __ldcs((const uint4*)src);
                } else if (p < NB + NS) {
                    const unsigned int g = (p - NB) / NC, c = (p - NB) % NC;
                    dst[r] = (uint4*)&sS[buf][g][(c ^ g) << 4];
                    src = (lm_chunk_up<GATE_UP>(c) ? US_expert : S_expert)
                        + (unsigned long long)(kb / 16 + g) * N + lm_chunk_col<GATE_UP>(cta_n, c);
                    v[r] = __ldcs((const uint4*)src);
                } else {
                    const unsigned int row = (p - NB - NS) / AP, col = (p - NB - NS) % AP * 16;
                    dst[r] = (uint4*)&sA[buf][row][col];
                    if (row < M_eff)
                        v[r] = __ldcg((const uint4*)&a_packed[
                            (unsigned long long)(unsigned int)sTok[row] * (K / 2) + kb / 2 + col]);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < PIECES; ++r)
            if (dst[r]) *dst[r] = v[r];
        if (t < LMOE_M * MI)
            *(unsigned long long*)&sAs[buf][t][0] = t < M_eff
                ? __ldcg((const unsigned long long*)&a_scale[
                      (unsigned long long)(unsigned int)sTok[t] * (K / 16) + kb / 16])
                : 0ull;
    };

    float acc[MI][4][4];
    #pragma unroll
    for (int mi = 0; mi < MI; mi++)
        #pragma unroll
        for (int nt = 0; nt < 4; nt++)
            acc[mi][nt][0] = acc[mi][nt][1] = acc[mi][nt][2] = acc[mi][nt][3] = 0.0f;

    const unsigned int stages = K / LMOE_KS;
    load(0, 0);
    __syncthreads();
    for (unsigned int st = 0; st < stages; ++st) {
        const int buf = st % LMOE_NBUF;
        if (st + 1 < stages) load((st + 1) % LMOE_NBUF, (st + 1) * LMOE_KS);
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
            lm_mma_stage<GATE_UP>(acc[mi], &sA[buf][mi * LMOE_M][0], &sAs[buf][mi * LMOE_M][0],
                                  &sB[buf][0][0], &sS[buf][0][0], warp_id, lane_id);
        __syncthreads();
    }

    const float s2 = scale2_vals[expert];
    if (GATE_UP) {
        const float s2_up = up.up_scale2_vals[expert];
        const float a2g = up.a2_gscale != nullptr ? up.a2_gscale[expert] : 1.0f;
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
            lm_epilogue_gate_up(acc[mi], warp_id, lane_id, cta_m + mi * LMOE_M,
                                M_eff > (unsigned int)(mi * LMOE_M)
                                    ? M_eff - mi * LMOE_M : 0u,
                                cta_n, s2, s2_up, swiglu_limit, a2g,
                                up.out_packed, up.out_scale, N);
    } else {
        #pragma unroll
        for (int mi = 0; mi < MI; mi++)
            lm_epilogue_down(acc[mi], warp_id, lane_id, cta_m + mi * LMOE_M,
                             M_eff > (unsigned int)(mi * LMOE_M) ? M_eff - mi * LMOE_M : 0u,
                             cta_n, s2, C, N);
    }
}
