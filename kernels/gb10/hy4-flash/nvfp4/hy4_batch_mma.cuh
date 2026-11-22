// SPDX-License-Identifier: AGPL-3.0-only
// A3 — native-FP4 block-scale MMA variant of hy4_batch_expert.
//
// Replaces the retired bf16 mma.sync path (negative GB10 production result:
// bf16 mma never beat the scalar-FMA proven kernel there). The expert GEMM
// now runs on GB10's native FP4 tensor-core instruction, ported from the
// proven glm-5.3 prequant kernel (moe_w4a4_grouped_gemm_prequant_t_k128):
//
//   mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X
//          .m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3
//
// Geometry per CTA: 128 output columns x 64 rows.
//   grid [ceil(n/128), 1, min(experts, rows*slots)], block 256 (8 warps),
//   warp (w&3) -> 16 rows, warp (w>>2) -> 64 columns; the expert's entry
//   list is processed in 64-entry row blocks (rows <= 256 slots <= 16).
//
// Layout differences vs the glm reference:
//   * Hy4Q4 weights are [n][k/2] row-major (contiguous k per output col),
//     which is ALREADY the fragment order sBt[n][kp] the k128 reference
//     builds with a PRMT transpose — so B loads cp.async straight into the
//     compute bank (same bank geometry/pitch as the reference sBt) with no
//     transpose pass. B traffic is the same strided pattern the proven
//     dequant path already emits.
//   * Activations arrive PRE-QUANTIZED (option (i)): the `x` argument is a
//     packed FP4 buffer (a_rows x k/2 e2m1 nibbles, row-major) followed by
//     its per-16 e4m3 scales (a_rows x k/16). hy4_expert_quantize (same
//     file) builds it per launch: phase 0's a_rows is `rows` (token rows,
//     entry/slots addressing), phase 1's is rows*slots (activated rows,
//     flat entry addressing) — mirroring the proven kernel exactly.
//
// Semantics preserved from hy4_batch_expert:
//   * launch signature unchanged (x reinterpreted as packed FP4);
//   * down==0: dual gate/up outputs with hy4_round, SiLU gate, limit clamp;
//     down==1: plain projection, round only;
//   * EP null-expert guard: null table->w zero-fills every valid entry and
//     valid column of the block;
//   * K must be a multiple of 128 (production 3072/1536 qualify; the
//     launcher keeps the proven kernel otherwise); N columns outside [n)
//     are predicated on both loads and stores.
//
// Numerics: the A operand goes through e2m1/e4m3 quantization, so outputs
// carry the activation-quantization error on top of the proven dequant
// (scripts/dev/mma_parity_test.cu measures it — deliberately NOT bit-exact).

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>

extern "C" __device__ __forceinline__ float hy4_sigmoid_mma(float v) {
    if (v >= 0.f) return 1.f / (1.f + expf(-v));
    float e = expf(v); return e / (1.f + e);
}
extern "C" __device__ __forceinline__ float hy4_round_mma(float x) {
    return (float)__float2bfloat16_rn(x);
}

// K per pipeline stage = two k64 native-MMA slices.
#define PMN_STAGE 128
#define PMN_KP (PMN_STAGE / 2)   // packed bytes per stage row (64)
#define PMN_AP (PMN_KP + 16)     // A row pitch, 16B-aligned (80)
#define PMN_BP (PMN_KP + 16)     // B compute-bank row pitch, 16B-aligned (80)

__device__ __forceinline__ void pmn_cp16(void* dst, const void* src, bool pred) {
    unsigned int dst_b = __cvta_generic_to_shared(dst);
    unsigned int bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst_b), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void pmn_cp8(void* dst, const void* src, bool pred) {
    unsigned int dst_b = __cvta_generic_to_shared(dst);
    unsigned int bytes = pred ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;"
                 :: "r"(dst_b), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void pmn_commit() {
    asm volatile("cp.async.commit_group;");
}
template<int N>
__device__ __forceinline__ void pmn_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// One native-FP4 m16n8k64 block-scale MMA. Scale-vector flag ::4X, both
// scales ue4m3, fp32 accumulator; bid/tid scale flags are zero (no TMA).
__device__ __forceinline__ void pmn_mma(
    float* acc, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1, unsigned int sfa, unsigned int sfb) {
    unsigned short bidA = 0, tidA_ = 0, bidB = 0, tidB_ = 0;
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13},"
        "{%14},{%15,%16},{%17},{%18,%19};"
        :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]),
         "r"(sfa),"h"(bidA),"h"(tidA_),"r"(sfb),"h"(bidB),"h"(tidB_));
}

// ── Activation quantize (BF16 -> packed e2m1 + per-16 e4m3 scales) ────────
// Contract mirrors the Hy4 weight format so dequant(a) = lut[nibble]*fp8(sf):
// per 16-element group, sf = round_e4m3_sat(amax/6) (6.0 is the e2m1
// magnitude ceiling; e4m3 saturates at 448), nibble = nearest e2m1 of v/sf.
// Group order + threshold comparisons only -> byte-stable output.

__device__ __forceinline__ unsigned char pmn_f32_to_e4m3(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1;
    float absv = fabsf(v);
    if (absv >= 448.0f) absv = 448.0f;   // saturate (NaN/Inf included)
    bits = __float_as_uint(absv);
    if (absv == 0.0f) return (unsigned char)(sign << 7);
    int e = (int)((bits >> 23) & 0xFF) - 127;
    unsigned int m = bits & 0x7FFFFF;
    if (e < -9) return (unsigned char)(sign << 7);
    if (e < -6) {                         // e4m3 subnormal
        int man = (int)(absv * 512.0f + 0.5f);
        if (man > 7) man = 7;
        return (unsigned char)((sign << 7) | (unsigned int)man);
    }
    int be = e + 7;
    unsigned int bm;
    if (be > 15) { be = 15; bm = 6; }     // max normal; 0x7F byte is NaN
    else {
        bm = (m + (1u << 19)) >> 20;      // round to nearest
        if (bm > 7) { bm = 0; ++be; if (be > 15) { be = 15; bm = 6; } }
    }
    return (unsigned char)((sign << 7) | ((unsigned int)be << 3) | bm);
}
__device__ __forceinline__ unsigned int pmn_e2m1_idx(float v) {
    float a = fabsf(v);
    unsigned int sign = (v < 0.0f) ? 8u : 0u;
    unsigned int idx;
    if      (a <= 0.25f) idx = 0;
    else if (a <= 0.75f) idx = 1;
    else if (a <= 1.25f) idx = 2;
    else if (a <= 1.75f) idx = 3;
    else if (a <= 2.5f)  idx = 4;
    else if (a <= 3.5f)  idx = 5;
    else if (a <= 5.0f)  idx = 6;
    else                 idx = 7;
    return sign | idx;
}
__device__ __forceinline__ float pmn_e4m3_decode(unsigned char b) {
    unsigned int s = (b >> 7) & 1, be = (b >> 3) & 0xF;
    unsigned int bm = b & 7u;
    float v;
    if (be == 0) v = (float)bm * 0.001953125f;
    else if (be == 15 && bm == 7) v = 0.f;
    else v = __uint_as_float(((be + 120u) << 23) | (bm << 20));
    return s ? -v : v;
}

// Quantize a_rows x K bf16 (row-major) into a single buffer laid out
// [a_rows x K/2 packed | a_rows x K/16 scales]. K % 16 == 0. Fills full
// rows so garbage-free, unconditionally deterministic.
extern "C" __global__ void hy4_expert_quantize(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ a_packed,
    unsigned char* __restrict__ a_scale,
    unsigned a_rows, unsigned k) {
    unsigned row = blockIdx.x;
    if (row >= a_rows) return;
    const __nv_bfloat16* xr = x + (size_t)row * k;
    unsigned char* pr = a_packed + (size_t)row * (k / 2);
    unsigned char* sr = a_scale + (size_t)row * (k / 16);
    for (unsigned g = threadIdx.x; g < k / 16; g += blockDim.x) {
        const __nv_bfloat16* gp = xr + g * 16;
        float amax = 0.f;
        #pragma unroll
        for (unsigned i = 0; i < 16; ++i) {
            float v = fabsf((float)gp[i]);
            if (v > amax) amax = v;
        }
        unsigned char sfb = pmn_f32_to_e4m3(amax * (1.0f / 6.0f));
        sr[g] = sfb;
        float sf = pmn_e4m3_decode(sfb);
        float inv = sf > 0.f ? 1.f / sf : 0.f;
        #pragma unroll
        for (unsigned i = 0; i < 16; i += 2) {
            unsigned n0 = pmn_e2m1_idx((float)gp[i] * inv);
            unsigned n1 = pmn_e2m1_idx((float)gp[i + 1] * inv);
            pr[g * 8 + i / 2] = (unsigned char)((n1 << 4) | (n0 & 0xF));
        }
    }
}

// ── The expert GEMM ──────────────────────────────────────────────────────
extern "C" __global__ void __launch_bounds__(256) hy4_batch_expert_mma(
    const unsigned char* x,               // packed FP4 activations + scales
    const Hy4Q4* table, const unsigned* groups, __nv_bfloat16* out,
    unsigned n, unsigned k, unsigned rows, unsigned slots, unsigned group,
    unsigned down, float limit, unsigned experts)
{
    const unsigned* active = groups + (size_t)experts * (rows + 1);
    if (blockIdx.z >= active[0]) return;
    unsigned e = active[1 + blockIdx.z];
    const unsigned cta_n = blockIdx.x * 128;
    if (cta_n >= n) return;
    const unsigned t = threadIdx.x;
    const unsigned warp_id = t / 32, lane_id = t % 32;
    const unsigned warp_m = (warp_id & 3) * 16;
    const unsigned warp_n = (warp_id >> 2) * 64;
    const unsigned gid = lane_id >> 2, tid4 = lane_id & 3;

    // Row space of the phase's activation buffer (mirrors proven indexing):
    // gate/up quantized `rows` token rows (entry/slots addressing), down
    // quantized rows*slots activated rows (flat entry addressing).
    const unsigned a_rows = down ? rows * slots : rows;
    const unsigned char* a_packed = x;
    const unsigned char* a_scale = x + (size_t)a_rows * (k / 2);

    const unsigned* list = groups + (size_t)e * (rows + 1);
    const unsigned count = list[0];

    const Hy4Q4 gq = table[(size_t)(e + group) * 3 + (down ? 2u : 0u)];
    const Hy4Q4 uq = table[(size_t)(e + group) * 3 + 1];

    // EP null-expert guard (remote expert): zero-fill every valid entry and
    // column this block covers — the proven path's null-descriptor output.
    if (!gq.w || (!down && !uq.w)) {
        for (unsigned i = t; i < count * 128; i += 256) {
            unsigned r = i >> 7, c = cta_n + (i & 127u);
            if (r < count && c < n)
                out[(size_t)list[1 + r] * n + c] = __float2bfloat16_rn(0.f);
        }
        return;
    }

    const float w_scale = gq.scale;
    const float u_scale = uq.scale;

    // A: double-buffered packed rows + scales.
    __shared__ __align__(16) unsigned char sA[2][64][PMN_AP];
    __shared__ __align__(16) unsigned char sAs[2][64][PMN_STAGE / 16];
    // B: double-buffered compute banks in fragment order (Hy4's [n][k/2]
    // weight layout IS fragment order; cp.async lands it directly).
    __shared__ __align__(16) unsigned char sB[2][128][PMN_BP];
    __shared__ __align__(16) unsigned char sSf[2][128][PMN_STAGE / 16];
    __shared__ int sTok[64];

    // sTok holds the flat entry; phase 0 quantized-buffer rows are
    // entry/slots, phase 1 rows are the entry itself (proven addressing).
    #define PMN_AROW(raw_) (down ? (raw_) : ((raw_) / slots))
    // Issue one full stage (A row-block + B 128-col tile) into bank `buf`.
    #define PMN_ISSUE(buf_, kb_, bw_, bs_) do { \
        { \
            const unsigned row_ = t >> 2, col_ = (t & 3) << 4; \
            const bool live_ = row_ < m_cnt; \
            pmn_cp16(&sA[(buf_)][row_][col_], \
                a_packed + (size_t)PMN_AROW((unsigned)sTok[row_]) * (k / 2) \
                    + (kb_) / 2 + col_, live_); \
            if (t < 64) { \
                const bool live2_ = t < m_cnt; \
                pmn_cp8(&sAs[(buf_)][t][0], \
                    a_scale + (size_t)PMN_AROW((unsigned)sTok[t]) * (k / 16) \
                        + (kb_) / 16, live2_); \
            } \
        } \
        { \
            _Pragma("unroll") \
            for (int r_ = 0; r_ < 2; ++r_) { \
                const unsigned j_ = t + r_ * 256; \
                const unsigned c_ = j_ & 127u, q_ = (j_ >> 7) << 4; \
                const unsigned gc_ = cta_n + c_; \
                pmn_cp16(&sB[(buf_)][c_][q_], \
                    (bw_) + (size_t)gc_ * (k / 2) + (kb_) / 2 + q_, gc_ < n); \
            } \
            if (t < 128) { \
                const unsigned gc2_ = cta_n + t; \
                pmn_cp8(&sSf[(buf_)][t][0], \
                    (bs_) + (size_t)gc2_ * (k / 16) + (kb_) / 16, gc2_ < n); \
            } \
        } \
    } while(0)
    // Sweep the two k64 slices of the current stage bank (8 n-tiles x 2 h).
    #define PMN_COMPUTE(buf_, ACC) do { \
        _Pragma("unroll") \
        for (int sl_ = 0; sl_ < 2; ++sl_) { \
            const unsigned ko_ = sl_ * 32; \
            unsigned a0_, a1_, a2_, a3_; \
            { \
                const unsigned j_ = lane_id >> 3, r_ = lane_id & 7; \
                const unsigned addr_ = __cvta_generic_to_shared( \
                    &sA[(buf_)][warp_m + r_ + (j_ & 1) * 8][ko_ + (j_ >> 1) * 16]); \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" \
                             : "=r"(a0_), "=r"(a1_), "=r"(a2_), "=r"(a3_) : "r"(addr_)); \
            } \
            const unsigned sfa_m_ = (lane_id & 1) * 8 + (lane_id >> 2); \
            const unsigned sfa_ = \
                *(const unsigned int*)&sAs[(buf_)][warp_m + sfa_m_][sl_ * 4]; \
            _Pragma("unroll") \
            for (int np_ = 0; np_ < 4; ++np_) { \
                unsigned b_[4]; \
                { \
                    const unsigned j_ = lane_id >> 3, r_ = lane_id & 7; \
                    const unsigned addr_ = __cvta_generic_to_shared( \
                        &sB[(buf_)][warp_n + np_ * 16 + (j_ >> 1) * 8 + r_][ko_ + (j_ & 1) * 16]); \
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" \
                                 : "=r"(b_[0]), "=r"(b_[1]), "=r"(b_[2]), "=r"(b_[3]) : "r"(addr_)); \
                } \
                _Pragma("unroll") \
                for (int h_ = 0; h_ < 2; ++h_) { \
                    const int nt_ = np_ * 2 + h_; \
                    const unsigned sfb_ = \
                        *(const unsigned int*)&sSf[(buf_)][warp_n + nt_ * 8 + (lane_id >> 2)][sl_ * 4]; \
                    pmn_mma(ACC[nt_], a0_, a1_, a2_, a3_, b_[h_ * 2], b_[h_ * 2 + 1], sfa_, sfb_); \
                } \
            } \
        } \
    } while(0)

    const unsigned stages = k / PMN_STAGE;
    for (unsigned base = 0; base < count; base += 64) {
        const unsigned m_cnt = min(64u, count - base);
        for (unsigned i = t; i < 64; ++i)
            sTok[i] = i < m_cnt ? (int)list[1 + base + i] : 0;
        __syncthreads();

        float accg[8][4], accu[8][4];
        #pragma unroll
        for (int i = 0; i < 8; ++i)
            accg[i][0]=accg[i][1]=accg[i][2]=accg[i][3]=
            accu[i][0]=accu[i][1]=accu[i][2]=accu[i][3]=0.f;

        // Two sequential reference-shaped K128 sweeps per row block: matrix 0
        // (gate, or down), then matrix 1 (up). The reference's proven
        // pipeline shape is kept verbatim per sweep (one outstanding cp.async
        // group, issue->compute overlap). Only the B-side transpose stage is
        // gone (Hy4 layout is already fragment order).
        #define PMN_SWEEP(ACC, bw_, bs_) do { \
            PMN_ISSUE(0, 0, bw_, bs_); \
            pmn_commit(); \
            int buf_ = 0; \
            for (unsigned st_ = 0; st_ < stages; ++st_, buf_ ^= 1) { \
                pmn_wait<0>(); \
                __syncthreads(); \
                if (st_ + 1 < stages) { \
                    PMN_ISSUE(buf_ ^ 1, (st_ + 1) * PMN_STAGE, bw_, bs_); \
                    pmn_commit(); \
                } \
                PMN_COMPUTE(buf_, ACC); \
            } \
            __syncthreads(); \
        } while(0)

        PMN_SWEEP(accg, gq.w, gq.s);
        if (!down) PMN_SWEEP(accu, uq.w, uq.s);
        #undef PMN_SWEEP

        // Epilogue (identical store/round/clamp semantics to the proven
        // kernel): acc[0..1] -> local row warp_m+gid, acc[2..3] -> row
        // warp_m+gid+8, columns cta_n+warp_n+nt*8+tid4*2 (+1). Padded rows
        // (>= m_cnt) and out-of-range columns are skipped.
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            const unsigned c0 = cta_n + warp_n + nt * 8 + tid4 * 2, c1 = c0 + 1;
            const unsigned r0 = warp_m + gid, r1 = r0 + 8;
            if (r0 < m_cnt) {
                const unsigned or0 = (unsigned)sTok[r0];
                if (c0 < n) {
                    float v = accg[nt][0] * w_scale;
                    if (!down) {
                        float u = accu[nt][0] * u_scale;
                        v = hy4_round_mma(v); u = hy4_round_mma(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid_mma(v) * u;
                    } else v = hy4_round_mma(v);
                    out[(size_t)or0 * n + c0] = __float2bfloat16_rn(v);
                }
                if (c1 < n) {
                    float v = accg[nt][1] * w_scale;
                    if (!down) {
                        float u = accu[nt][1] * u_scale;
                        v = hy4_round_mma(v); u = hy4_round_mma(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid_mma(v) * u;
                    } else v = hy4_round_mma(v);
                    out[(size_t)or0 * n + c1] = __float2bfloat16_rn(v);
                }
            }
            if (r1 < m_cnt) {
                const unsigned or1 = (unsigned)sTok[r1];
                if (c0 < n) {
                    float v = accg[nt][2] * w_scale;
                    if (!down) {
                        float u = accu[nt][2] * u_scale;
                        v = hy4_round_mma(v); u = hy4_round_mma(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid_mma(v) * u;
                    } else v = hy4_round_mma(v);
                    out[(size_t)or1 * n + c0] = __float2bfloat16_rn(v);
                }
                if (c1 < n) {
                    float v = accg[nt][3] * w_scale;
                    if (!down) {
                        float u = accu[nt][3] * u_scale;
                        v = hy4_round_mma(v); u = hy4_round_mma(u);
                        if (limit > 0.f) { v = fminf(v, limit); u = fminf(limit, fmaxf(-limit, u)); }
                        v = v * hy4_sigmoid_mma(v) * u;
                    } else v = hy4_round_mma(v);
                    out[(size_t)or1 * n + c1] = __float2bfloat16_rn(v);
                }
            }
        }
        __syncthreads();   // sTok/smem reuse across row blocks
    }
    #undef PMN_AROW
    #undef PMN_ISSUE
    #undef PMN_COMPUTE
}
