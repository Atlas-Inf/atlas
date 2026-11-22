// SPDX-License-Identifier: AGPL-3.0-only
// A3 — tensor-core variant of hy4_batch_linear for fp8-e4m3 weights
// (kind==1, the --attn-proj-dtype=fp8 path). Opt-in via
// ATLAS_HY4_LINEAR_MMA=1; the scalar GEMV stays the default.
//
//   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Numerics are the GEMV's: e4m3 weights decode exactly into bf16 (3-bit
// mantissa fits bf16's 8), activations stay bf16, accumulation is fp32 —
// so parity vs hy4_batch_linear is near-bitwise. No activation
// quantization pass exists; the per-column fp8 weight scale and the
// activation scale are unnecessary because bf16 covers the e4m3 range.
//
// Geometry per CTA: 128 output columns x 64 rows (grid
// [ceil(n/128), ceil(rows/64), heads], block 256). Warp (w&3) -> 16 rows,
// warp (w>>2) -> 64 columns. K sweeps 32-wide stages: B arrives as e4m3
// bytes and is decoded to bf16 fragment registers in-lane; A arrives as
// bf16 via ldmatrix. cp.async double-buffers both.

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define PLM_STAGE 32              // K per pipeline stage (2 k16 mma slices)
#define PLM_AB (PLM_STAGE * 2)    // A row bytes per stage (64)
#define PLM_AP (PLM_AB + 16)      // A smem row pitch (80, 16B-aligned)
#define PLM_BP (PLM_STAGE + 16)   // B smem row pitch, e4m3 bytes (48)

__device__ __forceinline__ void plm_cp16(void* dst, const void* src, bool pred) {
    unsigned int dst_b = __cvta_generic_to_shared(dst);
    unsigned int bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst_b), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void plm_commit() {
    asm volatile("cp.async.commit_group;");
}
template<int N>
__device__ __forceinline__ void plm_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// e4m3 -> bf16 lookup (shared per CTA). bf16 = top 16 bits of the fp32
// decode; e4m3 subnormals are exact, NaN (0x7F) maps to 0.
__device__ __forceinline__ unsigned short plm_lut_e4m3_bf16(unsigned char b) {
    unsigned int s = (b >> 7) & 1, be = (b >> 3) & 0xF, bm = b & 7u;
    float v;
    if (be == 0) v = (float)bm * 0.001953125f;
    else if (be == 15 && bm == 7) v = 0.f;
    else v = __uint_as_float(((be + 120u) << 23) | (bm << 20));
    unsigned int f = __float_as_uint(v);
    return (unsigned short)(((s << 31) | (f & 0x7FFFFFFFu)) >> 16);
}

// One bf16 m16n8k16 MMA: acc += A(16x16) * B(8x16).
__device__ __forceinline__ void plm_mma(
    float* acc, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]));
}

extern "C" __global__ void __launch_bounds__(256) hy4_batch_linear_mma(
    const __nv_bfloat16* x, const unsigned char* weights, const float* scales,
    void* out,
    unsigned n, unsigned k, unsigned rows, unsigned heads, unsigned xh,
    unsigned yh, unsigned xr, unsigned yr, unsigned wr)
{
    const unsigned cta_n = blockIdx.x * 128;
    if (cta_n >= n) return;
    const unsigned head = blockIdx.z;
    if (head >= heads) return;
    const unsigned row0 = blockIdx.y * 64;
    const unsigned m_cnt = min(64u, rows - row0);
    const unsigned t = threadIdx.x;
    const unsigned warp_id = t / 32, lane_id = t % 32;
    const unsigned warp_m = (warp_id & 3) * 16;
    const unsigned warp_n = (warp_id >> 2) * 64;
    const unsigned gid = lane_id >> 2, tid4 = lane_id & 3;

    const unsigned char* wbase = weights + (size_t)head * n * wr;
    const float* hscale = scales + (size_t)head * n;
    const __nv_bfloat16* xh_p = x + (size_t)head * xh;
    __nv_bfloat16* yh_p = (__nv_bfloat16*)out + (size_t)head * yh;

    __shared__ __align__(16) unsigned char sA[2][64][PLM_AP];
    __shared__ __align__(16) unsigned char sW[2][128][PLM_BP];
    __shared__ unsigned short sLut[256];
    for (unsigned i = t; i < 256; i += 256)
        sLut[i] = plm_lut_e4m3_bf16((unsigned char)i);

    // A: 64 rows x 32 bf16 per stage = 4 cp.async lines of 16B per row.
    // W: 128 cols x 32 e4m3 per stage = 2 lines per col (16B each).
    #define PLM_ISSUE(buf_, kb_) do { \
        { \
            const unsigned r_ = t >> 2, q_ = (t & 3) << 4; \
            const bool live_ = r_ < m_cnt; \
            plm_cp16(&sA[(buf_)][r_][q_], \
                xh_p + (size_t)(row0 + r_) * xr + (kb_) + (q_ >> 1), live_); \
        } \
        { \
            _Pragma("unroll") \
            for (int i_ = 0; i_ < 2; ++i_) { \
                const unsigned j_ = t + i_ * 256; \
                const unsigned c_ = j_ & 127u, q_ = (j_ >> 7) << 4; \
                const unsigned gc_ = cta_n + c_; \
                plm_cp16(&sW[(buf_)][c_][q_], \
                    wbase + (size_t)gc_ * wr + (kb_) + q_, gc_ < n); \
            } \
        } \
    } while(0)

    // Compute one stage: 2 k16 slices x 8 n8 tiles per warp.
    // B fragment for m16n8k16 (col-major 8x16): lane holds
    // (col l/4, k (l%4)*2 .. +1) and (col l/4, k+8 .. +9). Decode e4m3 ->
    // bf16 through sLut and pack pairs.
    #define PLM_BFRAG(buf_, nt_, ko_) ({ \
        const unsigned c_ = warp_n + (nt_) * 8 + gid; \
        const unsigned char* p_ = &sW[(buf_)][c_][(ko_)]; \
        unsigned lo_ = p_[tid4 * 2] | ((unsigned)p_[tid4 * 2 + 1] << 8); \
        unsigned hi_ = p_[tid4 * 2 + 8] | ((unsigned)p_[tid4 * 2 + 9] << 8); \
        unsigned b0_ = (unsigned)sLut[lo_ & 255] | ((unsigned)sLut[lo_ >> 8] << 16); \
        unsigned b1_ = (unsigned)sLut[hi_ & 255] | ((unsigned)sLut[hi_ >> 8] << 16); \
        make_uint2(b0_, b1_); \
    })

    #define PLM_COMPUTE(buf_, ACC) do { \
        _Pragma("unroll") \
        for (int sl_ = 0; sl_ < 2; ++sl_) { \
            const unsigned ko_ = sl_ * 16; \
            unsigned a0_, a1_, a2_, a3_; \
            { \
                const unsigned j_ = lane_id >> 3, r_ = lane_id & 7; \
                const unsigned addr_ = __cvta_generic_to_shared( \
                    &sA[(buf_)][warp_m + r_ + (j_ & 1) * 8][(ko_ + (j_ >> 1) * 8) * 2]); \
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];" \
                             : "=r"(a0_), "=r"(a1_), "=r"(a2_), "=r"(a3_) : "r"(addr_)); \
            } \
            _Pragma("unroll") \
            for (int nt_ = 0; nt_ < 8; ++nt_) { \
                const uint2 b_ = PLM_BFRAG(buf_, nt_, ko_); \
                plm_mma(ACC[nt_], a0_, a1_, a2_, a3_, b_.x, b_.y); \
            } \
        } \
    } while(0)

    const unsigned stages = k / PLM_STAGE;
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    plm_wait<0>();
    __syncthreads();
    PLM_ISSUE(0, 0);
    plm_commit();
    int buf = 0;
    for (unsigned st = 0; st < stages; ++st, buf ^= 1) {
        plm_wait<0>();
        __syncthreads();
        if (st + 1 < stages) {
            PLM_ISSUE(buf ^ 1, (st + 1) * PLM_STAGE);
            plm_commit();
        }
        PLM_COMPUTE(buf, acc);
    }
    __syncthreads();

    // Epilogue: acc[0..1] -> row warp_m+gid, acc[2..3] -> row +8; columns
    // cta_n+warp_n+nt*8+tid4*2 (+1). Multiply by the per-column fp8 scale.
    #pragma unroll
    for (int nt = 0; nt < 8; ++nt) {
        const unsigned c0 = cta_n + warp_n + nt * 8 + tid4 * 2;
        const unsigned r0 = row0 + warp_m + gid, r1 = r0 + 8;
        const float s0 = c0 < n ? hscale[c0] : 0.f;
        const float s1 = c0 + 1 < n ? hscale[c0 + 1] : 0.f;
        if (r0 < rows) {
            if (c0 < n) yh_p[(size_t)r0 * yr + c0] = __float2bfloat16_rn(acc[nt][0] * s0);
            if (c0 + 1 < n) yh_p[(size_t)r0 * yr + c0 + 1] = __float2bfloat16_rn(acc[nt][1] * s1);
        }
        if (r1 < rows) {
            if (c0 < n) yh_p[(size_t)r1 * yr + c0] = __float2bfloat16_rn(acc[nt][2] * s0);
            if (c0 + 1 < n) yh_p[(size_t)r1 * yr + c0 + 1] = __float2bfloat16_rn(acc[nt][3] * s1);
        }
    }
    #undef PLM_ISSUE
    #undef PLM_COMPUTE
    #undef PLM_BFRAG
}
