// SPDX-License-Identifier: AGPL-3.0-only
// Tensor-core variant of hy4_batch_linear for bf16 weights (kind==0),
// covering the FFN latent down/up projections and the Engram projections.
// Opt-in via ATLAS_HY4_LINEAR_MMA=1 together with the fp8 path; decode
// (rows < 32) stays on the scalar GEMV. A per-column bf16 bias (Engram)
// is supported through a separate entry point (hy4_batch_linear_bf16_mma_bias)
// so the no-bias kernel contains no bias load at all — ptxas hoists
// conditionally-guarded param loads out of their guard and traps on NULL.
//
//   mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
//
// Same geometry and pipeline as hy4_batch_linear_mma (128 output columns x
// 64 rows per CTA, grid [ceil(n/128), ceil(rows/64), heads], block 256,
// cp.async double-buffered A and W), but W arrives as bf16 so the B
// fragments load directly (no e4m3 LUT decode) and no per-column scale is
// applied. Accumulation is fp32; output is bf16 or fp32 (float_out).

#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>

#define PBM_STAGE 32              // K per pipeline stage (2 k16 mma slices)
#define PBM_AB (PBM_STAGE * 2)    // A row bytes per stage (64)
#define PBM_AP (PBM_AB + 16)      // A smem row pitch (80, 16B-aligned)
#define PBM_BP (PBM_STAGE * 2 + 16)  // W smem row pitch, bf16 bytes (80)

__device__ __forceinline__ void pbm_cp16(void* dst, const void* src, bool pred) {
    unsigned int dst_b = __cvta_generic_to_shared(dst);
    unsigned int bytes = pred ? 16 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :: "r"(dst_b), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void pbm_commit() {
    asm volatile("cp.async.commit_group;");
}
template<int N>
__device__ __forceinline__ void pbm_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}

// One bf16 m16n8k16 MMA: acc += A(16x16) * B(8x16).
__device__ __forceinline__ void pbm_mma(
    float* acc, unsigned int a0, unsigned int a1, unsigned int a2, unsigned int a3,
    unsigned int b0, unsigned int b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%10,%11,%12,%13};"
        :"=f"(acc[0]),"=f"(acc[1]),"=f"(acc[2]),"=f"(acc[3])
        :"r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1),
         "f"(acc[0]),"f"(acc[1]),"f"(acc[2]),"f"(acc[3]));
}

template<bool HAS_BIAS>
__device__ __forceinline__ void hy4_pbm_impl(
    const __nv_bfloat16* x, const __nv_bfloat16* weights, const __nv_bfloat16* bias,
    void* out,
    unsigned n, unsigned k, unsigned rows, unsigned heads, unsigned xh,
    unsigned yh, unsigned xr, unsigned yr, unsigned wr, unsigned float_out)
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

    const __nv_bfloat16* wbase = weights + (size_t)head * n * wr;
    const __nv_bfloat16* xh_p = x + (size_t)head * xh;
    void* yh_p = (char*)out + (size_t)head * yh * (float_out ? 4 : 2);

    __shared__ __align__(16) unsigned char sA[2][64][PBM_AP];
    __shared__ __align__(16) unsigned char sW[2][128][PBM_BP];

    // A: 64 rows x 32 bf16 per stage (4 16B lines per row).
    // W: 128 cols x 32 bf16 per stage = 64B per col (4 16B lines).
    #define PBM_ISSUE(buf_, kb_) do { \
        { \
            const unsigned r_ = t >> 2, q_ = (t & 3) << 4; \
            const bool live_ = r_ < m_cnt; \
            pbm_cp16(&sA[(buf_)][r_][q_], \
                (const char*)(xh_p + (size_t)(row0 + r_) * xr + (kb_) + (q_ >> 1)), live_); \
        } \
        { \
            _Pragma("unroll") \
            for (int i_ = 0; i_ < 2; ++i_) { \
                const unsigned j_ = t + i_ * 256; \
                const unsigned c_ = j_ & 127u, q_ = (j_ >> 7) << 4; \
                const unsigned gc_ = cta_n + c_; \
                pbm_cp16(&sW[(buf_)][c_][q_], \
                    (const char*)(wbase + (size_t)gc_ * wr + (kb_) + (q_ >> 1)), gc_ < n); \
            } \
        } \
    } while(0)

    // B fragment for m16n8k16 from row-major [n][k] bf16: lane holds
    // k pairs (tid4*2, tid4*2+1) and (tid4*2+8, tid4*2+9) of column
    // warp_n+nt*8+gid — two 4B loads of packed bf16 pairs. kb_ is a BYTE
    // offset into the sW row (2 bytes per k element).
    #define PBM_BFRAG(buf_, nt_, kb_) ({ \
        const unsigned c_ = warp_n + (nt_) * 8 + gid; \
        const unsigned char* p_ = &sW[(buf_)][c_][(kb_)]; \
        unsigned b0_ = *(const unsigned*)(p_ + tid4 * 4); \
        unsigned b1_ = *(const unsigned*)(p_ + tid4 * 4 + 16); \
        make_uint2(b0_, b1_); \
    })

    #define PBM_COMPUTE(buf_, ACC) do { \
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
                const uint2 b_ = PBM_BFRAG(buf_, nt_, ko_ * 2); \
                pbm_mma(ACC[nt_], a0_, a1_, a2_, a3_, b_.x, b_.y); \
            } \
        } \
    } while(0)

    const unsigned stages = k / PBM_STAGE;
    float acc[8][4];
    #pragma unroll
    for (int i = 0; i < 8; ++i)
        acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;

    pbm_wait<0>();
    __syncthreads();
    PBM_ISSUE(0, 0);
    pbm_commit();
    int buf = 0;
    for (unsigned st = 0; st < stages; ++st, buf ^= 1) {
        pbm_wait<0>();
        __syncthreads();
        if (st + 1 < stages) {
            PBM_ISSUE(buf ^ 1, (st + 1) * PBM_STAGE);
            pbm_commit();
        }
        PBM_COMPUTE(buf, acc);
    }
    __syncthreads();

    // Epilogue: acc[0..1] -> row warp_m+gid, acc[2..3] -> row +8; columns
    // cta_n+warp_n+nt*8+tid4*2 (+1). bf16 weights need no per-column scale.
    // HAS_BIAS is a compile-time literal: the bias load exists only in the
    // _bias instantiation.
    #define PBM_EPILOGUE(HAS_BIAS) do { \
        _Pragma("unroll") \
        for (int nt = 0; nt < 8; ++nt) { \
            const unsigned c0 = cta_n + warp_n + nt * 8 + tid4 * 2; \
            const unsigned r0 = row0 + warp_m + gid, r1 = r0 + 8; \
            float b0 = 0.f, b1 = 0.f; \
            if (HAS_BIAS) { \
                if (c0 < n) b0 = __bfloat162float(bias[c0]); \
                if (c0 + 1 < n) b1 = __bfloat162float(bias[c0 + 1]); \
            } \
            if (r0 < rows) { \
                if (c0 < n) { \
                    if (float_out) ((float*)yh_p)[(size_t)r0 * yr + c0] = acc[nt][0] + b0; \
                    else ((__nv_bfloat16*)yh_p)[(size_t)r0 * yr + c0] = __float2bfloat16_rn(acc[nt][0] + b0); \
                } \
                if (c0 + 1 < n) { \
                    if (float_out) ((float*)yh_p)[(size_t)r0 * yr + c0 + 1] = acc[nt][1] + b1; \
                    else ((__nv_bfloat16*)yh_p)[(size_t)r0 * yr + c0 + 1] = __float2bfloat16_rn(acc[nt][1] + b1); \
                } \
            } \
            if (r1 < rows) { \
                if (c0 < n) { \
                    if (float_out) ((float*)yh_p)[(size_t)r1 * yr + c0] = acc[nt][2] + b0; \
                    else ((__nv_bfloat16*)yh_p)[(size_t)r1 * yr + c0] = __float2bfloat16_rn(acc[nt][2] + b0); \
                } \
                if (c0 + 1 < n) { \
                    if (float_out) ((float*)yh_p)[(size_t)r1 * yr + c0 + 1] = acc[nt][3] + b1; \
                    else ((__nv_bfloat16*)yh_p)[(size_t)r1 * yr + c0 + 1] = __float2bfloat16_rn(acc[nt][3] + b1); \
                } \
            } \
        } \
    } while (0)
    PBM_EPILOGUE(HAS_BIAS);
    #undef PBM_EPILOGUE
    #undef PBM_ISSUE
    #undef PBM_COMPUTE
    #undef PBM_BFRAG
}

extern "C" __global__ void __launch_bounds__(256) hy4_batch_linear_bf16_mma(
    const __nv_bfloat16* x, const __nv_bfloat16* weights, const __nv_bfloat16* bias,
    void* out,
    unsigned n, unsigned k, unsigned rows, unsigned heads, unsigned xh,
    unsigned yh, unsigned xr, unsigned yr, unsigned wr, unsigned float_out) {
    hy4_pbm_impl<false>(x, weights, bias, out, n, k, rows, heads, xh, yh, xr, yr, wr, float_out);
}
extern "C" __global__ void __launch_bounds__(256) hy4_batch_linear_bf16_mma_bias(
    const __nv_bfloat16* x, const __nv_bfloat16* weights, const __nv_bfloat16* bias,
    void* out,
    unsigned n, unsigned k, unsigned rows, unsigned heads, unsigned xh,
    unsigned yh, unsigned xr, unsigned yr, unsigned wr, unsigned float_out) {
    hy4_pbm_impl<true>(x, weights, bias, out, n, k, rows, heads, xh, yh, xr, yr, wr, float_out);
}
