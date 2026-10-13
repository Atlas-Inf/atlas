// SPDX-License-Identifier: AGPL-3.0-only

// MXFP8 weights for small-row projections (speculative drafter blocks).
//
// Format: W[N, K] as E4M3 bytes (row-major) plus one E8M0 power-of-two scale
// per 32 consecutive K values (S[N, K/32]), the OCP MX layout. E4M3 values and
// power-of-two scales are both exactly representable in BF16, so the tensor-
// core GEMV below dequantizes into BF16 MMA operands without rounding.
//
//   mxfp8_quantize_bf16: one thread per 32-value block; the scale is the
//     smallest power of two with amax / scale <= 448, values round-to-nearest
//     with saturation.
//   mxfp8_gemv_tc{8,16,32}: C[M, N] = A[M, K] · dequant(W)[N, K]^T for up to
//     8/16/32 rows. CTA = 16 weight rows; 8 warps split K in 64-wide chunks;
//     lane (g = lane/4, c = lane%4) loads 16 consecutive K values of rows g
//     and g+8 and of each activation tile row 8t+g, prefetching its next
//     chunk. MMA step j (0..3) feeds k-slots {2c,2c+1} <- values 4j+{0,1} and
//     {2c+8,2c+9} <- 4j+{2,3} of the lane's run in both A and B.
//
//   mxfp8_gemv_tc{8,16}_grouped: the same per head h = blockIdx.z for
//     per-head weights (GLM absorbed W_uk / W_uv): A[:, h*K..] (row stride
//     lda) times head h of W [G, N, K] / S [G, N, K/32] into C[:, h*N..].
//     No 32-row tier: at K = 256 half its K-split warps idle and it lost to
//     BF16 cuBLAS (scripts/dev/mxfp8_grouped_bench.cu).
//
// A: BF16 [M, K] (row stride K). C rows at C + m*out_stride. K % 32 == 0.
// Grid: (ceil(N/16),1,G) Block: (256,1,1); G = 1 for the plain tiers.

#include "atlas_pdl_touch.cuh"
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define MX_WARPS 8
#define MX_WARP 32
#define MX_BLOCK 32

extern "C" __global__ void mxfp8_quantize_bf16(
    const __nv_bfloat16* __restrict__ W,
    unsigned char* __restrict__ Q,
    unsigned char* __restrict__ S,
    unsigned long long blocks
) {
    const unsigned long long b = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const __nv_bfloat16* w = W + b * MX_BLOCK;
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < MX_BLOCK; i++) amax = fmaxf(amax, fabsf(__bfloat162float(w[i])));
    // Smallest e with amax * 2^-e <= 448 (E4M3 max), clamped to E8M0's range.
    int e = -127;
    if (amax > 0.0f) {
        e = (int)ceilf(log2f(amax / 448.0f));
        e = max(-127, min(127, e));
        if (ldexpf(amax, -e) > 448.0f && e < 127) e++;
    }
    S[b] = (unsigned char)(e + 127);
    #pragma unroll
    for (int i = 0; i < MX_BLOCK; i++) {
        const __nv_fp8_e4m3 q(ldexpf(__bfloat162float(w[i]), -e));
        Q[b * MX_BLOCK + i] = *reinterpret_cast<const unsigned char*>(&q);
    }
}

template <int NT>
struct MxChunk {
    uint4 w0, w1;          // 16 E4M3 values of rows g and g+8
    uint4 x[NT][2];        // 16 BF16 activations of row 8t+g per tile t
    unsigned int s0, s1;   // E8M0 block scales of rows g, g+8
};

template <int NT>
__device__ __forceinline__ void mx_load(
    MxChunk<NT>& t, const unsigned char* w0p, const unsigned char* w1p,
    const unsigned char* s0p, const unsigned char* s1p, const __nv_bfloat16* A,
    unsigned int g, unsigned int M, unsigned int K, unsigned int lda, unsigned int kb,
    bool v0, bool v1
) {
    const uint4 z = make_uint4(0u, 0u, 0u, 0u);
    t.w0 = z; t.w1 = z; t.s0 = 0u; t.s1 = 0u;
    #pragma unroll
    for (int i = 0; i < NT; i++) { t.x[i][0] = z; t.x[i][1] = z; }
    if (kb >= K) return;
    if (v0) { t.w0 = *(const uint4*)(w0p + kb); t.s0 = s0p[kb / MX_BLOCK]; }
    if (v1) { t.w1 = *(const uint4*)(w1p + kb); t.s1 = s1p[kb / MX_BLOCK]; }
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        const unsigned int row = 8u * i + g;
        if (row < M) {
            const uint4* xp = (const uint4*)(A + (unsigned long long)row * lda + kb);
            t.x[i][0] = xp[0];
            t.x[i][1] = xp[1];
        }
    }
}

// BF16 2^(s-127) (E8M0 scale; 0 decodes to 0, as does the NaN code 255).
__device__ __forceinline__ __nv_bfloat162 mx_scale(unsigned int s) {
    const unsigned short bits = (s == 0u || s == 255u) ? 0 : (unsigned short)(s << 7);
    const __nv_bfloat16 v = __ushort_as_bfloat16(bits);
    return __halves2bfloat162(v, v);
}

template <int NT>
__device__ __forceinline__ void mxfp8_gemv_tc_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int lda,
    unsigned int out_stride
) {
    // E4M3 byte -> BF16 bits (exact).
    __shared__ unsigned short s_e4m3[256];
    __shared__ float s_red[MX_WARPS][MX_WARP][NT * 4];
    for (unsigned int i = threadIdx.x; i < 256; i += blockDim.x) {
        __nv_fp8_e4m3 f;
        *reinterpret_cast<unsigned char*>(&f) = (unsigned char)i;
        s_e4m3[i] = __bfloat16_as_ushort(__float2bfloat16((float)f));
    }
    __syncthreads();

    const unsigned int warp = threadIdx.x / MX_WARP;
    const unsigned int lane = threadIdx.x % MX_WARP;
    const unsigned int g = lane >> 2, c = lane & 3u;
    const unsigned int r0 = blockIdx.x * 16u + g, r1 = r0 + 8u;
    const bool v0 = r0 < N, v1 = r1 < N;
    const unsigned long long blocks = K / MX_BLOCK;
    const unsigned char* w0p = W + (unsigned long long)(v0 ? r0 : 0u) * K;
    const unsigned char* w1p = W + (unsigned long long)(v1 ? r1 : 0u) * K;
    const unsigned char* s0p = S + (unsigned long long)(v0 ? r0 : 0u) * blocks;
    const unsigned char* s1p = S + (unsigned long long)(v1 ? r1 : 0u) * blocks;

    float acc[NT][4];
    #pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
    const unsigned int chunks = (K + 63u) / 64u;
    MxChunk<NT> cur, nxt;
    unsigned int ch = warp;
    if (ch < chunks)
        mx_load<NT>(cur, w0p, w1p, s0p, s1p, A, g, M, K, lda, ch * 64u + 16u * c, v0, v1);
    for (; ch < chunks; ch += MX_WARPS) {
        const unsigned int next = ch + MX_WARPS;
        if (next < chunks)
            mx_load<NT>(nxt, w0p, w1p, s0p, s1p, A, g, M, K, lda, next * 64u + 16u * c, v0, v1);
        const __nv_bfloat162 sc0 = mx_scale(cur.s0), sc1 = mx_scale(cur.s1);
        const unsigned int wb0[4] = {cur.w0.x, cur.w0.y, cur.w0.z, cur.w0.w};
        const unsigned int wb1[4] = {cur.w1.x, cur.w1.y, cur.w1.z, cur.w1.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            // Values 4j..4j+3 of each row's 16-value run = bytes 4j..4j+3 = word j.
            const unsigned int x0 = wb0[j], x1 = wb1[j];
            unsigned int a[4];
            __nv_bfloat162 v;
            v = __hmul2(__halves2bfloat162(__ushort_as_bfloat16(s_e4m3[x0 & 0xFFu]),
                                           __ushort_as_bfloat16(s_e4m3[(x0 >> 8) & 0xFFu])), sc0);
            a[0] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(__halves2bfloat162(__ushort_as_bfloat16(s_e4m3[x1 & 0xFFu]),
                                           __ushort_as_bfloat16(s_e4m3[(x1 >> 8) & 0xFFu])), sc1);
            a[1] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(__halves2bfloat162(__ushort_as_bfloat16(s_e4m3[(x0 >> 16) & 0xFFu]),
                                           __ushort_as_bfloat16(s_e4m3[x0 >> 24])), sc0);
            a[2] = *reinterpret_cast<unsigned int*>(&v);
            v = __hmul2(__halves2bfloat162(__ushort_as_bfloat16(s_e4m3[(x1 >> 16) & 0xFFu]),
                                           __ushort_as_bfloat16(s_e4m3[x1 >> 24])), sc1);
            a[3] = *reinterpret_cast<unsigned int*>(&v);
            #pragma unroll
            for (int i = 0; i < NT; i++) {
                const uint4 lo = cur.x[i][0], hi = cur.x[i][1];
                const unsigned int xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                    : "+f"(acc[i][0]), "+f"(acc[i][1]), "+f"(acc[i][2]), "+f"(acc[i][3])
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
                      "r"(xw[2 * j]), "r"(xw[2 * j + 1]));
            }
        }
        cur = nxt;
    }

    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) s_red[warp][lane][i * 4 + q] = acc[i][q];
    }
    __syncthreads();
    if (warp != 0) return;
    #pragma unroll
    for (int i = 0; i < NT; i++) {
        #pragma unroll
        for (int q = 0; q < 4; q++) {
            float v = 0.0f;
            #pragma unroll
            for (int w = 0; w < MX_WARPS; w++) v += s_red[w][lane][i * 4 + q];
            const unsigned int n = (q < 2) ? r0 : r1;
            const unsigned int m = 8u * i + 2u * c + (q & 1u);
            if (n < N && m < M) C[(unsigned long long)m * out_stride + n] = __float2bfloat16(v);
        }
    }
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    atlas_pdl_enter();
    mxfp8_gemv_tc_impl<1>(A, W, S, C, M, N, K, K, out_stride);
}

// `mxfp8_gemv_tc8` whose first `touch_ctas` CTAs pull the first `touch_rows`
// weight rows (values and scales) into L2 while the kernel waits on its PDL
// predecessor (atlas_pdl_touch.cuh). Same body: bit-identical.
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / MX_BLOCK, K / MX_BLOCK}, touch_rows, blockIdx.x,
                          touch_ctas);
    mxfp8_gemv_tc_impl<1>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_impl<2>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride) {
    mxfp8_gemv_tc_impl<4>(A, W, S, C, M, N, K, K, out_stride);
}

// Touch twins of the 16/32-row tiers (see `mxfp8_gemv_tc8_touch`).
extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / MX_BLOCK, K / MX_BLOCK}, touch_rows, blockIdx.x,
                          touch_ctas);
    mxfp8_gemv_tc_impl<2>(A, W, S, C, M, N, K, K, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc32_touch(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int out_stride,
    unsigned int touch_rows, unsigned int touch_ctas) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / MX_BLOCK, K / MX_BLOCK}, touch_rows, blockIdx.x,
                          touch_ctas);
    mxfp8_gemv_tc_impl<4>(A, W, S, C, M, N, K, K, out_stride);
}

template <int NT>
__device__ __forceinline__ void mxfp8_gemv_tc_grouped(
    const __nv_bfloat16* A, const unsigned char* W, const unsigned char* S, __nv_bfloat16* C,
    unsigned int M, unsigned int N, unsigned int K, unsigned int lda, unsigned int out_stride) {
    atlas_pdl_enter();
    const unsigned long long h = blockIdx.z, nk = (unsigned long long)N * K;
    mxfp8_gemv_tc_impl<NT>(A + h * K, W + h * nk, S + h * (nk / MX_BLOCK), C + h * N,
                           M, N, K, lda, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc8_grouped(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C, unsigned int M,
    unsigned int N, unsigned int K, unsigned int lda, unsigned int out_stride) {
    mxfp8_gemv_tc_grouped<1>(A, W, S, C, M, N, K, lda, out_stride);
}

extern "C" __global__ void __launch_bounds__(MX_WARPS * MX_WARP) mxfp8_gemv_tc16_grouped(
    const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C, unsigned int M,
    unsigned int N, unsigned int K, unsigned int lda, unsigned int out_stride) {
    mxfp8_gemv_tc_grouped<2>(A, W, S, C, M, N, K, lda, out_stride);
}
