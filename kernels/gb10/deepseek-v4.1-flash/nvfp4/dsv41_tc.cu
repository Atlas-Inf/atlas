// SPDX-License-Identifier: AGPL-3.0-only

// DeepSeek-V4.1-Flash: the projections on tensor cores.
//
// A routed expert's weights are FP4 as the release stores them: W[N, K] as
// E2M1 values, two a byte, low nibble first, plus one E8M0 power-of-two
// scale per 32 consecutive K values (S[N, K/32]). The other projections are
// FP8: E4M3 bytes under the same kind of scale. Every E2M1 or E4M3 value
// times a power of two is exact in BF16, and the activations are FP8-rounded
// BF16, so the BF16 MMA below multiplies the checkpoint's weights unrounded
// and accumulates in FP32, which is the arithmetic of `dsv41_fp4_gemv` (the
// plain form) in another summation order.
//
// The `_f32` entries leave the FP32 sum unrounded. They serve the
// projections whose input is split in four parts (`dsv41_parts.cu`): each
// part's sum is a partial, and the parts are added in a fixed order before
// the one rounding, whichever Spark computed them.
//
//   dsv41_{fp4,fp8}_gemv_tc{8,16,32}[_f32]: C[M, N] = A[M, K] .
//     dequant(W)[N, K]^T for up
//     to 8/16/32 rows. The layout is `mxfp8_gemv_tc`'s (glm-5.3-flash
//     mxfp8_gemv.cu): CTA = 16 weight rows; 8 warps split K in 64-wide
//     chunks; lane (g = lane/4, c = lane%4) loads 16 consecutive K values of
//     rows g and g+8 (8 packed bytes each) and of each activation tile row
//     8t+g, prefetching its next chunk. MMA step j (0..3) feeds k-slots
//     {2c,2c+1} <- values 4j+{0,1} and {2c+8,2c+9} <- 4j+{2,3} of the lane's
//     run in both A and B. A row's output bits do not depend on how many
//     rows share the launch.
//
// A: BF16 [M, K] (row stride K). C rows at C + m*out_stride. K % 32 == 0.
// Grid: (ceil(N/16),1,1) Block: (256,1,1). The body is `dsv41_tc.cuh`.

#include "dsv41_tc.cuh"

#define DSV41_TC(NAME, NT, FP4, F32OUT, CTYPE)                                              \
    extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) NAME(                 \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,           \
        const unsigned char* __restrict__ S, CTYPE* __restrict__ C, unsigned int M,         \
        unsigned int N, unsigned int K, unsigned int out_stride) {                          \
        dsv41_gemv_tc_impl<NT, FP4, F32OUT>(A, W, S, C, M, N, K, out_stride, blockIdx.x);   \
    }

DSV41_TC(dsv41_fp4_gemv_tc8, 1, true, false, __nv_bfloat16)
DSV41_TC(dsv41_fp4_gemv_tc16, 2, true, false, __nv_bfloat16)
DSV41_TC(dsv41_fp4_gemv_tc32, 4, true, false, __nv_bfloat16)
DSV41_TC(dsv41_fp4_gemv_tc8_f32, 1, true, true, float)
DSV41_TC(dsv41_fp4_gemv_tc16_f32, 2, true, true, float)
DSV41_TC(dsv41_fp4_gemv_tc32_f32, 4, true, true, float)
DSV41_TC(dsv41_fp8_gemv_tc8, 1, false, false, __nv_bfloat16)
DSV41_TC(dsv41_fp8_gemv_tc16, 2, false, false, __nv_bfloat16)
DSV41_TC(dsv41_fp8_gemv_tc32, 4, false, false, __nv_bfloat16)
DSV41_TC(dsv41_fp8_gemv_tc8_f32, 1, false, true, float)
DSV41_TC(dsv41_fp8_gemv_tc16_f32, 2, false, true, float)
DSV41_TC(dsv41_fp8_gemv_tc32_f32, 4, false, true, float)
