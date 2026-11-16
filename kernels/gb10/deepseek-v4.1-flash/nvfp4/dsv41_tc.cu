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
// A: BF16 [M, K] (row stride K). C rows at C + m*out_stride. K % 32 == 0,
// K <= 5120, and N a multiple of 16 or the weights' last rows padded to one.
// Grid: (ceil(N/16),1,1) Block: (256,1,1). The body is `dsv41_tc.cuh`.

#include "dsv41_tc.cuh"

#define DSV41_TC(NAME, NT, FP4, F32OUT, CTYPE)                                              \
    extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) NAME(                 \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,           \
        const unsigned char* __restrict__ S, CTYPE* __restrict__ C, unsigned int M,         \
        unsigned int N, unsigned int K, unsigned int out_stride) {                          \
        __shared__ F4Shared<NT> sh;                                                         \
        dsv41_gemv_tc_impl<NT, FP4, F32OUT>(sh, A, W, S, C, M, N, K, out_stride,            \
                                            blockIdx.x, K);                                 \
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

// The same with the input rounded to FP8 on the way in (`F4_QUANT` in
// `dsv41_tc.cuh`): `A` is BF16 as it was computed, and the bytes are those
// of `dsv41_fp8_quant_dequant` and then the entry above.
#define DSV41_TC_Q(NAME, F32OUT, CTYPE)                                                    \
    extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) NAME(                 \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,           \
        const unsigned char* __restrict__ S, CTYPE* __restrict__ C, unsigned int M,         \
        unsigned int N, unsigned int K, unsigned int out_stride) {                          \
        __shared__ F4Shared<1> sh;                                                          \
        dsv41_gemv_tc_impl<1, false, F32OUT, F4_QUANT>(sh, A, W, S, C, M, N, K, out_stride, \
                                                       blockIdx.x, K);                      \
    }

DSV41_TC_Q(dsv41_fp8_gemv_tc8_q, false, __nv_bfloat16)
DSV41_TC_Q(dsv41_fp8_gemv_tc8_f32_q, true, float)

// An FP8 projection of an input the ranks gathered, which is in pieces
// (`f4_at` in `dsv41_tc.cuh`): row m of piece p is at A[p * piece_stride +
// m * a_stride]. `_q` rounds the input to FP8 on the way in. C [M, N] BF16.
#define DSV41_TC_P(NAME, IN)                                                                \
    extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) NAME(                 \
        const __nv_bfloat16* __restrict__ A, const unsigned char* __restrict__ W,           \
        const unsigned char* __restrict__ S, __nv_bfloat16* __restrict__ C, unsigned int M, \
        unsigned int N, unsigned int K, unsigned int out_stride, unsigned int a_stride,     \
        unsigned int piece, unsigned long long piece_stride) {                              \
        __shared__ F4Shared<1> sh;                                                          \
        dsv41_gemv_tc_impl<1, false, false, IN>(sh, A, W, S, C, M, N, K, out_stride,        \
                                                blockIdx.x, a_stride, nullptr, nullptr,     \
                                                piece, piece_stride);                       \
    }

DSV41_TC_P(dsv41_fp8_gemv_pieces, F4_ROUNDED)
DSV41_TC_P(dsv41_fp8_gemv_pieces_q, F4_QUANT)

// Two FP8 projections of the same rows in one launch: C0 [M, N0] = A . W0^T
// and C1 [M, N1] = A . W1^T, A [M, K] FP8-rounded BF16, M at most 8. A block
// computes what the same block of `dsv41_fp8_gemv_tc8` computes, from staged
// weights: this is for long rows. Grid (ceil(N0 / 16) + ceil(N1 / 16)).
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_two(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W0,
    const unsigned char* __restrict__ S0,
    __nv_bfloat16* __restrict__ C0,
    unsigned int N0,
    const unsigned char* __restrict__ W1,
    const unsigned char* __restrict__ S1,
    __nv_bfloat16* __restrict__ C1,
    unsigned int N1,
    unsigned int K,
    unsigned int M) {
    __shared__ F4Shared<1> sh;
    __shared__ F4Staged staged;
    const unsigned int first = (N0 + 15u) / 16u;
    if (blockIdx.x < first) {
        dsv41_gemv_staged_impl<1, false, false>(sh, staged, A, W0, S0, C0, M, N0, K, N0,
                                                blockIdx.x, K);
    } else {
        dsv41_gemv_staged_impl<1, false, false>(sh, staged, A, W1, S1, C1, M, N1, K, N1,
                                                blockIdx.x - first, K);
    }
}

// `dsv41_fp8_gemv_groups_q` of an input that is FP8-rounded already
// (`dsv41_fp8_quant_dequant`): a block that rounds its own input does it
// again for every 16 rows of the weights, which at one token costs more
// than the launch that rounds it once.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_groups(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int M) {
    __shared__ F4Shared<1> sh;
    __shared__ F4Staged staged;
    const unsigned long long y = blockIdx.y;
    dsv41_gemv_staged_impl<1, false, false>(
        sh, staged, A + y * K, W + y * N * K, S + y * N * (K / F4_BLOCK), C + y * N, M, N, K,
        gridDim.y * N, blockIdx.x, gridDim.y * K);
}

// A block-diagonal FP8 projection in one launch: group y of the grid is
// C[m][y] [N] = W[y] [N, K] . fp8(A[m][y] [K]) for each of M rows (at most
// 8), with W and S the groups' one after another and A and C a row's groups
// one after another. A block computes what the same block of
// `dsv41_fp8_gemv_tc8_q` computes for its group, from staged weights: this
// is for long rows. Grid (ceil(N / 16), groups).
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_groups_q(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int M) {
    __shared__ F4Shared<1> sh;
    __shared__ F4Staged staged;
    const unsigned long long y = blockIdx.y;
    dsv41_gemv_staged_impl<1, false, false, F4_QUANT>(
        sh, staged, A + y * K, W + y * N * K, S + y * N * (K / F4_BLOCK), C + y * N, M, N, K,
        gridDim.y * N, blockIdx.x, gridDim.y * K);
}
