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
//   dsv41_{fp4,fp8}_gemv_tc8[_f32]: C[M, N] = A[M, K] . dequant(W)[N, K]^T
//     for up to 8 rows. The layout is `mxfp8_gemv_tc`'s (glm-5.3-flash
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

// An entry a weight format and an output: FP4 or FP8 weights, the sum
// rounded to BF16 or left in FP32.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp4_gemv_tc8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride) {
    atlas_pdl_enter();
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, true, false>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x, K);
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp4_gemv_tc8_f32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride) {
    atlas_pdl_enter();
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, true, true>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x, K);
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_tc8(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / F4_BLOCK, K / F4_BLOCK}, dsv41_touched(N, K),
                          blockIdx.x, dsv41_touchers(gridDim.x));
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, false, false>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x, K);
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_tc8_f32(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / F4_BLOCK, K / F4_BLOCK}, dsv41_touched(N, K),
                          blockIdx.x, dsv41_touchers(gridDim.x));
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, false, true>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x, K);
}

// The same with the input rounded to FP8 on the way in (`F4_QUANT` in
// `dsv41_tc.cuh`): `A` is BF16 as it was computed, and the bytes are those
// of `dsv41_fp8_quant_dequant` and then the entry above.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_tc8_q(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride) {
    atlas_pdl_enter();
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, false, false, F4_QUANT>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x, K);
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_tc8_f32_q(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    float* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride) {
    atlas_pdl_enter();
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, false, true, F4_QUANT>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x, K);
}

// An FP8 projection of an input the ranks gathered, which is in pieces
// (`f4_at` in `dsv41_tc.cuh`): row m of piece p is at A[p * piece_stride +
// m * a_stride]. `_q` rounds the input to FP8 on the way in. C [M, N] BF16.
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_pieces(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    unsigned int a_stride,
    unsigned int piece,
    unsigned long long piece_stride) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / F4_BLOCK, K / F4_BLOCK}, dsv41_touched(N, K),
                          blockIdx.x, dsv41_touchers(gridDim.x));
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, false, false, F4_ROUNDED>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x,
                                                 a_stride, nullptr, nullptr, piece, piece_stride);
}

extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_pieces_q(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride,
    unsigned int a_stride,
    unsigned int piece,
    unsigned long long piece_stride) {
    atlas_pdl_enter();
    __shared__ F4Shared<1> sh;
    dsv41_gemv_tc_impl<1, false, false, F4_QUANT>(sh, A, W, S, C, M, N, K, out_stride, blockIdx.x,
                                               a_stride, nullptr, nullptr, piece, piece_stride);
}

// Two FP8 projections of the same rows in one launch: C0 [M, N0] = A . W0^T
// and C1 [M, N1] = A . W1^T, A [M, K] FP8-rounded BF16, M at most 8. A block
// computes what the same block of `dsv41_fp8_gemv_tc8` computes, the same
// way: the launch touches its weights ahead of its wait, and on weights the
// cache holds staging them is the slower form (38 us a launch against 20).
// Grid (ceil(N0 / 16) + ceil(N1 / 16)).
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
    atlas_pdl_enter_touch(
        blockIdx.x < (N0 + 15u) / 16u ? AtlasTouch{W0, K, K} : AtlasTouch{W1, K, K},
        blockIdx.x < (N0 + 15u) / 16u ? AtlasTouch{S0, K / F4_BLOCK, K / F4_BLOCK}
                                      : AtlasTouch{S1, K / F4_BLOCK, K / F4_BLOCK},
        blockIdx.x < (N0 + 15u) / 16u ? N0 : N1,
        blockIdx.x < (N0 + 15u) / 16u ? blockIdx.x : blockIdx.x - (N0 + 15u) / 16u,
        dsv41_touchers(blockIdx.x < (N0 + 15u) / 16u ? (N0 + 15u) / 16u : (N1 + 15u) / 16u));
    __shared__ F4Shared<1> sh;
    const unsigned int first = (N0 + 15u) / 16u;
    if (blockIdx.x < first) {
        dsv41_gemv_tc_impl<1, false, false>(sh, A, W0, S0, C0, M, N0, K, N0,
                                                blockIdx.x, K);
    } else {
        dsv41_gemv_tc_impl<1, false, false>(sh, A, W1, S1, C1, M, N1, K, N1,
                                                blockIdx.x - first, K);
    }
}

// A block-diagonal FP8 projection in one launch: group y of the grid is
// C[m][y] [N] = W[y] [N, K] . A[m][y] [K] for each of M rows (at most 8),
// with W and S the groups' one after another and A and C a row's groups one
// after another. A is FP8-rounded already (`dsv41_fp8_quant_dequant`): a
// block that rounds its own input does it again for every 16 rows of the
// weights, which at one token costs more than the launch that rounds it
// once.
//
// A block computes what two blocks of `dsv41_fp8_gemv_tc8` compute for its
// group, one after the other, reading its weights as it uses them: they are
// touched ahead of the wait. Half the blocks of a seventh of the shared
// memory that staging takes are few enough for the GPU to hold the grid at
// once, so the projection after this one starts, and touches, beside it.
// Grid (ceil(N / 32), groups).
extern "C" __global__ void __launch_bounds__(F4_WARPS * F4_WARP) dsv41_fp8_gemv_groups(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ W,
    const unsigned char* __restrict__ S,
    __nv_bfloat16* __restrict__ C,
    unsigned int N,
    unsigned int K,
    unsigned int M) {
    atlas_pdl_enter_touch({W, K, K}, {S, K / F4_BLOCK, K / F4_BLOCK},
                          dsv41_touched(gridDim.y * N, K), blockIdx.y * gridDim.x + blockIdx.x,
                          dsv41_touchers(gridDim.x * gridDim.y));
    __shared__ F4Shared<1> sh;
    const unsigned long long y = blockIdx.y;
    for (unsigned int cta = 2u * blockIdx.x; cta < 2u * blockIdx.x + 2u; ++cta) {
        dsv41_gemv_tc_impl<1, false, false>(
            sh, A + y * K, W + y * N * K, S + y * N * (K / F4_BLOCK), C + y * N, M, N, K,
            gridDim.y * N, cta, gridDim.y * K);
        // The first warp adds up the block's sums before the next overwrite them.
        __syncthreads();
    }
}
