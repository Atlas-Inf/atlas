// SPDX-License-Identifier: AGPL-3.0-only

// Atlas W8A16 batched GEMV (M<=4) — block-scaled FP8 weight, BF16 activations.
//
//   C[t, n] = sum_k A[t, k] * E4M3_LUT[B[n, k]] * block_scale[n/128, k/128]
//             for t in 0..M  (M <= 4)
//
// The M=4 sibling of `w8a16_gemv` (M=1). At n=4 batched decode the SSM QKVZ /
// out_proj projections share weights across the 4 sequences, so a single pass
// over the FP8 weight matrix serves all M rows: the weight byte is dequantized
// (LUT * block_scale) ONCE and MAC'd into M independent FP32 accumulators. This
// replaces `w8a16_gemm_pipelined` for n<=4, which pads M to a 128-row MMA tile
// (32x compute over-provision, issue/occupancy-bound). One DRAM pass over the
// same weight bytes, no tensor cores, no M padding.
//
// Per-row accumulation order is IDENTICAL to `w8a16_gemv`, so the output is
// bit-identical to running `w8a16_gemv` M times. This claim was false until
// 2026-08-18: the batch body grouped each BF16 pair as `acc += x*w0 + y*w1`,
// while scalar GEMV performs two sequential accumulations. The exact BF16
// microtest now requires zero mismatches. A:[M,K] BF16, B:[N,K] FP8 E4M3,
// block_scale:[N/128,K/128] FP32,
// C:[M,N] BF16. Grid: (ceil(N/4), 1, 1)  Block: (256, 1, 1).

#include <cuda_bf16.h>
#include "w8a16_row_tiled.cuh"

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

// FP8 E4M3 -> f32 lookup table (256 entries). Identical table to w8a16_gemv.cu.
// LUT lives in w8a16_row_tiled.cuh (shared with M=1).


// Templated on the MAX rows (register array bound + unroll). The runtime `M`
// (<= MAX_M) selects how many activation rows are actually computed, so one
// instantiation serves all n in [1, MAX_M]. MAX_M=4 is the optimal common path
// (n<=4); MAX_M=16 covers high-concurrency decode (n=5..16) without falling
// back to the M-padded MMA.
template <int MAX_M>
__device__ __forceinline__ void w8a16_gemv_batchm_impl(
    const __nv_bfloat16* __restrict__ A,    // [M, K] BF16
    const unsigned char* __restrict__ B,     // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,   // [N/128, K/128] FP32
    __nv_bfloat16* __restrict__ C,           // [M, N] BF16
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[256];
    s_lut[threadIdx.x] = E4M3_LUT_T[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int K16 = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;

    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    // Block-aligned tiling — identical per-row order to w8a16_gemv's M=1.
    w8a16_row_tiled<MAX_M>(A, B, block_scale, s_lut, n, n_block, k_blocks,
                           M, K, lane, threads_per_out, acc);

    // Cross-warp reduction (threads_per_out=64 -> 2 warps/output), per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) {
        if ((unsigned int)t >= M) continue;
        float a = acc[t];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        if (lane % WARP_SIZE == 0) smem[t][local_out * 2 + warp_in_out] = a;
    }
    __syncthreads();

    if (lane == 0) {
        #pragma unroll
        for (int t = 0; t < MAX_M; t++) {
            if ((unsigned int)t >= M) continue;
            float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * N + n] = __float2bfloat16(r);
        }
    }
}

// M<=4 (common-path batched decode). Optimal: no wasted unroll past 4 rows.
extern "C" __global__ void w8a16_gemv_batch4(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<4>(A, B, block_scale, C, M, N, K);
}

// M<=16 (high-concurrency decode, n=5..16). Same weight-streaming pass; one
// extra weight read serves up to 16 activation rows instead of the M-padded
// MMA the pipelined kernel would use at these batch sizes.
extern "C" __global__ void w8a16_gemv_batch16(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<16>(A, B, block_scale, C, M, N, K);
}
