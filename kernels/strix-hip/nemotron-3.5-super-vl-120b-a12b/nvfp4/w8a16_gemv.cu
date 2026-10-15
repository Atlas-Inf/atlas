// SPDX-License-Identifier: AGPL-3.0-only

// W8A16 GEMV (M=1) — strix-hip n35vl shadow of common/w8a16_gemv.cu.
//
// Same launch contract and signatures; the inner loop is re-tiled so each
// lane stays inside ONE 128-wide FP8 scale block per pass — the block scale
// hoists out of the inner loop (1 load per 2048 k-elements instead of per
// 16), FMA accumulation, 16-byte B/A loads. Shares `w8a16_row_tiled` with
// w8a16_gemv_batch{4,16} so per-row order is identical across the twins.
// NOT bit-identical to the legacy strided kernel (accuracy-gated lever).

#include <cuda_bf16.h>
#include "w8a16_row_tiled.cuh"

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

extern "C" __global__ void w8a16_gemv(
    const __nv_bfloat16* __restrict__ A,            // [1, K]
    const unsigned char* __restrict__ B,             // [N, K] FP8 E4M3
    const float* __restrict__ block_scale,           // [N/BS, K/BS] FP32
    __nv_bfloat16* __restrict__ C,                   // [1, N]
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;  // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;

    __shared__ float s_lut[256];
    __shared__ float smem[N_PER_BLOCK * 2];
    s_lut[threadIdx.x] = E4M3_LUT_T[threadIdx.x];
    __syncthreads();

    if (n >= N) return;

    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;

    float acc = 0.0f;
    w8a16_row_tiled<1>(A, B, block_scale, s_lut, n, n_block, k_blocks,
                       1, K, lane, threads_per_out, &acc);

    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
    }
    if (lane % WARP_SIZE == 0) {
        smem[local_out * 2 + lane / WARP_SIZE] = acc;
    }
    __syncthreads();
    if (lane == 0) {
        C[n] = __float2bfloat16(smem[local_out * 2] + smem[local_out * 2 + 1]);
    }
}
