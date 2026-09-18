// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense BF16 dual-GEMV (batch=2) for gfx1151 (Strix Halo).
//
// The batch=2 sibling of dense_gemv_bf16: computes two output rows from ONE
// pass over the BF16 weight matrix, halving weight bandwidth vs two M=1
// launches. Bit-identical to running dense_gemv_bf16 twice — each token's
// accumulator follows the exact same K-iteration order and reduction tree;
// the second token only adds an independent accumulator over the same loop.
//
//   C[t, n] = dot(A[t, :], B[n, :])   for t in {0, 1}
//
//   A: [2, K] BF16 (two activation rows, contiguous)
//   B: [N, K] BF16 (weights, row-major)
//   C: two rows at C + t * out_stride (BF16 elements)
//
// `out_stride` decouples the output row stride from N so callers can write
// straight into per-token strided layouts (e.g. the K=2 verify qkv buffer).
//
// Used by the K=2 MTP verify path (SSM in_proj_qkvz on FP8 checkpoints,
// which dequant GDN in-projections to dense BF16 at load).
//
// Grid: (ceil(N / 8), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 8
#define WARP_SIZE 32
#define VEC_SIZE 8  // BF16 values per vectorized load (uint4 = 16 bytes)

extern "C" __global__ void dense_gemv_bf16_batch2(
    const __nv_bfloat16* __restrict__ A,  // [2, K]
    const __nv_bfloat16* __restrict__ B,  // [N, K]
    __nv_bfloat16* __restrict__ C,        // rows at C + t*out_stride
    unsigned int N,
    unsigned int K,
    unsigned int out_stride                // BF16 elements between output rows
) {
    const unsigned int local_out = threadIdx.x / WARP_SIZE;
    const unsigned int lane = threadIdx.x % WARP_SIZE;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    float acc0 = 0.0f;
    float acc1 = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* A0_vec = (const uint4*)A;
    const uint4* A1_vec = (const uint4*)(A + K);
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += WARP_SIZE) {
        uint4 a0_data = A0_vec[kv];
        uint4 a1_data = A1_vec[kv];
        uint4 b_data = B_vec[kv];

        const unsigned int a0_raw[4] = {a0_data.x, a0_data.y, a0_data.z, a0_data.w};
        const unsigned int a1_raw[4] = {a1_data.x, a1_data.y, a1_data.z, a1_data.w};
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        #pragma unroll
        for (int i = 0; i < 4; i++) {
            __nv_bfloat16 b_lo, b_hi;
            *(unsigned short*)&b_lo = (unsigned short)(b_raw[i] & 0xFFFF);
            *(unsigned short*)&b_hi = (unsigned short)(b_raw[i] >> 16);
            const float bf_lo = __bfloat162float(b_lo);
            const float bf_hi = __bfloat162float(b_hi);

            __nv_bfloat16 a_lo, a_hi;
            *(unsigned short*)&a_lo = (unsigned short)(a0_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a0_raw[i] >> 16);
            acc0 += __bfloat162float(a_lo) * bf_lo;
            acc0 += __bfloat162float(a_hi) * bf_hi;

            *(unsigned short*)&a_lo = (unsigned short)(a1_raw[i] & 0xFFFF);
            *(unsigned short*)&a_hi = (unsigned short)(a1_raw[i] >> 16);
            acc1 += __bfloat162float(a_lo) * bf_lo;
            acc1 += __bfloat162float(a_hi) * bf_hi;
        }
    }

    // Scalar tail for K not divisible by VEC_SIZE (never hits for model dims)
    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        const __nv_bfloat16* B_row = B + (unsigned long long)n * K;
        for (unsigned int k = tail_start + lane; k < K; k += WARP_SIZE) {
            const float bf = __bfloat162float(B_row[k]);
            acc0 += __bfloat162float(A[k]) * bf;
            acc1 += __bfloat162float(A[K + k]) * bf;
        }
    }

    // One warp per output: pure shuffle reduction, no shared memory or barrier.
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        acc0 += __shfl_down_sync(0xFFFFFFFF, acc0, offset);
        acc1 += __shfl_down_sync(0xFFFFFFFF, acc1, offset);
    }

    if (lane == 0) {
        C[n] = __float2bfloat16(acc0);
        C[(unsigned long long)out_stride + n] = __float2bfloat16(acc1);
    }
}
