// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Dense FP8-weight batched GEMV (M rows) for SM121 (GB10).
//
// The M-row generalisation of dense_gemv_fp8w_batch2: computes M output rows
// from ONE pass over the FP8 weight matrix, so weight bandwidth is paid once
// instead of M times. Per-row scale is applied ONCE after each row's
// accumulator finishes — identical math to running dense_gemv_fp8w M times
// (same K-iteration order, same reduction tree, same scale position).
//
//   C[t, n] = scale[n] * dot(A[t, :], dequant(B[n, :]))   for t in [0, M)
//
//   A: [M, K] BF16 (activation rows, contiguous)
//   B: [N, K] FP8 E4M3 (weights, row-major — ModelOpt per-row-scale layout)
//   row_scale: [N] FP32
//   C: M rows at C + t * out_stride (BF16 elements)
//
// Grid: (ceil(N / 4), 1, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
__device__ __forceinline__ float scl_fp8_bm(unsigned char b) {
    unsigned int s = (b >> 7) & 1u, e = (b >> 3) & 0xFu, m = b & 0x7u; float v;
    if (e == 0u)               v = (float)m * 0.001953125f;
    else if (e == 15u && m == 7u) v = 0.0f;
    else                       v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    return s ? -v : v;
}
#endif

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define VEC_SIZE 16  // FP8 values per uint4 load
#define MAX_M 8      // compile-time cap on batched rows; callers must pass M <= MAX_M

// Accumulate one 4-FP8 chunk against one activation row's 4 BF16 values.
// Four sequential acc += — the same add order as dense_gemv_fp8w's inner
// loop (NOT a tree sum), which is what the per-row bit-identity rests on.
__device__ __forceinline__ void mac4_bm(float& acc, unsigned int a32_lo, unsigned int a32_hi,
                                        float wf0, float wf1, float wf2, float wf3) {
    __nv_bfloat16 a0, a1, a2, a3;
    *(unsigned short*)&a0 = (unsigned short)(a32_lo & 0xFFFF);
    *(unsigned short*)&a1 = (unsigned short)(a32_lo >> 16);
    *(unsigned short*)&a2 = (unsigned short)(a32_hi & 0xFFFF);
    *(unsigned short*)&a3 = (unsigned short)(a32_hi >> 16);
    acc += __bfloat162float(a0) * wf0;
    acc += __bfloat162float(a1) * wf1;
    acc += __bfloat162float(a2) * wf2;
    acc += __bfloat162float(a3) * wf3;
}

extern "C" __global__ void dense_gemv_fp8w_batchm(
    const __nv_bfloat16* __restrict__ A,    // [M, K] BF16
    const unsigned char* __restrict__ B,     // [N, K] FP8 E4M3
    const float* __restrict__ row_scale,     // [N] f32
    __nv_bfloat16* __restrict__ C,           // rows at C + t*out_stride
    unsigned int M,
    unsigned int N,
    unsigned int K,
    unsigned int out_stride                  // BF16 elements between output rows
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK; // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;

    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    if (n >= N) return;

    const float scale = row_scale[n];
    const unsigned int m = (M > MAX_M) ? MAX_M : M;
    float acc[MAX_M];
    #pragma unroll
    for (int t = 0; t < MAX_M; t++) acc[t] = 0.0f;

    const unsigned int K_VEC = K / VEC_SIZE;
    const uint4* B_vec = (const uint4*)(B + (unsigned long long)n * K);

    for (unsigned int kv = lane; kv < K_VEC; kv += threads_per_out) {
        // ONE 16-byte weight load (16 FP8) feeds every row — the point of the
        // batched tier.
        uint4 b_data = B_vec[kv];
        const unsigned int b_raw[4] = {b_data.x, b_data.y, b_data.z, b_data.w};

        // Dequantize the 16 FP8 weights once for all M rows.
        float wf[16];
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const unsigned int w32 = b_raw[i];
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            wf[i * 4 + 0] = scl_fp8_bm((unsigned char)(w32 & 0xFF));
            wf[i * 4 + 1] = scl_fp8_bm((unsigned char)((w32 >> 8) & 0xFF));
            wf[i * 4 + 2] = scl_fp8_bm((unsigned char)((w32 >> 16) & 0xFF));
            wf[i * 4 + 3] = scl_fp8_bm((unsigned char)((w32 >> 24) & 0xFF));
#else
            __nv_fp8_e4m3 a, b, c, d;
            *(unsigned char*)&a = (unsigned char)(w32 & 0xFF);
            *(unsigned char*)&b = (unsigned char)((w32 >> 8) & 0xFF);
            *(unsigned char*)&c = (unsigned char)((w32 >> 16) & 0xFF);
            *(unsigned char*)&d = (unsigned char)((w32 >> 24) & 0xFF);
            wf[i * 4 + 0] = (float)a; wf[i * 4 + 1] = (float)b;
            wf[i * 4 + 2] = (float)c; wf[i * 4 + 3] = (float)d;
#endif
        }

        for (unsigned int t = 0; t < m; t++) {
            const __nv_bfloat16* At = A + (unsigned long long)t * K;
            // 16 BF16 activations per row via 2x uint4 — same positions as
            // dense_gemv_fp8w's per-token read (same accumulation order).
            uint4 a_d0 = ((const uint4*)At)[kv * 2];
            uint4 a_d1 = ((const uint4*)At)[kv * 2 + 1];
            const unsigned int a_raw0[4] = {a_d0.x, a_d0.y, a_d0.z, a_d0.w};
            const unsigned int a_raw1[4] = {a_d1.x, a_d1.y, a_d1.z, a_d1.w};
            float a = acc[t];
            // Same weight order as dense_gemv_fp8w: b_raw[0],b_raw[1] against
            // a_raw0 (weights 0..7), then b_raw[2],b_raw[3] against a_raw1
            // (weights 8..15).
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                mac4_bm(a, a_raw0[i * 2], a_raw0[i * 2 + 1],
                        wf[i * 4 + 0], wf[i * 4 + 1], wf[i * 4 + 2], wf[i * 4 + 3]);
            }
            #pragma unroll
            for (int i = 0; i < 2; i++) {
                mac4_bm(a, a_raw1[i * 2], a_raw1[i * 2 + 1],
                        wf[8 + i * 4 + 0], wf[8 + i * 4 + 1],
                        wf[8 + i * 4 + 2], wf[8 + i * 4 + 3]);
            }
            acc[t] = a;
        }
    }

    // Scalar tail for K not divisible by VEC_SIZE (never hits for model dims).
    {
        const unsigned int tail_start = K_VEC * VEC_SIZE;
        for (unsigned int k = tail_start + lane; k < K; k += threads_per_out) {
            float wv;
#if defined(__SCALE__) || defined(__HIP_PLATFORM_AMD__)
            wv = scl_fp8_bm(B[(unsigned long long)n * K + k]);
#else
            __nv_fp8_e4m3 w8;
            *(unsigned char*)&w8 = B[(unsigned long long)n * K + k];
            wv = (float)w8;
#endif
            for (unsigned int t = 0; t < m; t++) {
                acc[t] += __bfloat162float(A[(unsigned long long)t * K + k]) * wv;
            }
        }
    }

    const unsigned int warp_lane = threadIdx.x % WARP_SIZE;

    for (unsigned int t = 0; t < m; t++) {
        float a = acc[t] * scale;
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
            a += __shfl_down_sync(0xFFFFFFFF, a, offset);
        }
        acc[t] = a;
    }

    // 2 warps per output: cross-warp reduce via shared memory, per row.
    __shared__ float smem[MAX_M][N_PER_BLOCK * 2];

    if (warp_lane == 0) {
        const unsigned int smem_idx = local_out * 2 + (lane / WARP_SIZE);
        for (unsigned int t = 0; t < m; t++) smem[t][smem_idx] = acc[t];
    }
    __syncthreads();

    if (lane == 0) {
        for (unsigned int t = 0; t < m; t++) {
            const float r = smem[t][local_out * 2] + smem[t][local_out * 2 + 1];
            C[(unsigned long long)t * out_stride + n] = __float2bfloat16(r);
        }
    }
}
