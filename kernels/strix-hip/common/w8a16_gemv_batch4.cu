// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>

#define BLOCK_SIZE 256
#define N_PER_BLOCK 4
#define WARP_SIZE 32
#define FP8_BLOCK 128

__device__ __forceinline__ float atlas_e4m3_to_f32(unsigned char bits) {
    const unsigned int sign = bits >> 7;
    const unsigned int exponent = (bits >> 3) & 0x0fu;
    const unsigned int mantissa = bits & 0x07u;
    float value;
    if (exponent == 0u) {
        value = (float)mantissa * 0.001953125f;
    } else if (exponent == 15u && mantissa == 7u) {
        value = 0.0f;
    } else {
        value = __uint_as_float(((exponent + 120u) << 23) | (mantissa << 20));
    }
    return sign ? -value : value;
}

__device__ __forceinline__ float atlas_bf16_bits_to_f32(unsigned int bits) {
    return __uint_as_float(bits << 16);
}

template <int MAX_M>
__device__ __forceinline__ void w8a16_gemv_batchm_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int threads_per_out = BLOCK_SIZE / N_PER_BLOCK;
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * N_PER_BLOCK + local_out;
    const bool valid_n = n < N;
    const unsigned int k16_count = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;

    __shared__ float e4m3_lut[256];
    e4m3_lut[threadIdx.x] = atlas_e4m3_to_f32((unsigned char)threadIdx.x);
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row_index = 0; row_index < MAX_M; ++row_index)
        acc[row_index] = 0.0f;

    if (valid_n) {
        for (unsigned int k16 = lane; k16 < k16_count; k16 += threads_per_out) {
            const unsigned int base_k = k16 * 16;
            const float scale = block_scale[n_block * k_blocks + base_k / FP8_BLOCK];
            const uint4 packed_weights =
                reinterpret_cast<const uint4*>(B + (unsigned long long)n * K)[k16];
            const unsigned int weight_words[4] = {
                packed_weights.x, packed_weights.y, packed_weights.z, packed_weights.w
            };
            float weights[16];
            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                const unsigned int packed = weight_words[word];
                weights[word * 4] = e4m3_lut[packed & 0xffu] * scale;
                weights[word * 4 + 1] = e4m3_lut[(packed >> 8) & 0xffu] * scale;
                weights[word * 4 + 2] = e4m3_lut[(packed >> 16) & 0xffu] * scale;
                weights[word * 4 + 3] = e4m3_lut[packed >> 24] * scale;
            }

            #pragma unroll
            for (int row_index = 0; row_index < MAX_M; ++row_index) {
                if ((unsigned int)row_index >= M) continue;
                const __nv_bfloat16* row = A + (unsigned long long)row_index * K;
                const uint4 lo = reinterpret_cast<const uint4*>(row)[k16 * 2];
                const uint4 hi = reinterpret_cast<const uint4*>(row)[k16 * 2 + 1];
                const unsigned int activation_words[8] = {
                    lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w
                };
                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const unsigned int packed = activation_words[pair];
                    acc[row_index] +=
                        atlas_bf16_bits_to_f32(packed & 0xffffu) * weights[pair * 2]
                        + atlas_bf16_bits_to_f32(packed >> 16) * weights[pair * 2 + 1];
                }
            }
        }
    }

    __shared__ float partial[MAX_M][N_PER_BLOCK * 2];
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int row_index = 0; row_index < MAX_M; ++row_index) {
        if ((unsigned int)row_index >= M) continue;
        float value = acc[row_index];
        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
            value += __shfl_down_sync(0xffffffffu, value, offset);
        if (warp_lane == 0)
            partial[row_index][local_out * 2 + warp_in_out] = value;
    }
    __syncthreads();

    if (valid_n && lane == 0) {
        #pragma unroll
        for (int row_index = 0; row_index < MAX_M; ++row_index) {
            if ((unsigned int)row_index < M)
                C[(unsigned long long)row_index * N + n] = __float2bfloat16(
                    partial[row_index][local_out * 2]
                    + partial[row_index][local_out * 2 + 1]);
        }
    }
}

extern "C" __global__ void w8a16_gemv_batch4(
    const __nv_bfloat16* A,
    const unsigned char* B,
    const float* block_scale,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<4>(A, B, block_scale, C, M, N, K);
}

extern "C" __global__ void w8a16_gemv_batch16(
    const __nv_bfloat16* A,
    const unsigned char* B,
    const float* block_scale,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<16>(A, B, block_scale, C, M, N, K);
}
