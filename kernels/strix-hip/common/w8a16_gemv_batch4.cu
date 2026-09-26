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

template <int MAX_M, bool FIXED = false, int VLANES = 1>
__device__ __forceinline__ void w8a16_gemv_batchm_impl(
    const __nv_bfloat16* __restrict__ A,
    const unsigned char* __restrict__ B,
    const float* __restrict__ block_scale,
    __nv_bfloat16* __restrict__ C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    // VLANES=2: 32 physical threads per output; physical lane p owns logical
    // lanes p (acc = old warp 0) and p+32 (acc2 = old warp 1). Both k16 chunks'
    // weight+scale loads issue before any decode/FMA; the block then covers
    // N_PER_BLOCK*VLANES outputs (grid ceil(N/8)). Bit-identical: same per-
    // element expressions, same shfl trees, same final pair-add.
    const unsigned int lanes_per_out = BLOCK_SIZE / (N_PER_BLOCK * VLANES);
    const unsigned int local_out = threadIdx.x / lanes_per_out;
    const unsigned int lane = threadIdx.x % lanes_per_out;
    const unsigned int n = blockIdx.x * (N_PER_BLOCK * VLANES) + local_out;
    const bool valid_n = n < N;
    const unsigned int k16_count = K / 16;
    const unsigned int k16_stride = lanes_per_out * VLANES;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;
    const unsigned int n_block = n / FP8_BLOCK;

    __shared__ float e4m3_lut[256];
    e4m3_lut[threadIdx.x] = atlas_e4m3_to_f32((unsigned char)threadIdx.x);
    __syncthreads();

    float acc[MAX_M];
    #pragma unroll
    for (int row_index = 0; row_index < MAX_M; ++row_index)
        acc[row_index] = 0.0f;

    float acc2[MAX_M];
    #pragma unroll
    for (int row_index = 0; row_index < MAX_M; ++row_index)
        acc2[row_index] = 0.0f;

    if (valid_n) {
        for (unsigned int k16 = lane; k16 < k16_count; k16 += k16_stride) {
            const unsigned int k2 = k16 + lanes_per_out;
            const bool k2v = VLANES == 2 && k2 < k16_count;
            // Both logical lanes' loads issue before either is decoded.
            const unsigned int base_k = k16 * 16;
            const float scale = block_scale[n_block * k_blocks + base_k / FP8_BLOCK];
            const uint4 packed_weights =
                reinterpret_cast<const uint4*>(B + (unsigned long long)n * K)[k16];
            const uint4 packed_weights2 = k2v
                ? reinterpret_cast<const uint4*>(B + (unsigned long long)n * K)[k2]
                : make_uint4(0, 0, 0, 0);
            const float scale2 = k2v
                ? block_scale[n_block * k_blocks + (k2 * 16) / FP8_BLOCK]
                : 0.0f;
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
            const unsigned int weight_words2[4] = {
                packed_weights2.x, packed_weights2.y, packed_weights2.z, packed_weights2.w
            };
            float weights2[16];
            #pragma unroll
            for (int word = 0; word < 4; ++word) {
                const unsigned int packed = weight_words2[word];
                weights2[word * 4] = e4m3_lut[packed & 0xffu] * scale2;
                weights2[word * 4 + 1] = e4m3_lut[(packed >> 8) & 0xffu] * scale2;
                weights2[word * 4 + 2] = e4m3_lut[(packed >> 16) & 0xffu] * scale2;
                weights2[word * 4 + 3] = e4m3_lut[packed >> 24] * scale2;
            }

            #pragma unroll
            for (int row_index = 0; row_index < MAX_M; ++row_index) {
                if (!FIXED && (unsigned int)row_index >= M) continue;
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
            if (VLANES == 2 && k2v) {
                #pragma unroll
                for (int row_index = 0; row_index < MAX_M; ++row_index) {
                    if (!FIXED && (unsigned int)row_index >= M) continue;
                    const __nv_bfloat16* row = A + (unsigned long long)row_index * K;
                    const uint4 lo = reinterpret_cast<const uint4*>(row)[k2 * 2];
                    const uint4 hi = reinterpret_cast<const uint4*>(row)[k2 * 2 + 1];
                    const unsigned int activation_words[8] = {
                        lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w
                    };
                    #pragma unroll
                    for (int pair = 0; pair < 8; ++pair) {
                        const unsigned int packed = activation_words[pair];
                        acc2[row_index] +=
                            atlas_bf16_bits_to_f32(packed & 0xffffu) * weights2[pair * 2]
                            + atlas_bf16_bits_to_f32(packed >> 16) * weights2[pair * 2 + 1];
                    }
                }
            }
        }
    }

    if (VLANES == 2) {
        // 32 physical lanes per output = one warp per output: the two reduces
        // equal the old warp0/warp1 sums; their add is the old partial sum.
        #pragma unroll
        for (int row_index = 0; row_index < MAX_M; ++row_index) {
            if (!FIXED && (unsigned int)row_index >= M) continue;
            float v0 = acc[row_index];
            float v1 = acc2[row_index];
            #pragma unroll
            for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
                v0 += __shfl_down_sync(0xffffffffu, v0, offset);
                v1 += __shfl_down_sync(0xffffffffu, v1, offset);
            }
            if (valid_n && lane == 0) {
                C[(unsigned long long)row_index * N + n] = __float2bfloat16(v0 + v1);
            }
        }
        return;
    }
    __shared__ float partial[MAX_M][N_PER_BLOCK * 2];
    const unsigned int warp_lane = lane % WARP_SIZE;
    const unsigned int warp_in_out = lane / WARP_SIZE;
    #pragma unroll
    for (int row_index = 0; row_index < MAX_M; ++row_index) {
        if (!FIXED && (unsigned int)row_index >= M) continue;
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
            if (FIXED || (unsigned int)row_index < M)
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

extern "C" __global__ void w8a16_gemv_batch8(
    const __nv_bfloat16* A,
    const unsigned char* B,
    const float* block_scale,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<8>(A, B, block_scale, C, M, N, K);
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

// vl2 variants (2 virtual lanes per thread; grid is ceil(N/8), block 256).
// Bit-identical to w8a16_gemv_batch8: same k16 walks and reduce halves,
// only the lane→thread mapping changed.
extern "C" __global__ void w8a16_gemv_batch8_vl2(
    const __nv_bfloat16* A,
    const unsigned char* B,
    const float* block_scale,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<8, true, 2>(A, B, block_scale, C, M, N, K);
}

extern "C" __global__ void w8a16_gemv_batch8_dyn_vl2(
    const __nv_bfloat16* A,
    const unsigned char* B,
    const float* block_scale,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<8, false, 2>(A, B, block_scale, C, M, N, K);
}
