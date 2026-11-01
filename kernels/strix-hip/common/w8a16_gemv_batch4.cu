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

template <int MAX_M, int VLANES = 1>
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
            if (VLANES == 2 && k2v) {
                #pragma unroll
                for (int row_index = 0; row_index < MAX_M; ++row_index) {
                    if ((unsigned int)row_index >= M) continue;
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
            if ((unsigned int)row_index >= M) continue;
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

// vl2 variant (2 virtual lanes per thread; grid is ceil(N/8), block 256).
// Bit-identical to w8a16_gemv_batch8: same k16 walks and reduce halves, only
// the lane→thread mapping changed. dyn only — the guarded twin beat the fixed
// one at m==8 (475 vs 551 us/call; fixed's 98-133 VGPR triple vs 47-75).

extern "C" __global__ void w8a16_gemv_batch8_dyn_vl2(
    const __nv_bfloat16* A,
    const unsigned char* B,
    const float* block_scale,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    w8a16_gemv_batchm_impl<8, 2>(A, B, block_scale, C, M, N, K);
}

// N2 variant: each physical thread serves TWO outputs (n0, n1 = adjacent
// columns) on the same two logical lanes — the per-k16 activation loads and
// bf16 unpacks are shared across both outputs, halving the A-side VMEM/VALU
// per output. 32 threads per output × VL2; block covers 16 outputs; grid is
// ceil(N/16). Per-output lane→k16 assignment, accumulate order, and the
// 32-lane shfl reduction are IDENTICAL to w8a16_gemv_batch8_dyn_vl2, so the
// output is bit-identical.
extern "C" __global__ void __launch_bounds__(256) w8a16_gemv_batch8_dyn_vl2_n2(
    const __nv_bfloat16* A,
    const unsigned char* B,
    const float* block_scale,
    __nv_bfloat16* C,
    unsigned int M,
    unsigned int N,
    unsigned int K
) {
    const unsigned int local_out = threadIdx.x / 32;            // 0..7
    const unsigned int lane = threadIdx.x % 32;                 // phys lane
    const unsigned int n0 = blockIdx.x * 16 + local_out * 2;
    const unsigned int n1 = n0 + 1;
    const bool v0 = n0 < N;
    const bool v1 = n1 < N;
    const unsigned int k16_count = K / 16;
    const unsigned int k_blocks = (K + FP8_BLOCK - 1) / FP8_BLOCK;

    __shared__ float e4m3_lut[256];
    e4m3_lut[threadIdx.x] = atlas_e4m3_to_f32((unsigned char)threadIdx.x);
    __syncthreads();

    float acc[2][8] = {}; // [vlane][row] for n0
    float acc_b[2][8] = {}; // [vlane][row] for n1
    const unsigned int nb0 = n0 / FP8_BLOCK, nb1 = n1 / FP8_BLOCK;

    for (unsigned int k16 = lane; k16 < k16_count; k16 += 64) {
        const unsigned int k2 = k16 + 32;
        const bool k2v = k2 < k16_count;
        const unsigned int base_k = k16 * 16;
        const float sc0 = v0 ? block_scale[nb0 * k_blocks + base_k / FP8_BLOCK] : 0.f;
        const float sc1 = v1 ? block_scale[nb1 * k_blocks + base_k / FP8_BLOCK] : 0.f;
        const uint4 p0 = v0 ? reinterpret_cast<const uint4*>(B + (unsigned long long)n0 * K)[k16] : make_uint4(0,0,0,0);
        const uint4 p1 = v1 ? reinterpret_cast<const uint4*>(B + (unsigned long long)n1 * K)[k16] : make_uint4(0,0,0,0);
        float w0[16], w1[16];
        {
            const unsigned int w[4] = {p0.x, p0.y, p0.z, p0.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                w0[j*4]   = e4m3_lut[w[j] & 0xffu] * sc0;
                w0[j*4+1] = e4m3_lut[(w[j] >> 8) & 0xffu] * sc0;
                w0[j*4+2] = e4m3_lut[(w[j] >> 16) & 0xffu] * sc0;
                w0[j*4+3] = e4m3_lut[w[j] >> 24] * sc0;
            }
        }
        {
            const unsigned int w[4] = {p1.x, p1.y, p1.z, p1.w};
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                w1[j*4]   = e4m3_lut[w[j] & 0xffu] * sc1;
                w1[j*4+1] = e4m3_lut[(w[j] >> 8) & 0xffu] * sc1;
                w1[j*4+2] = e4m3_lut[(w[j] >> 16) & 0xffu] * sc1;
                w1[j*4+3] = e4m3_lut[w[j] >> 24] * sc1;
            }
        }
        #pragma unroll
        for (int row = 0; row < 8; ++row) {
            if (row >= M) continue;
            const __nv_bfloat16* arow = A + (unsigned long long)row * K;
            const uint4 lo = reinterpret_cast<const uint4*>(arow)[k16 * 2];
            const uint4 hi = reinterpret_cast<const uint4*>(arow)[k16 * 2 + 1];
            const unsigned int aw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            #pragma unroll
            for (int pair = 0; pair < 8; ++pair) {
                const float a_lo = atlas_bf16_bits_to_f32(aw[pair] & 0xffffu);
                const float a_hi = atlas_bf16_bits_to_f32(aw[pair] >> 16);
                acc[0][row] += a_lo * w0[pair * 2] + a_hi * w0[pair * 2 + 1];
                acc_b[0][row] += a_lo * w1[pair * 2] + a_hi * w1[pair * 2 + 1];
            }
        }
        if (k2v) {
            const unsigned int base2 = k2 * 16;
            const float sc20 = v0 ? block_scale[nb0 * k_blocks + base2 / FP8_BLOCK] : 0.f;
            const float sc21 = v1 ? block_scale[nb1 * k_blocks + base2 / FP8_BLOCK] : 0.f;
            const uint4 q0 = v0 ? reinterpret_cast<const uint4*>(B + (unsigned long long)n0 * K)[k2] : make_uint4(0,0,0,0);
            const uint4 q1 = v1 ? reinterpret_cast<const uint4*>(B + (unsigned long long)n1 * K)[k2] : make_uint4(0,0,0,0);
            float u0[16], u1[16];
            {
                const unsigned int w[4] = {q0.x, q0.y, q0.z, q0.w};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    u0[j*4]   = e4m3_lut[w[j] & 0xffu] * sc20;
                    u0[j*4+1] = e4m3_lut[(w[j] >> 8) & 0xffu] * sc20;
                    u0[j*4+2] = e4m3_lut[(w[j] >> 16) & 0xffu] * sc20;
                    u0[j*4+3] = e4m3_lut[w[j] >> 24] * sc20;
                }
            }
            {
                const unsigned int w[4] = {q1.x, q1.y, q1.z, q1.w};
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    u1[j*4]   = e4m3_lut[w[j] & 0xffu] * sc21;
                    u1[j*4+1] = e4m3_lut[(w[j] >> 8) & 0xffu] * sc21;
                    u1[j*4+2] = e4m3_lut[(w[j] >> 16) & 0xffu] * sc21;
                    u1[j*4+3] = e4m3_lut[w[j] >> 24] * sc21;
                }
            }
            #pragma unroll
            for (int row = 0; row < 8; ++row) {
                if (row >= M) continue;
                const __nv_bfloat16* arow = A + (unsigned long long)row * K;
                const uint4 lo = reinterpret_cast<const uint4*>(arow)[k2 * 2];
                const uint4 hi = reinterpret_cast<const uint4*>(arow)[k2 * 2 + 1];
                const unsigned int aw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
                #pragma unroll
                for (int pair = 0; pair < 8; ++pair) {
                    const float a_lo = atlas_bf16_bits_to_f32(aw[pair] & 0xffffu);
                    const float a_hi = atlas_bf16_bits_to_f32(aw[pair] >> 16);
                    acc[1][row] += a_lo * u0[pair * 2] + a_hi * u0[pair * 2 + 1];
                    acc_b[1][row] += a_lo * u1[pair * 2] + a_hi * u1[pair * 2 + 1];
                }
            }
        }
    }

    // Per output, one warp of 32 physical lanes owns logical lanes p and
    // p+32 — the same two-half reduce as the VL2 single-out kernel.
    #pragma unroll
    for (int row = 0; row < 8; ++row) {
        if (row >= M) continue;
        float x0 = acc[0][row], x1 = acc[1][row];
        float y0 = acc_b[0][row], y1 = acc_b[1][row];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            x0 += __shfl_down_sync(0xffffffffu, x0, off);
            x1 += __shfl_down_sync(0xffffffffu, x1, off);
            y0 += __shfl_down_sync(0xffffffffu, y0, off);
            y1 += __shfl_down_sync(0xffffffffu, y1, off);
        }
        if (lane == 0) {
            if (v0) C[(unsigned long long)row * N + n0] = __float2bfloat16(x0 + x1);
            if (v1) C[(unsigned long long)row * N + n1] = __float2bfloat16(y0 + y1);
        }
    }
}
