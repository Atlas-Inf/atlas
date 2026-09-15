// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>
#include "gdn_reduce_wyn.cuh"

#define WYN_BLOCK_SIZE 128

template <int K_TOKENS>
__device__ __forceinline__ void gated_delta_rule_wyn_impl(
    float* __restrict__ h_state,
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output,
    float* __restrict__ h_state_inter_base,
    unsigned int inter_stride_floats,
    unsigned int batch_size,
    unsigned int num_k_heads,
    unsigned int num_v_heads,
    unsigned int k_dim,
    unsigned int v_dim,
    unsigned int qk_stride,
    unsigned int v_stride,
    unsigned int gb_stride
) {
    const unsigned int value_head = blockIdx.x;
    const unsigned int batch = blockIdx.y;
    if (value_head >= num_v_heads || batch >= batch_size) return;

    const unsigned int thread = threadIdx.x;
    const unsigned int heads_per_key = num_v_heads / num_k_heads;
    const unsigned int key_head = value_head / heads_per_key;
    const unsigned int state_size = k_dim * v_dim;
    float* state = h_state + ((batch * num_v_heads + value_head) * state_size);
    float* intermediate = h_state_inter_base + ((batch * num_v_heads + value_head) * state_size);

    __shared__ float keys[K_TOKENS][128];
    __shared__ float queries[K_TOKENS][128];
    __shared__ float gates[K_TOKENS];
    __shared__ float betas[K_TOKENS];
    __shared__ float warp_sums[4];

    if (thread < k_dim) {
        #pragma unroll
        for (int token = 0; token < K_TOKENS; ++token) {
            const __nv_bfloat16* query_row =
                query + (batch * K_TOKENS + token) * qk_stride + key_head * k_dim;
            const __nv_bfloat16* key_row =
                key + (batch * K_TOKENS + token) * qk_stride + key_head * k_dim;
            queries[token][thread] = __bfloat162float(query_row[thread]);
            keys[token][thread] = __bfloat162float(key_row[thread]);
        }
    }
    if (thread < K_TOKENS) {
        const float raw_gate = gate[(batch * K_TOKENS + thread) * gb_stride + value_head];
        gates[thread] = fminf(fmaxf(raw_gate, 1e-6f), 1.0f - 1e-6f);
        betas[thread] = beta[(batch * K_TOKENS + thread) * gb_stride + value_head];
    }
    __syncthreads();

    __shared__ float key_dots[K_TOKENS * (K_TOKENS - 1) / 2];
    #pragma unroll
    for (int token = 1; token < K_TOKENS; ++token) {
        #pragma unroll
        for (int previous = 0; previous < token; ++previous) {
            const float product = thread < k_dim
                ? keys[token][thread] * keys[previous][thread]
                : 0.0f;
            const float sum = atlas_wyn_block_sum(product, warp_sums, thread);
            if (thread == 0)
                key_dots[token * (token - 1) / 2 + previous] = sum;
            __syncthreads();
        }
    }

    if (thread < v_dim) {
        float values[K_TOKENS];
        #pragma unroll
        for (int token = 0; token < K_TOKENS; ++token) {
            const __nv_bfloat16* value_row =
                value + (batch * K_TOKENS + token) * v_stride + value_head * v_dim;
            values[token] = __bfloat162float(value_row[thread]);
        }

        float state_key[K_TOKENS];
        #pragma unroll
        for (int token = 0; token < K_TOKENS; ++token) state_key[token] = 0.0f;
        #pragma unroll 4
        for (unsigned int index = 0; index < k_dim; index += 4) {
            const float h0 = state[(index + 0) * v_dim + thread];
            const float h1 = state[(index + 1) * v_dim + thread];
            const float h2 = state[(index + 2) * v_dim + thread];
            const float h3 = state[(index + 3) * v_dim + thread];
            #pragma unroll
            for (int token = 0; token < K_TOKENS; ++token) {
                state_key[token] += h0 * keys[token][index + 0]
                                  + h1 * keys[token][index + 1]
                                  + h2 * keys[token][index + 2]
                                  + h3 * keys[token][index + 3];
            }
        }

        float value_delta[K_TOKENS];
        value_delta[0] = (values[0] - gates[0] * state_key[0]) * betas[0];
        for (int token = 1; token < K_TOKENS; ++token) {
            float leading = 1.0f;
            for (int index = 0; index < token; ++index) leading *= gates[index];
            float corrected = leading * state_key[token];
            for (int previous = 0; previous < token; ++previous) {
                float product = 1.0f;
                for (int index = previous + 1; index < token; ++index)
                    product *= gates[index];
                corrected += product * key_dots[token * (token - 1) / 2 + previous]
                           * value_delta[previous];
            }
            value_delta[token] = (values[token] - gates[token] * corrected) * betas[token];
        }

        float query_dot[K_TOKENS];
        #pragma unroll
        for (int token = 0; token < K_TOKENS; ++token) query_dot[token] = 0.0f;
        #pragma unroll 4
        for (unsigned int index = 0; index < k_dim; index += 4) {
            float h0 = state[(index + 0) * v_dim + thread];
            float h1 = state[(index + 1) * v_dim + thread];
            float h2 = state[(index + 2) * v_dim + thread];
            float h3 = state[(index + 3) * v_dim + thread];
            #pragma unroll
            for (int token = 0; token < K_TOKENS; ++token) {
                h0 = gates[token] * h0 + keys[token][index + 0] * value_delta[token];
                h1 = gates[token] * h1 + keys[token][index + 1] * value_delta[token];
                h2 = gates[token] * h2 + keys[token][index + 2] * value_delta[token];
                h3 = gates[token] * h3 + keys[token][index + 3] * value_delta[token];
                if (token < K_TOKENS - 1) {
                    float* token_state = intermediate + token * inter_stride_floats;
                    token_state[(index + 0) * v_dim + thread] = h0;
                    token_state[(index + 1) * v_dim + thread] = h1;
                    token_state[(index + 2) * v_dim + thread] = h2;
                    token_state[(index + 3) * v_dim + thread] = h3;
                } else {
                    state[(index + 0) * v_dim + thread] = h0;
                    state[(index + 1) * v_dim + thread] = h1;
                    state[(index + 2) * v_dim + thread] = h2;
                    state[(index + 3) * v_dim + thread] = h3;
                }
                query_dot[token] += h0 * queries[token][index + 0]
                                  + h1 * queries[token][index + 1]
                                  + h2 * queries[token][index + 2]
                                  + h3 * queries[token][index + 3];
            }
        }

        const float normalization = rsqrtf((float)k_dim);
        #pragma unroll
        for (int token = 0; token < K_TOKENS; ++token) {
            output[((batch * K_TOKENS + token) * num_v_heads + value_head) * v_dim + thread] =
                __float2bfloat16(query_dot[token] * normalization);
        }
    }
}

#define INSTANTIATE_WYN(K) \
    extern "C" __global__ void gated_delta_rule_wy##K( \
        float* h_state, const __nv_bfloat16* query, const __nv_bfloat16* key, \
        const __nv_bfloat16* value, const float* gate, const float* beta, \
        __nv_bfloat16* output, float* h_state_inter_base, \
        unsigned int inter_stride_floats, unsigned int batch_size, \
        unsigned int num_k_heads, unsigned int num_v_heads, \
        unsigned int k_dim, unsigned int v_dim, unsigned int qk_stride, \
        unsigned int v_stride, unsigned int gb_stride) { \
        gated_delta_rule_wyn_impl<K>(h_state, query, key, value, gate, beta, output, \
            h_state_inter_base, inter_stride_floats, batch_size, num_k_heads, \
            num_v_heads, k_dim, v_dim, qk_stride, v_stride, gb_stride); \
    }

INSTANTIATE_WYN(5)
INSTANTIATE_WYN(6)
INSTANTIATE_WYN(7)
INSTANTIATE_WYN(8)

#undef INSTANTIATE_WYN
