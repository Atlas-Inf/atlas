// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// Eight lanes reduce one state column. Loading four time steps ahead overlaps
// independent memory reads with the sequential updates to the FP32 state.
extern "C" __global__ void ling_kda_subwarp(
    const __nv_bfloat16* __restrict__ qkv, const float* __restrict__ query,
    const float* __restrict__ key, const float* __restrict__ decay,
    const float* __restrict__ beta, float* __restrict__ state,
    __nv_bfloat16* __restrict__ output, int tokens, int heads, int dim) {
    constexpr int width = 8, time_tile = 4, head_dim = 128;
    constexpr int entries = head_dim / width;
    if (dim != head_dim) return;
    const int head = blockIdx.x;
    const int lane = threadIdx.x % width;
    const int column = blockIdx.y * (128 / width) + threadIdx.x / width;

    float values[entries];
    #pragma unroll
    for (int i = 0; i < entries; ++i) {
        values[i] = state[(head * head_dim + lane + i * width) * head_dim + column];
    }
    for (int first = 0; first < tokens; first += time_tile) {
        float qs[time_tile][entries], ks[time_tile][entries], ds[time_tile][entries];
        float vs[time_tile], betas[time_tile];
        #pragma unroll
        for (int t = 0; t < time_tile; ++t) {
            const int row = min(first + t, tokens - 1);
            const int base = (row * heads + head) * head_dim;
            #pragma unroll
            for (int i = 0; i < entries; ++i) {
                const int index = base + lane + i * width;
                qs[t][i] = query[index];
                ks[t][i] = key[index];
                ds[t][i] = decay[index];
            }
            vs[t] = float(qkv[((row * 3 + 2) * heads + head) * head_dim + column]);
            betas[t] = beta[row * heads + head];
        }
        #pragma unroll
        for (int t = 0; t < time_tile; ++t) {
            if (first + t == tokens) break;
            float prediction = 0;
            #pragma unroll
            for (int i = 0; i < entries; ++i) {
                values[i] *= ds[t][i];
                prediction += values[i] * ks[t][i];
            }
            #pragma unroll
            for (int step = width / 2; step; step >>= 1)
                prediction += __shfl_xor_sync(0xffffffff, prediction, step, width);
            const float correction = (vs[t] - prediction) * betas[t];
            float result = 0;
            #pragma unroll
            for (int i = 0; i < entries; ++i) {
                values[i] += correction * ks[t][i];
                result += values[i] * qs[t][i];
            }
            #pragma unroll
            for (int step = width / 2; step; step >>= 1)
                result += __shfl_xor_sync(0xffffffff, result, step, width);
            if (lane == 0)
                output[((first + t) * heads + head) * head_dim + column] = __float2bfloat16(result);
        }
    }
    #pragma unroll
    for (int i = 0; i < entries; ++i) {
        state[(head * head_dim + lane + i * width) * head_dim + column] = values[i];
    }
}
