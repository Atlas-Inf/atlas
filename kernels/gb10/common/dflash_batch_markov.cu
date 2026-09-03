// SPDX-License-Identifier: AGPL-3.0-only

#include <cuda_bf16.h>

// Add contiguous [B,V] Markov bias to one depth of row-major [B,gamma,V].
extern "C" __global__ void dflash_batch_add_depth_bias(
    __nv_bfloat16* __restrict__ logits,
    const __nv_bfloat16* __restrict__ bias,
    unsigned int batch,
    unsigned int gamma,
    unsigned int vocab,
    unsigned int depth
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned long long total = (unsigned long long)batch * vocab;
    if ((unsigned long long)index >= total) return;
    const unsigned int sequence = index / vocab;
    const unsigned int token = index - sequence * vocab;
    const unsigned long long row = (unsigned long long)sequence * gamma + depth;
    const unsigned long long dst = row * vocab + token;
    logits[dst] = __float2bfloat16(
        __bfloat162float(logits[dst]) + __bfloat162float(bias[index])
    );
}

// Scatter contiguous [B] sampled IDs into depth rows of [B,gamma].
extern "C" __global__ void dflash_batch_store_depth_tokens(
    unsigned int* __restrict__ tokens,
    const unsigned int* __restrict__ sampled,
    unsigned int batch,
    unsigned int gamma,
    unsigned int depth
) {
    const unsigned int sequence = blockIdx.x * blockDim.x + threadIdx.x;
    if (sequence < batch) {
        tokens[sequence * gamma + depth] = sampled[sequence];
    }
}
