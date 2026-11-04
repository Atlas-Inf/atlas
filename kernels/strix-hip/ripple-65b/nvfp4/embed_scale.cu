// SPDX-License-Identifier: AGPL-3.0-only

// Embedding scale: output[i] *= scale (Ripple: embed_scale = sqrt(5120)).
//
// Same kernel as gb10 gemma-4-*/nvfp4/embed_scale.cu. The strix-hip tree had
// no copy because no strix-hip target scaled its embeddings before Ripple;
// without it the model silently ran on 1/71.55-sized embeddings.
//
// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void bf16_scale_inplace(
    __nv_bfloat16* __restrict__ data,
    unsigned int N,
    float scale
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(data[idx]);
    data[idx] = __float2bfloat16(x * scale);
}
