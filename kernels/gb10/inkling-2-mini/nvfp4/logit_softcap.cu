// SPDX-License-Identifier: AGPL-3.0-only

// Logit softcapping: logits = cap * tanh(logits / cap), in place on BF16 logits.
// Copied unchanged from gemma-4-26b-a4b/nvfp4/logit_softcap.cu (this repository);
// Inkling 2 Mini caps at final_logit_softcapping = 30.
//
// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void logit_softcap_bf16(
    __nv_bfloat16* __restrict__ logits,
    unsigned int N,
    float inv_cap,   // 1.0 / cap
    float cap        // softcap value (e.g. 30.0)
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float x = __bfloat162float(logits[idx]);
    float y = cap * tanhf(x * inv_cap);
    logits[idx] = __float2bfloat16(y);
}
