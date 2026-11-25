// SPDX-License-Identifier: AGPL-3.0-only

// BailingMoeV3VL's text-side linear_proj uses torch.nn.GELU() with its
// default approximate="none".  The Qwen vision blocks still use the
// checkpoint-declared gelu_pytorch_tanh, so this exact-erf projector kernel
// must remain a separate symbol.
#include <cuda_bf16.h>
#include <math.h>

extern "C" __global__ void ling_projector_gelu_exact(
    __nv_bfloat16* __restrict__ x,
    unsigned int n
) {
    const unsigned int i = blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    const float value = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(0.5f * value * (1.0f + erff(value * 0.7071067811865475f)));
}
