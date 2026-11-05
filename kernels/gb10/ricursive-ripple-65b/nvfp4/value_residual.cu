// SPDX-License-Identifier: AGPL-3.0-only

// Ricursive value residual: out = lerp(x0, h, w), elementwise, BF16 in/out.
//
// torch.lerp's CUDA kernel (opmath = float for BF16):
//     w < 0.5 ? start + w * (end - start) : end - (end - start) * (1 - w)
// rounded once to BF16. `w` is already bf16(sigmoid(lambda)) on the host.
// This file builds with --fmad=false, so the two forms round as torch's do.
//
// Grid: (ceil(N/256), 1, 1)  Block: (256, 1, 1)

#include <cuda_bf16.h>

extern "C" __global__ void value_residual_lerp(
    const __nv_bfloat16* __restrict__ x0,
    const __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ out,
    unsigned int n,
    float w
) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float start = __bfloat162float(x0[i]);
    float end = __bfloat162float(h[i]);
    float r = (w < 0.5f) ? start + w * (end - start) : end - (end - start) * (1.0f - w);
    out[i] = __float2bfloat16(r);
}
