// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// DFlash2 two-tap grouped dynamic causal convolution.
//
// Parameters:
//   y: output buffer [K, hidden_size]
//   x: input buffer [K, hidden_size]
//   delta_kernel: dynamic delta kernel [K, 2, num_groups]
//   base_kernel: static base kernel [2, hidden_size]
//   K: number of draft tokens in the block
//   hidden_size: model hidden dimension D (e.g. 5120)
//   group_size: number of channels sharing a dynamic delta (e.g. 16)
//   num_groups: D / group_size (e.g. 320)
//
// Formulation:
//   For each token position t in [0, K) and channel c in [0, hidden_size):
//     g = c / group_size
//     tap0 = base_kernel[0, c] + delta_kernel[t, 0, g]
//     tap1 = base_kernel[1, c] + delta_kernel[t, 1, g]
//     y[t, c] = tap0 * x[t, c] + (t > 0 ? tap1 * x[t - 1, c] : 0)
//
// Causal boundary: convolution never crosses token 0 (the block boundary).

extern "C" __global__ void dflash2_grouped_dynamic_causal_conv(
    __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ delta_kernel,
    const __nv_bfloat16* __restrict__ base_kernel,
    unsigned int K,
    unsigned int hidden_size,
    unsigned int group_size,
    unsigned int num_groups
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = K * hidden_size;
    if (idx >= total) return;

    const unsigned int t = idx / hidden_size;
    const unsigned int c = idx % hidden_size;
    const unsigned int g = c / group_size;

    // Tap 0: current position t
    float b0 = __bfloat162float(base_kernel[c]);
    float d0 = __bfloat162float(delta_kernel[t * (4 * num_groups) + g]);
    float k0 = b0 + d0;
    float x0 = __bfloat162float(x[idx]);
    float val = k0 * x0;

    // Tap 1: previous position t - 1 (causal within block)
    if (t > 0) {
        float b1 = __bfloat162float(base_kernel[hidden_size + c]);
        float d1 = __bfloat162float(delta_kernel[t * (4 * num_groups) + num_groups + g]);
        float k1 = b1 + d1;
        float x1 = __bfloat162float(x[(t - 1) * hidden_size + c]);
        val += k1 * x1;
    }

    y[idx] = __float2bfloat16(val);
}
