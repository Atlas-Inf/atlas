// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// DFlash2 grouped dynamic causal convolution (runtime kernel size k).
//
// Parameters:
//   y: output buffer [K, hidden_size]
//   x: input buffer [K, hidden_size]
//   delta_kernel: dynamic delta kernel for ONE io side [K, k, num_groups]
//   base_kernel: static base kernel for ONE io side [k, hidden_size]
//   K: number of draft tokens in the block
//   hidden_size: model hidden dimension D (e.g. 5120)
//   group_size: number of channels sharing a dynamic delta (e.g. 16)
//   num_groups: D / group_size (e.g. 320)
//   kernel_size: number of taps k (2 for Qwen3.8-DFlash2, 4 for Ripple-65B)
//
// Formulation:
//   For each token position t in [0, K) and channel c in [0, hidden_size):
//     g = c / group_size
//     y[t, c] = Σ_{s in [0,k)} (base_kernel[s, c] + delta_kernel[t, s, g])
//               * (t >= s ? x[t - s, c] : 0)
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
    unsigned int num_groups,
    unsigned int kernel_size
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = K * hidden_size;
    if (idx >= total) return;

    const unsigned int t = idx / hidden_size;
    const unsigned int c = idx % hidden_size;
    const unsigned int g = c / group_size;

    float val = 0.0f;
    const unsigned int taps = (t < kernel_size) ? (t + 1) : kernel_size;
    // Each delta row holds BOTH io sides: [io][tap][group] = 2*k*groups.
    const unsigned int delta_row = t * (2 * kernel_size * num_groups);
    for (unsigned int s = 0; s < taps; s++) {
        const float b = __bfloat162float(base_kernel[s * hidden_size + c]);
        const float d = __bfloat162float(delta_kernel[delta_row + s * num_groups + g]);
        const float xv = __bfloat162float(x[(t - s) * hidden_size + c]);
        val += (b + d) * xv;
    }

    y[idx] = __float2bfloat16(val);
}
