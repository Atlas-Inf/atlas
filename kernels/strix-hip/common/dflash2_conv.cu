// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// DFlash2 grouped dynamic causal convolution, `taps` taps (2 on the Qwen
// drafters, 4 on Ripple's).
//
// Parameters:
//   y: output buffer [K, hidden_size]
//   x: input buffer [K, hidden_size]
//   delta_kernel: dynamic delta kernel rows [K, 2 * taps * num_groups] (input half or output half)
//   base_kernel: static base kernel [taps, hidden_size] (input or output side)
//   K: number of draft tokens in the block
//   hidden_size: model hidden dimension D (e.g. 5120)
//   group_size: number of channels sharing a dynamic delta (e.g. 16)
//   num_groups: D / group_size (e.g. 320)
//
// Formulation:
//   For each token position t in [0, K) and channel c in [0, hidden_size):
//     g = c / group_size
//     tap_j = base_kernel[j, c] + delta_kernel[t, j, g]
//     y[t, c] = sum_{j < taps, j <= t} tap_j * x[t - j, c]
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
    unsigned int taps
) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = K * hidden_size;
    if (idx >= total) return;

    const unsigned int t = idx / hidden_size;
    const unsigned int c = idx % hidden_size;
    const unsigned int g = c / group_size;
    // One delta row is [input taps | output taps] x num_groups; the caller
    // passes the input or output half, both strided by the full row.
    const unsigned int row_stride = 2 * taps * num_groups;

    // Tap j weights position t - j (causal within the block). Tap 0 seeds the
    // sum so the 2-tap drafters keep the former kernel's exact arithmetic.
    float val = (__bfloat162float(base_kernel[c]) + __bfloat162float(delta_kernel[t * row_stride + g]))
              * __bfloat162float(x[idx]);
    for (unsigned int j = 1; j < taps && j <= t; ++j) {
        float b = __bfloat162float(base_kernel[j * hidden_size + c]);
        float d = __bfloat162float(delta_kernel[t * row_stride + j * num_groups + g]);
        float xv = __bfloat162float(x[(t - j) * hidden_size + c]);
        val += (b + d) * xv;
    }

    y[idx] = __float2bfloat16(val);
}
