// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>
#include "atlas_pdl.cuh"

// DFlash2 grouped dynamic causal convolution over `taps` taps.
//
// Parameters:
//   y: output buffer [K, hidden_size]
//   x: input buffer [K, hidden_size]
//   delta_kernel: this side's dynamic delta kernel, row t at
//                 t * (2 * taps * num_groups): [taps, num_groups]
//                 (the projection emits [K, 2 (input/output), taps, num_groups])
//   base_kernel: this side's static base kernel [taps, hidden_size]
//   K: number of draft tokens in the block
//   hidden_size: model hidden dimension D (e.g. 5120)
//   group_size: number of channels sharing a dynamic delta (e.g. 16)
//   num_groups: D / group_size (e.g. 320)
//   taps: kernel size (checkpoint `conv_kernel_size`, >= 1)
//
// Formulation:
//   For each token position t in [0, K) and channel c in [0, hidden_size):
//     g = c / group_size
//     tap_j = base_kernel[j, c] + delta_kernel[t, j, g]
//     y[t, c] = sum_{j < taps, j <= t} tap_j * x[t - j, c]
//   summed in tap order (tap 0 first), so a 2-tap launch computes exactly the
//   two-term expression this kernel computed before it was generic.
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
    atlas_pdl_enter();
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int total = K * hidden_size;
    if (idx >= total) return;

    const unsigned int t = idx / hidden_size;
    const unsigned int c = idx % hidden_size;
    const unsigned int g = c / group_size;
    const __nv_bfloat16* delta_row = delta_kernel + t * (2 * taps * num_groups);

    // Tap 0: current position t
    float b0 = __bfloat162float(base_kernel[c]);
    float d0 = __bfloat162float(delta_row[g]);
    float k0 = b0 + d0;
    float x0 = __bfloat162float(x[idx]);
    float val = k0 * x0;

    // Taps 1..taps-1: previous positions t - j (causal within block)
    for (unsigned int j = 1; j < taps && j <= t; j++) {
        float bj = __bfloat162float(base_kernel[j * hidden_size + c]);
        float dj = __bfloat162float(delta_row[j * num_groups + g]);
        float kj = bj + dj;
        float xj = __bfloat162float(x[(t - j) * hidden_size + c]);
        val += kj * xj;
    }

    y[idx] = __float2bfloat16(val);
}
