// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Ling-3.2-air MLA rope helpers (sm_121, shared by the gated-MLA kernels).
//
// Ground truth, context/modeling_bailing_moe_v3_vl.py:
//   - BailingMoeV3VLRotaryEmbedding forces head_dim = qk_rope_head_dim (64)
//     and partial_rotary_factor = 1.0 on that slice, so the whole 64-dim rope
//     section rotates (partial_rotary_factor 0.5 in the checkpoint's
//     rope_parameters is NOT used here — see the comment there).
//   - inv_freq = theta ** -(arange(0, 64, 2) / 64): 32 pair frequencies,
//     pair p owns channels (2p, 2p+1) of the rope slice.
//   - rope_interleave defaults true (configuration_bailing_moe_v3_vl.py),
//     the checkpoint's text_config does not override it, and
//     apply_rotary_pos_emb_interleave confirms the pair-interleaved channel
//     layout [x0, y0, x1, y1, ...] (not rotate-half).
//   - Text serving collapses M-RoPE section [8, 12, 12] to one position
//     stream, so per-pair freq_p * pos reproduces the reference tables
//     exactly; kernels compute the angle in-kernel (fp32) like the house
//     rope.cu / rope_mrope_interleaved.cu convention.
//
// Scale 1/attention_scaling (the rope init returns 1.0) applies nowhere:
// the softmax scale stays qk_head_dim ** -0.5 from the attention class.

#include <cuda_bf16.h>
#include <math.h>

// inv_freq[p] = theta ** (-2p / rotary_dim), p in [0, rotary_dim/2).
__device__ __forceinline__ float ling_mla_rope_pair_freq(
    unsigned int pair, unsigned int rotary_dim, double theta
) {
    return (float)(1.0 / pow(theta, (double)(2u * pair) / (double)rotary_dim));
}

// Pair-interleaved rotation of channels (even, odd) = (x_{2p}, x_{2p+1}).
__device__ __forceinline__ void ling_mla_rope_rotate_pair(
    float even, float odd, float cos_v, float sin_v, float& out_even, float& out_odd
) {
    out_even = even * cos_v - odd * sin_v;
    out_odd = odd * cos_v + even * sin_v;
}
