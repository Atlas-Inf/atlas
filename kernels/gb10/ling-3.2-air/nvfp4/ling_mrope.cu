// SPDX-License-Identifier: AGPL-3.0-only
#include <cuda_bf16.h>

// Ling 3.2 / Bailing-MoE-v3 body-diagonal MRoPE for the q/k RoPE slices of
// MLA.  The checkpoint's `mrope_section=[8,12,12]` is not the Qwen3.6
// round-robin layout above.  After the reference `apply_video_rope` transform
// and adjacent-pair rotation, pair i owns:
//   i < H+W: H for even i, W for odd i;
//   i >= H+W: T.
// The frequency remains inv_freq[i].  Q and K are the one-token compact MLA
// slices: Q=[num_q_heads,rotary_dim], K=[rotary_dim].  Position streams are
// signed two's-complement values carried in Atlas's u32 metadata buffers;
// negative centered H/W coordinates are valid in the reference.
extern "C" __global__ void ling_mrope_rope_bf16(
    __nv_bfloat16* __restrict__ Q,
    __nv_bfloat16* __restrict__ K,
    const unsigned int* __restrict__ pos_t_bits,
    const unsigned int* __restrict__ pos_h_bits,
    const unsigned int* __restrict__ pos_w_bits,
    const unsigned int num_q_heads,
    const unsigned int head_dim,
    const unsigned int rotary_dim,
    const float* __restrict__ inv_freq,
    const unsigned int t_section,
    const unsigned int h_section,
    const unsigned int w_section
) {
    const unsigned int pair_idx = threadIdx.x;
    const unsigned int pairs = rotary_dim / 2;
    if (pair_idx >= pairs || 2 * pair_idx + 1 >= head_dim) return;

    const unsigned int hw_pairs = h_section + w_section;
    const int pos_t = static_cast<int>(*pos_t_bits);
    const int pos_h = static_cast<int>(*pos_h_bits);
    const int pos_w = static_cast<int>(*pos_w_bits);
    int position;
    if (pair_idx < hw_pairs) {
        position = (pair_idx & 1u) == 0u ? pos_h : pos_w;
    } else {
        // Keep the explicit section argument in the ABI: it documents the
        // model layout and guards accidental use with a mismatched table.
        if (pair_idx >= hw_pairs + t_section) return;
        position = pos_t;
    }

    const float angle = static_cast<float>(position) * inv_freq[pair_idx];
    const float c = cosf(angle);
    const float s = sinf(angle);
    const unsigned int d0 = 2 * pair_idx;
    const unsigned int d1 = d0 + 1;

    for (unsigned int head = blockIdx.x; head < num_q_heads + 1;
         head += gridDim.x) {
        __nv_bfloat16* ptr = head < num_q_heads
            ? Q + static_cast<size_t>(head) * head_dim
            : K;
        const float x0 = static_cast<float>(ptr[d0]);
        const float x1 = static_cast<float>(ptr[d1]);
        ptr[d0] = __float2bfloat16(x0 * c - x1 * s);
        ptr[d1] = __float2bfloat16(x1 * c + x0 * s);
    }
}
