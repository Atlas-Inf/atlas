// SPDX-License-Identifier: AGPL-3.0-only

// Grammar bitmask application for BF16 logits rows.
//
// Writes bf16(-inf) over every disallowed token id in ONE logits row, so the
// downstream argmax / DFlash2 candidate-selector top-k can only pick a
// grammar-legal token. XGrammar bitmask convention (grammar/state.rs): bit
// `t` of word `t >> 5` set = ALLOWED; clear = masked.
//
// Grid: (ceil(vocab / blockDim), 1, 1) — one launch per logits row.
// The caller launches it only for rows 0 and 1 (the position-0 predictions)
// and only when a mask exists — a null mask adds no launch.

#include <cuda_bf16.h>

extern "C" __global__ void atlas_apply_grammar_bitmask(
    __nv_bfloat16* __restrict__ logits,
    const int* __restrict__ bitmask,
    unsigned int vocab
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= vocab) return;
    const int word = bitmask[i >> 5];
    if (((word >> (i & 31)) & 1) == 0) {
        logits[i] = __float2bfloat16(-INFINITY);
    }
}
