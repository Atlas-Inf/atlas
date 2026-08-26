// SPDX-License-Identifier: AGPL-3.0-only
//
// Token embedding gather. For each (token, hidden) pair, copy one
// bfloat from the embedding table into the output. Mirrors the
// CUDA `embed_from_argmax` kernel's signature so spark-model code
// that talks to the GPU through the kernel-name lookup needs no
// per-backend conditional logic.
//
// Layout:
//   token_ids   : uint32 [num_tokens]
//   embed_table : bfloat [vocab_size, hidden_size]
//   out         : bfloat [num_tokens, hidden_size]

#include <metal_stdlib>
using namespace metal;

kernel void embed_lookup(
    constant uint &num_tokens   [[buffer(0)]],
    constant uint &hidden_size  [[buffer(1)]],
    constant uint &vocab_size   [[buffer(2)]],
    device const uint   *token_ids   [[buffer(3)]],
    device const bfloat *embed_table [[buffer(4)]],
    device bfloat       *out         [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint tok_idx = gid.y;
    uint hid_idx = gid.x;
    if (tok_idx >= num_tokens || hid_idx >= hidden_size) {
        return;
    }
    uint v = token_ids[tok_idx];
    if (v >= vocab_size) {
        // Out-of-range tokens get zero — matches the CUDA path's
        // bounds-check behaviour and keeps the prefill loop robust
        // against malformed input.
        out[tok_idx * hidden_size + hid_idx] = bfloat(0.0);
        return;
    }
    out[tok_idx * hidden_size + hid_idx] =
        embed_table[v * hidden_size + hid_idx];
}
