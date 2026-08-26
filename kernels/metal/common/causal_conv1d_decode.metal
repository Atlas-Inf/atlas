// SPDX-License-Identifier: AGPL-3.0-only
//
// Single-token causal 1-D depthwise convolution for decode. Each
// channel keeps a `kernel_size - 1` element ring buffer of past
// inputs (`conv_state`); a new token's contribution is the dot of
// that buffer (with the new input appended) against the per-channel
// weight vector.
//
// Used as the front of the Qwen3.5 GDN linear-attention block:
// `conv1d.weight` has shape `[num_channels, kernel_size, 1]` (depth-
// wise separable), and the decode-time conv state lives in
// `[num_channels, kernel_size - 1]`.
//
// Layout:
//   weights     : bfloat [num_channels, kernel_size]    (squeeze out W=1)
//   new_input   : bfloat [num_channels]
//   conv_state  : bfloat [num_channels, kernel_size - 1]    (in-place)
//   output      : bfloat [num_channels]
//
// Per-channel state shift is in-place; we copy the row to stack
// first to avoid the read-after-write hazard.

#include <metal_stdlib>
using namespace metal;

constant uint MAX_K = 8;

kernel void causal_conv1d_decode(
    constant uint &num_channels [[buffer(0)]],
    constant uint &kernel_size  [[buffer(1)]],
    device const bfloat *weights    [[buffer(2)]],
    device const bfloat *new_input  [[buffer(3)]],
    device bfloat       *conv_state [[buffer(4)]],
    device bfloat       *output     [[buffer(5)]],
    uint c [[thread_position_in_grid]])
{
    if (c >= num_channels) {
        return;
    }
    if (kernel_size < 1u || kernel_size > MAX_K) {
        return;
    }

    uint k = kernel_size;
    uint state_len = k - 1u;

    // Snapshot past values + the new input into a stack buffer so
    // the shift below doesn't clobber values we still need to read
    // for the conv dot product.
    float past[MAX_K];
    for (uint i = 0; i < state_len; ++i) {
        past[i] = float(conv_state[c * state_len + i]);
    }
    past[state_len] = float(new_input[c]);

    // Output: dot product over kernel_size taps.
    float acc = 0.0f;
    for (uint i = 0; i < k; ++i) {
        acc += float(weights[c * k + i]) * past[i];
    }
    output[c] = bfloat(acc);

    // State update: shift left by one (drop oldest), append new.
    for (uint i = 0; i < state_len; ++i) {
        conv_state[c * state_len + i] = bfloat(past[i + 1u]);
    }
}
