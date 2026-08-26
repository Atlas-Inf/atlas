// SPDX-License-Identifier: AGPL-3.0-only
//
// Element-wise BF16 addition: `out[i] = a[i] + b[i]`.
//
// Used for residual connections (`x = x + attn_out`,
// `x = x + ffn_out`) at every transformer layer. Trivial kernel,
// but its absence is the only thing standing between the existing
// kernel set and a complete attention-block forward.
//
// Internal accumulation in FP32 so the residual stream's tail
// digits don't lose precision to the BF16 intermediate.
//
// Layout:
//   a, b, out : bfloat [n]

#include <metal_stdlib>
using namespace metal;

kernel void bf16_add(
    constant uint &n         [[buffer(0)]],
    device const bfloat *a   [[buffer(1)]],
    device const bfloat *b   [[buffer(2)]],
    device bfloat       *out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    out[gid] = bfloat(float(a[gid]) + float(b[gid]));
}
