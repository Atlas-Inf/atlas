// SPDX-License-Identifier: AGPL-3.0-only
//
// Rotary position embedding (RoPE) — applied in-place to a Q or K
// tensor.
//
// GPT-NeoX layout (Qwen / Llama lineage). Pairs are `(d, d + rotary_dim/2)`
// for d in [0, rotary_dim/2):
//
//   theta = pos * inv_freq[d]
//   x[d]                  =  x_old[d] * cos(theta) - x_old[d + rotary_dim/2] * sin(theta)
//   x[d + rotary_dim/2]   =  x_old[d] * sin(theta) + x_old[d + rotary_dim/2] * cos(theta)
//
// `inv_freq` is precomputed (one entry per dimension pair) on the
// host: `inv_freq[i] = 1.0 / (rope_theta ^ (2i / rotary_dim))`.
//
// Partial RoPE: when `rotary_dim < head_dim`, only the first
// `rotary_dim` elements of each head are rotated; dims
// `[rotary_dim, head_dim)` pass through untouched. Qwen3.5-VL uses
// `partial_rotary_factor = 0.25` → `rotary_dim = head_dim / 4`.
// Pass `rotary_dim = head_dim` for the full-rotation classic dense
// path (Llama, Qwen3 dense).
//
// Layout:
//   x         : bfloat [num_tokens, num_heads, head_dim]
//   inv_freq  : float  [rotary_dim / 2]
//   positions : uint32 [num_tokens]
//
// Grid: (rotary_dim/2 threads, num_heads, num_tokens)

#include <metal_stdlib>
using namespace metal;

kernel void rope_apply(
    constant uint  &num_tokens [[buffer(0)]],
    constant uint  &num_heads  [[buffer(1)]],
    constant uint  &head_dim   [[buffer(2)]],
    constant uint  &rotary_dim [[buffer(3)]],
    device const uint   *positions [[buffer(4)]],
    device const float  *inv_freq  [[buffer(5)]],
    device bfloat       *x         [[buffer(6)]],
    uint3 gid [[thread_position_in_grid]])
{
    uint d  = gid.x;          // 0 .. rotary_dim/2
    uint h  = gid.y;          // head index
    uint tok = gid.z;         // token index
    uint half_rot = rotary_dim >> 1u;
    if (d >= half_rot || h >= num_heads || tok >= num_tokens) {
        return;
    }

    uint pos = positions[tok];
    float theta = float(pos) * inv_freq[d];
    float c = cos(theta);
    float s = sin(theta);

    uint base = (tok * num_heads + h) * head_dim;
    uint i_lo = base + d;
    uint i_hi = base + d + half_rot;
    float lo = float(x[i_lo]);
    float hi = float(x[i_hi]);

    x[i_lo] = bfloat(lo * c - hi * s);
    x[i_hi] = bfloat(lo * s + hi * c);
}
