// SPDX-License-Identifier: AGPL-3.0-only
//
// Deinterleave Qwen3.5's q_proj output into separate Q and gate
// buffers. Upstream MLX layout (qwen3_next.Qwen3NextAttention):
//
//   q_proj(x): [B, L, num_heads * head_dim * 2]
//   reshape:   [B, L, num_heads, head_dim * 2]
//   split:     queries = [..., :head_dim],   gate = [..., head_dim:]
//
// In flat 1-D terms (single token), q_proj output is laid out as
// `[Q_h0_d0..d255, gate_h0_d0..d255, Q_h1_d0..d255, gate_h1_d0..d255, ...]`
// — interleaved per head. This kernel separates them into two
// contiguous `[num_heads, head_dim]` buffers so the existing
// per-head q_norm / rope / attention kernels can consume them
// without per-head stride awareness.

#include <metal_stdlib>
using namespace metal;

kernel void qwen35_qkv_split(
    constant uint &num_heads [[buffer(0)]],
    constant uint &head_dim  [[buffer(1)]],
    device const bfloat *q_full [[buffer(2)]],
    device bfloat       *q_out  [[buffer(3)]],
    device bfloat       *gate_out [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]])
{
    uint h = gid.y;
    uint d = gid.x;
    if (h >= num_heads || d >= head_dim) {
        return;
    }
    uint stride = 2u * head_dim;
    uint q_src    = h * stride + d;
    uint gate_src = h * stride + head_dim + d;
    uint dst      = h * head_dim + d;
    q_out[dst]    = q_full[q_src];
    gate_out[dst] = q_full[gate_src];
}
