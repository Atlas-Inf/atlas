// SPDX-License-Identifier: AGPL-3.0-only
//
// Sigmoid-gated elementwise multiply: `out[i] = sigmoid(gate[i]) * x[i]`.
//
// Qwen3.5 uses this as `attn_output_gate` — half of the q_proj output
// is a gate that scales the attention result before the o_proj.
// Distinct from `silu_gate` which uses `silu = x * sigmoid(x)`
// instead of plain `sigmoid`.
//
// Layout:
//   gate, x, out : bfloat [n]

#include <metal_stdlib>
using namespace metal;

kernel void sigmoid_gate(
    constant uint &n         [[buffer(0)]],
    device const bfloat *gate [[buffer(1)]],
    device const bfloat *x   [[buffer(2)]],
    device bfloat       *out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float g = float(gate[gid]);
    float v = float(x[gid]);
    float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = bfloat(sig * v);
}
