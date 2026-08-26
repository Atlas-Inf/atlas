// SPDX-License-Identifier: AGPL-3.0-only
//
// SwiGLU FFN activation, fused into a single pass:
//
//   out[i] = silu(gate[i]) * up[i]
//          = (gate[i] * sigmoid(gate[i])) * up[i]
//
// One thread per element. FP32 internal so the sigmoid stays
// numerically clean for large negative inputs (where naive
// `exp(-x)` could overflow when accumulated).
//
// Layout:
//   gate : bfloat [n]
//   up   : bfloat [n]
//   out  : bfloat [n]

#include <metal_stdlib>
using namespace metal;

kernel void silu_gate(
    constant uint &n           [[buffer(0)]],
    device const bfloat *gate  [[buffer(1)]],
    device const bfloat *up    [[buffer(2)]],
    device bfloat       *out   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float g = float(gate[gid]);
    float u = float(up[gid]);
    // Stable SiLU: x * sigmoid(x). Metal's `precise::exp` is fine
    // here — we already paid for FP32, no need for `fast::exp`.
    float sig = 1.0f / (1.0f + exp(-g));
    out[gid] = bfloat(g * sig * u);
}
