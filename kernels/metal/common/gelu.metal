// SPDX-License-Identifier: AGPL-3.0-only
//
// GeLU activation, tanh approximation:
//
//   gelu(x) = 0.5 * x * (1 + tanh( sqrt(2/π) * (x + 0.044715 * x³) ))
//
// Used by ViT-style transformers in their MLP blocks (the
// SwiGLU-using LLM trunk uses `silu_gate` instead). The tanh
// approximation matches the reference numerics every modern
// implementation ships (HuggingFace, PyTorch, MLX) without the
// exotic numerical tradeoffs of the exact-erf formulation.
//
// Layout:
//   x, out : bfloat [n]    (in-place safe — set out == x)

#include <metal_stdlib>
using namespace metal;

constant float SQRT_2_OVER_PI = 0.7978845608028654f;
constant float GELU_C        = 0.044715f;

kernel void gelu(
    constant uint &n         [[buffer(0)]],
    device const bfloat *x   [[buffer(1)]],
    device bfloat       *out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float v = float(x[gid]);
    float v3 = v * v * v;
    float arg = SQRT_2_OVER_PI * (v + GELU_C * v3);
    float t = tanh(arg);
    out[gid] = bfloat(0.5f * v * (1.0f + t));
}
