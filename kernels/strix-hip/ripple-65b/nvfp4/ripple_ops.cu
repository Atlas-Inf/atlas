// SPDX-License-Identifier: AGPL-3.0-only

// Ripple-65B (`model_type = "ricursive"`) element-wise ops.
//
// ripple_silu_mul_clamp: the clamped SwiGLU of RicursiveMLP —
//   gate = min(gate, C); up = clamp(up, -C, C); out = silu(gate) * up
// with C = ffn_clamp_threshold = 20.0 (config.json). Same signature as
// `moe_silu_mul`, so DenseFfnLayer swaps it in for its `act_mul` handle and
// every split-SiLU site picks it up. The constant is compile-time, like the
// DeepSeek/Step shadows of `moe_silu_mul`; the loader refuses a checkpoint
// whose ffn_clamp_threshold differs.
//
// ripple_valres_lerp: the value-residual input of every attention layer —
//   out = torch.lerp(vres, h, blend)   (BF16 in/out, FP32 opmath)
// where vres = scaled layer-0 embeddings, h = input_layernorm(x), and
// blend = bf16(sigmoid(value_res_lambda)). Matches ATen's lerp formula
// (weight < 0.5 ? a + w (b - a) : b - (b - a)(1 - w)).

#include <cuda_bf16.h>

#define RIPPLE_FFN_CLAMP 20.0f

extern "C" __global__ void ripple_silu_mul_clamp(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output,
    unsigned int total_elements
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_elements) return;
    float g = fminf(__bfloat162float(gate[idx]), RIPPLE_FFN_CLAMP);
    float u = fminf(fmaxf(__bfloat162float(up[idx]), -RIPPLE_FFN_CLAMP), RIPPLE_FFN_CLAMP);
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    output[idx] = __float2bfloat16(g * sigmoid_g * u);
}

// Grid: (ceil(rows * dim / 256), 1, 1)  Block: (256, 1, 1).
// vres rows are indexed through `vres_stride` (elements) so a decode batch can
// read its rows out of a wider per-slot buffer; h/out are packed [rows, dim].
extern "C" __global__ void ripple_valres_lerp(
    const __nv_bfloat16* __restrict__ vres,
    const __nv_bfloat16* __restrict__ h,
    __nv_bfloat16* __restrict__ out,
    unsigned int rows,
    unsigned int dim,
    unsigned int vres_stride,
    float blend
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * dim) return;
    unsigned int r = idx / dim, c = idx - r * dim;
    float a = __bfloat162float(vres[(unsigned long long)r * vres_stride + c]);
    float b = __bfloat162float(h[idx]);
    float d = b - a;
    float v = (blend < 0.5f) ? a + blend * d : b - d * (1.0f - blend);
    out[idx] = __float2bfloat16(v);
}
