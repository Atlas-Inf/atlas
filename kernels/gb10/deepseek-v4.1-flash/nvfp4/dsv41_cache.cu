// SPDX-License-Identifier: AGPL-3.0-only
//
// What DeepSeek-V4.1-Flash writes into its attention caches: rotary
// positions on the last values of a vector, and the reference's FP8 and FP4
// quantize-then-dequantize, which leave BF16 values that are exactly what
// the quantized form would decode to. The host statement of each rule, which
// these kernels are tested against byte for byte, is
// `spark-model/src/layers/deepseek_v41/cache_recipe.rs`.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

// Rotate the last `rope` values of every vector to its row's position.
// x [rows, vectors, dim] BF16, positions [rows], freqs [rope / 2] radians per
// position. `inverse` turns the other way. One thread per value pair.
extern "C" __global__ void dsv41_rotate(
    __nv_bfloat16* __restrict__ x,
    const unsigned int* __restrict__ positions,
    const float* __restrict__ freqs,
    unsigned int rows,
    unsigned int vectors,
    unsigned int dim,
    unsigned int rope,
    unsigned int inverse) {
    const unsigned int pairs = rope / 2;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * vectors * pairs) return;
    const unsigned int pair = (unsigned int)(i % pairs);
    const size_t vector = i / pairs;
    const float angle = (float)positions[vector / vectors] * freqs[pair];
    const float c = cosf(angle), s = inverse ? -sinf(angle) : sinf(angle);
    __nv_bfloat16* p = x + vector * dim + (dim - rope) + 2 * pair;
    const float re = __bfloat162float(p[0]), im = __bfloat162float(p[1]);
    p[0] = __float2bfloat16(re * c - im * s);
    p[1] = __float2bfloat16(re * s + im * c);
}

// FP8 (E4M3) under one power-of-two scale per 32 values, written as BF16:
// `blocks` blocks of 32 from `src` into `dst`, which may be the same buffer.
// One thread per block.
extern "C" __global__ void dsv41_fp8_quant_dequant(
    const __nv_bfloat16* src, __nv_bfloat16* dst, unsigned int blocks) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= blocks) return;
    const __nv_bfloat16* p = src + b * 32;
    float v[32];
    for (int i = 0; i < 32; ++i) v[i] = __bfloat162float(p[i]);
    dsv41_fp8_block(v, dst + b * 32);
}

// FP4 (E2M1) written back as BF16, one thread per group. `e4m3_scale` = 0:
// groups of 32 under a power-of-two scale (index keys). 1: groups of 16 under
// an E4M3 scale (compressed keys).
extern "C" __global__ void dsv41_fp4_quant_dequant(
    __nv_bfloat16* __restrict__ x, unsigned int groups, unsigned int e4m3_scale) {
    const size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= groups) return;
    const int width = e4m3_scale ? 16 : 32;
    __nv_bfloat16* p = x + g * width;
    float amax = 0.0f;
    for (int i = 0; i < width; ++i) amax = fmaxf(amax, fabsf(__bfloat162float(p[i])));
    const float scale = e4m3_scale
        ? dsv41_round_e4m3(fmaxf(amax, 6.0f * 0x1p-9f) / 6.0f)
        : dsv41_pow2_ceil(fmaxf(amax, 6.0f * 0x1p-126f) * (1.0f / 6.0f));
    for (int i = 0; i < width; ++i) {
        const float scaled = fminf(fmaxf(__bfloat162float(p[i]) / scale, -6.0f), 6.0f);
        p[i] = __float2bfloat16(dsv41_round_fp4(scaled) * scale);
    }
}
