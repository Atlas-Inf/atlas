// SPDX-License-Identifier: AGPL-3.0-only
//
// What DeepSeek-V4.1-Flash writes into its attention caches: rotary
// positions on the last values of a vector, and the reference's FP8 and FP4
// quantize-then-dequantize, which leave BF16 values that are exactly what
// the quantized form would decode to. The host statement of each rule, which
// these kernels are tested against byte for byte, is
// `spark-model/src/layers/deepseek_v41/cache_recipe.rs`.

#include <cuda_bf16.h>

// 2^ceil(log2(ratio)), read off the float's exponent.
__device__ __forceinline__ float dsv41_pow2_ceil(float ratio) {
    const unsigned int bits = __float_as_uint(ratio);
    const int exponent = (int)((bits >> 23) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u);
    return __uint_as_float((unsigned int)(exponent + 127) << 23);
}

__device__ __forceinline__ int dsv41_exponent(float magnitude) {
    return (int)((__float_as_uint(magnitude) >> 23) & 0xffu) - 127;
}

// Nearest E4M3 value: 3 mantissa bits, least exponent -6, at most 448.
__device__ __forceinline__ float dsv41_round_e4m3(float v) {
    const float mag = fabsf(v);
    const float step = ldexpf(1.0f, max(dsv41_exponent(mag), -6) - 3);
    return copysignf(fminf(rintf(mag / step) * step, 448.0f), v);
}

// Nearest E2M1 value: 0, 0.5, 1, 1.5, 2, 3, 4 or 6.
__device__ __forceinline__ float dsv41_round_fp4(float v) {
    const float mag = fabsf(v);
    const float step = ldexpf(1.0f, min(max(dsv41_exponent(mag), 0), 2) - 1);
    return copysignf(rintf(mag / step) * step, v);
}

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
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        v[i] = __bfloat162float(p[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float scale = dsv41_pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
    for (int i = 0; i < 32; ++i) {
        const float scaled = fminf(fmaxf(v[i] / scale, -448.0f), 448.0f);
        dst[b * 32 + i] = __float2bfloat16(dsv41_round_e4m3(scaled) * scale);
    }
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
