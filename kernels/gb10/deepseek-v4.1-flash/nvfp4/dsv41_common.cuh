// SPDX-License-Identifier: AGPL-3.0-only
//
// Arithmetic that more than one DeepSeek-V4.1-Flash kernel performs, stated
// once: a kernel that fuses several steps into one launch calls the same
// lines as the plain kernel of each step, so their bytes cannot differ.

#pragma once

#include <cuda_bf16.h>

#define DSV41_PARTS 4

__device__ __forceinline__ float dsv41_bf16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

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

// FP8 (E4M3) under one power-of-two scale per 32 values: the scale of a
// block whose largest magnitude is `amax`, and a value under it, as the
// number its FP8 form decodes to (exact in BF16).
__device__ __forceinline__ float dsv41_fp8_scale(float amax) {
    return dsv41_pow2_ceil(fmaxf(amax, 1e-4f) * (1.0f / 448.0f));
}

__device__ __forceinline__ float dsv41_fp8_value(float v, float scale) {
    const float scaled = fminf(fmaxf(v / scale, -448.0f), 448.0f);
    return dsv41_round_e4m3(scaled) * scale;
}

// The 32 values of `v`, written to `dst` as that.
__device__ __forceinline__ void dsv41_fp8_block(const float* v, __nv_bfloat16* dst) {
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const float scale = dsv41_fp8_scale(amax);
    for (int i = 0; i < 32; ++i) dst[i] = __float2bfloat16(dsv41_fp8_value(v[i], scale));
}

// Two BF16 values in a word.
__device__ __forceinline__ void dsv41_unpack2(unsigned int packed, float& v0, float& v1) {
    v0 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed & 0xFFFFu)));
    v1 = __bfloat162float(__ushort_as_bfloat16((unsigned short)(packed >> 16)));
}

__device__ __forceinline__ unsigned int dsv41_pack2(float v0, float v1) {
    return (unsigned int)__bfloat16_as_ushort(__float2bfloat16(v0)) |
           ((unsigned int)__bfloat16_as_ushort(__float2bfloat16(v1)) << 16);
}

// One rotary pair turned by `angle` radians, or back, rounded to BF16.
__device__ __forceinline__ void dsv41_rotate_pair(
    float re, float im, float angle, bool inverse, float& out_re, float& out_im) {
    const float c = cosf(angle), s = inverse ? -sinf(angle) : sinf(angle);
    out_re = dsv41_bf16(re * c - im * s);
    out_im = dsv41_bf16(re * s + im * c);
}

// Value `i` of a projection computed in four parts (`dsv41_parts.cu`):
// BF16(((p0 + p1) + p2) + p3), parts [4, stride] FP32.
__device__ __forceinline__ float dsv41_parts_sum(const float* parts, size_t stride, size_t i) {
    float sum = parts[i];
    for (unsigned int part = 1; part < DSV41_PARTS; ++part) sum += parts[part * stride + i];
    return dsv41_bf16(sum);
}

// Value `d` of a sublayer's output from its parts [4, slots, n]: every slot
// rounded as above, the slots added in FP32 in slot order, the sum rounded.
// One slot is the attention's output; the MoE has a slot per expert.
__device__ __forceinline__ float dsv41_parts_combine(
    const float* parts, unsigned int slots, unsigned int n, unsigned int d) {
    const size_t stride = (size_t)slots * n;
    float total = 0.0f;
    for (unsigned int slot = 0; slot < slots; ++slot)
        total += dsv41_parts_sum(parts, stride, (size_t)slot * n + d);
    return dsv41_bf16(total);
}

// weight * silu(min(gate, limit)) * clamp(up, -limit, limit), rounded to BF16.
__device__ __forceinline__ float dsv41_swiglu_one(float gate, float up, float weight, float limit) {
    const float g = fminf(gate, limit);
    const float u = fminf(fmaxf(up, -limit), limit);
    const float silu = g * (1.0f / (1.0f + expf(-g)));
    return dsv41_bf16(weight * (silu * u));
}

#define DSV41_DOT_THREADS 256

// x . row over k BF16 values in FP32, by one block of 256 threads: a thread
// adds every 256th product, and the threads' sums meet in a tree. Every
// thread returns the sum.
__device__ __forceinline__ float dsv41_dot_wide(
    const __nv_bfloat16* x, const __nv_bfloat16* row, unsigned int k) {
    __shared__ float red[DSV41_DOT_THREADS];
    const unsigned int tid = threadIdx.x;
    float acc = 0.0f;
    for (unsigned int d = tid; d < k; d += DSV41_DOT_THREADS)
        acc += __bfloat162float(x[d]) * __bfloat162float(row[d]);
    red[tid] = acc;
    __syncthreads();
    for (unsigned int width = DSV41_DOT_THREADS / 2; width > 0; width >>= 1) {
        if (tid < width) red[tid] += red[tid + width];
        __syncthreads();
    }
    return red[0];
}
