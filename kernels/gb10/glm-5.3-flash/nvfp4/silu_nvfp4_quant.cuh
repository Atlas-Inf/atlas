// SPDX-License-Identifier: AGPL-3.0-only

// DeepSeek-clamped SiLU·mul and token-major NVFP4 group quantization, shared
// by silu_mul_quant_nvfp4 (moe_silu_mul.cu) and the fused gate/up epilogue of
// moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w (moe_w4a16_grouped_gemm.cu)
// so both produce the same bytes. swiglu_limit = 10.0 is the checkpoint's
// config value (DeepSeek-V4 and GLM-5.3 both declare it).

#pragma once
#include <cuda_bf16.h>

// silu(clamp(g)) * clamp(u), rounded through BF16 like the unfused pair.
__device__ __forceinline__ float silu_nvfp4_act(float g, float u) {
    const float SWIGLU_LIMIT = 10.0f;
    g = fminf(g, SWIGLU_LIMIT);
    u = fminf(fmaxf(u, -SWIGLU_LIMIT), SWIGLU_LIMIT);
    const float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    return __bfloat162float(__float2bfloat16(g * sigmoid_g * u));
}

__device__ __forceinline__ unsigned char silu_nvfp4_float_to_e4m3(float v) {
    unsigned int bits = __float_as_uint(v);
    unsigned int sign = (bits >> 31) & 1;
    if ((bits & 0x7FFFFFFF) == 0) return (unsigned char)(sign << 7);

    float absv = fabsf(v);
    if (absv > 448.0f) absv = 448.0f;
    bits = __float_as_uint(absv);
    int f32_exp = (int)((bits >> 23) & 0xFF) - 127;
    unsigned int f32_man = bits & 0x7FFFFF;

    if (f32_exp < -9) {
        return (unsigned char)(sign << 7);
    }
    if (f32_exp < -6) {
        int man = (int)(absv * 512.0f + 0.5f);
        if (man > 7) man = 7;
        if (man < 0) man = 0;
        return (unsigned char)((sign << 7) | man);
    }

    int fp8_exp = f32_exp + 7;
    unsigned int fp8_man;
    if (fp8_exp < 1) fp8_exp = 1;
    if (fp8_exp > 15) {
        fp8_exp = 15;
        fp8_man = 6;
    } else {
        fp8_man = (f32_man + (1 << 19)) >> 20;
        if (fp8_man > 7) {
            fp8_man = 0;
            fp8_exp++;
            if (fp8_exp > 15) {
                fp8_exp = 15;
                fp8_man = 6;
            }
        }
    }
    return (unsigned char)((sign << 7) | (fp8_exp << 3) | fp8_man);
}

__device__ __forceinline__ unsigned int silu_nvfp4_quantize_e2m1(float v) {
    const float absv = fabsf(v);
    const unsigned int sign = (v < 0.0f) ? 8u : 0u;
    unsigned int idx;
    if      (absv <= 0.25f) idx = 0;
    else if (absv <= 0.75f) idx = 1;
    else if (absv <= 1.25f) idx = 2;
    else if (absv <= 1.75f) idx = 3;
    else if (absv <= 2.5f)  idx = 4;
    else if (absv <= 3.5f)  idx = 5;
    else if (absv <= 5.0f)  idx = 6;
    else                    idx = 7;
    return sign | idx;
}

// The E4M3 scale byte of a 16-value group with absolute maximum `group_max`
// and the reciprocal of its decoded value (0 for a zero scale).
__device__ __forceinline__ unsigned char silu_nvfp4_group_scale(float group_max, float* inv) {
    const unsigned char fp8 = silu_nvfp4_float_to_e4m3(group_max / 6.0f);
    const unsigned int exp = (fp8 >> 3) & 0xF;
    const unsigned int man = fp8 & 0x7;
    float decoded;
    if (exp == 0) {
        decoded = (float)man * 0.001953125f;
    } else if (exp == 15 && man == 7) {
        decoded = 0.0f;
    } else {
        decoded = __uint_as_float((exp + 120u) << 23 | (man << 20));
    }
    *inv = decoded > 0.0f ? 1.0f / decoded : 0.0f;
    return fp8;
}
