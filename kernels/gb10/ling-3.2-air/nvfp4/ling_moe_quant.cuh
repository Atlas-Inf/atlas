// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

// Ling-3.2-air decode-path MoE: NVFP4 quantization helpers and the Bailing
// swiglu-limit activation, local to this target (the include closure of every
// other gb10 target stays untouched). The e2m1/e4m3 conversion helpers mirror
// kernels/gb10/glm-5.3-flash/nvfp4/silu_nvfp4_quant.cuh (the GLM fused
// gate/up epilogue's exact bytes) and the activation group quantizer mirrors
// kernels/gb10/common/quantize_bf16_to_nvfp4.cu's nvfp4_quantize_group, so
// the Ling decode path inherits the proven GB10 NVFP4 contracts:
//
//   * A quantization: per 16-value group, e4m3 group scale
//     sf = e4m3(amax * inv_scale2 / 6), nibble = e2m1(x / (sf * scale2)),
//     i.e. the dequant global of the packed activation is `scale2` — the
//     RECIPROCAL of the checkpoint's `input_global_scale` (the compressed-
//     -tensors convention: the scalar quantizes, its reciprocal dequantizes).
//   * GEMM epilogue: the mxf4nvf4 accumulator already carries both e4m3
//     group scales, so scale2_vals[e] = (1/gx) * (1/weight_global_scale[e])
//     folds the reciprocals of both per-tensor scales
//     (cf. GLM fp4l/input_scales — that build's globals are the dequant
//     multipliers themselves, Ling's are their reciprocals).
//   * Down-input re-quantization: silu.mul is quantized with
//     g = input_global_scale (down), so q = x * g / sf and the dequant
//     global of the packed intermediate is 1 / input_global_scale (down).
//
// The Ling swiglu limit is the Bailing clamp (modeling_bailing_moe_v3_vl.py
// BailingMoeV3MLP.forward): routed experts clamp gate to max=limit and up to
// [-limit, limit] on layers whose expert_swiglu_limit_list entry is nonzero;
// the shared expert clamps with share_expert_swiglu_limit_list. A limit of 0
// means no clamp (the reference's `_get_swiglu_limit` maps 0/absent to None).
// The gate activation is silu (configuration_bailing_moe_v3_vl.py
// hidden_act="silu" default; the Ling text config does not override it).

#include <cuda_bf16.h>

#define LMOE_GROUP 16   // NVFP4 scale-group size along K

// ── E2M1 quantization (mirrors silu_nvfp4_quant.cuh) ──────────────────────

__device__ __forceinline__ unsigned int ling_e2m1_quantize(float v) {
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

// f32 -> e4m3 byte (mirrors silu_nvfp4_float_to_e4m3: SM121's hardware cast
// path is avoided on purpose, matching the streaming quantizer's bytes).
__device__ __forceinline__ unsigned char ling_float_to_e4m3(float v) {
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

// The e4m3 scale byte of a 16-value group with absolute maximum `group_max`
// under activation global scale `g` (= input_global_scale for the down
// input; see the file comment), and the multiplier taking a value to its
// e2m1 code. sf = e4m3(amax * g / 6), q = e2m1(x * g / sf) — silu_nvfp4_
// group_scale's arithmetic, so the packed intermediate matches GLM's fused
// epilogue bit for bit. g = 1 with decoded 0 yields q = 0 (the zero group).
__device__ __forceinline__ unsigned char ling_group_scale(float group_max, float* inv, float g) {
    const unsigned char fp8 = ling_float_to_e4m3(group_max * g / 6.0f);
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
    *inv = decoded > 0.0f ? g / decoded : 0.0f;
    return fp8;
}

// ── Bailing swiglu-limit activation ───────────────────────────────────────
//
// The reference chain, per element, on the bf16 weights' dequantized values:
//   gate = silu(gate_proj(x))     [bf16 tensor]
//   up   = up_proj(x)             [bf16 tensor]
//   gate = gate.clamp(max=limit); up = up.clamp(min=-limit, max=limit)
//   out  = down_proj(gate * up)    [bf16 product]
// `limit <= 0` disables both clamps. Each intermediate is rounded through
// BF16 exactly once (torch's opmath elementwise semantics), so the routed
// expert epilogue and the shared expert kernels share these two helpers and
// produce the same bytes.
__device__ __forceinline__ float ling_silu_bf16(float g_bf16, float limit) {
    float g = g_bf16;
    float sigmoid_g = 1.0f / (1.0f + __expf(-g));
    const float silu = __bfloat162float(__float2bfloat16(g * sigmoid_g));
    // silu output is a bf16 tensor; the clamp compares its fp32 value.
    return (limit > 0.0f && silu > limit) ? limit : silu;
}

__device__ __forceinline__ float ling_gate_up_mul(float gate_silu, float up_bf16, float limit) {
    float u = up_bf16;
    if (limit > 0.0f) u = fminf(fmaxf(u, -limit), limit);
    return __bfloat162float(__float2bfloat16(gate_silu * u));
}

// ── Activation group quantizer (mirrors nvfp4_quantize_group) ──────────────
//
// One 16-value group of `row_in` (bf16) into `row_packed`/`row_scale` at
// group `g`, with the dequant global `scale2` (the checkpoint's
// input_global_scale for the projection(s) about to read the packed row).
__device__ __forceinline__ void ling_quantize_group(
    const __nv_bfloat16* __restrict__ row_in,
    unsigned char* __restrict__ row_packed,
    unsigned char* __restrict__ row_scale,
    unsigned int g,
    float scale2,
    float inv_scale2
) {
    const unsigned int base = g * LMOE_GROUP;

    float group_max = 0.0f;
    #pragma unroll
    for (int i = 0; i < LMOE_GROUP; i++) {
        float v = fabsf(__bfloat162float(row_in[base + i]));
        if (v > group_max) group_max = v;
    }

    float fp8_float = (group_max > 0.0f) ? (group_max * inv_scale2 / 6.0f) : 0.0f;
    unsigned char fp8_byte = ling_float_to_e4m3(fp8_float);
    row_scale[g] = fp8_byte;

    const unsigned int fp8_sign = (fp8_byte >> 7) & 1;
    const unsigned int fp8_exp = (fp8_byte >> 3) & 0xF;
    const unsigned int fp8_man = fp8_byte & 0x7;
    float fp8_decoded;
    if (fp8_exp == 0) {
        fp8_decoded = (float)fp8_man * 0.001953125f;
    } else if (fp8_exp == 15 && fp8_man == 7) {
        fp8_decoded = 0.0f;
    } else {
        fp8_decoded = __uint_as_float(((fp8_exp + 120u) << 23) | (fp8_man << 20));
    }
    if (fp8_sign) fp8_decoded = -fp8_decoded;

    const float effective_scale = fp8_decoded * scale2;
    const float inv_eff = (effective_scale > 0.0f) ? (1.0f / effective_scale) : 0.0f;

    #pragma unroll
    for (int i = 0; i < LMOE_GROUP; i += 2) {
        float v0 = __bfloat162float(row_in[base + i]) * inv_eff;
        float v1 = __bfloat162float(row_in[base + i + 1]) * inv_eff;
        const unsigned int n0 = ling_e2m1_quantize(v0);
        const unsigned int n1 = ling_e2m1_quantize(v1);
        row_packed[g * 8 + i / 2] = (unsigned char)((n1 << 4) | (n0 & 0xF));
    }
}
