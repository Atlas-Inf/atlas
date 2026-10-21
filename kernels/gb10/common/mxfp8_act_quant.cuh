// SPDX-License-Identifier: AGPL-3.0-only
// Ported from Enntity/atlas (branch `sparkqwen` @ 92dd543, AGPL-3.0-only),
// `kernels/gb10/common/mxfp8_act_quant.cuh`, with Atlas-Inf/atlas#198's integration changes.
//
// MXFP8 activation quantization for the cuBLASLt W8A8 prefill GEMM
// (`spark_runtime::cublaslt::fp8_gemm_act_weight_t_mx`): device functions a
// producer kernel calls to emit E4M3 values plus UE8M0 scales already in
// cuBLASLt's tiled scale layout, instead of BF16 that a separate pass would
// quantize. `mxfp8_act_quant.cu` is the standalone pass built on the same
// functions, so a fused producer and "producer then quant" give the same bytes.
//
// Rounding: ModelOpt's NV E8M0 rule, the one the FP4-Local checkpoint was
// produced with (NVIDIA/Model-Optimizer @ c7966119,
// modelopt/torch/kernels/quantization/gemm/tensor_quant_mx.cu,
// `compute_scale` -> `compute_scale_e8m0_NV`, Apache-2.0), per block of 32:
//   amax  = max |x| over the block, each |x| clamped to FLT_MAX (so an inf
//           counts as FLT_MAX) and NaNs skipped (fmaxf);
//   amax == 0 / inf / NaN -> e = 0 (E8M0 127);
//   ratio = amax / 448 in FP32, round-to-nearest-even;
//   e     = ceil(log2(ratio)) by an exponent-field test on `ratio` (it may be
//           the floor when the RNE division lands exactly on a power of two;
//           ModelOpt documents that and keeps it);
//   q     = E4M3(x * 2^-e), round-to-nearest-even, saturating to +-448
//           (`static_cast<__nv_fp8_e4m3>`, i.e. __NV_SATFINITE).
// ModelOpt writes the fake-quantized float sign * E4M3(|x| * 2^-e) * 2^e; the
// bytes here decode to exactly that value (+-0 both decode to 0).
//
// Scale layout (cuBLAS docs, "1D block scaling factors layout", the layout
// CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0 reads): for an [outer, K] operand
// with inner = K/32 scale columns padded to inner_pad = round_up(inner, 4),
// 128 x 4 tiles of 512 bytes, tiles row-major over (outer / 128,
// inner_pad / 4); inside a tile
//   (outer % 32) * 16 + ((outer % 128) / 32) * 4 + inner % 4.
// Entries past `inner` must be finite (written 0 here); entries of rows past
// `outer` only feed output rows that are never stored.

#pragma once

#include <cuda_fp8.h>

// E4M3's largest finite magnitude.
#define MX_E4M3_MAX 448.0f

// Byte offset of scale (outer, inner) in the tiled layout.
__host__ __device__ __forceinline__ unsigned long long mx_sf_offset(
    unsigned int outer, unsigned int inner, unsigned int inner_pad) {
    const unsigned long long tile =
        (unsigned long long)(outer / 128u) * (inner_pad / 4u) + inner / 4u;
    return tile * 512ull + (outer % 32u) * 16u + ((outer % 128u) / 32u) * 4u + (inner % 4u);
}

// |x| as ModelOpt's amax reduction sees it (inf -> FLT_MAX, NaN -> skipped
// by the caller's fmaxf).
__device__ __forceinline__ float mx_abs_clamped(float x) {
    return fminf(fabsf(x), 3.40282347e+38f);
}

// The block's unscale exponent e (scale 2^e) from its amax.
__device__ __forceinline__ int mx_e8m0_exponent(float amax) {
    if (amax == 0.0f || isinf(amax) || isnan(amax)) return 0;
    const unsigned int r = __float_as_uint(__fdiv_rn(amax, MX_E4M3_MAX));
    const unsigned int ef = (r >> 23) & 0xffu;
    const unsigned int sig = r & 0x7fffffu;
    const bool up = sig > 0u && ef != 0xfeu && !(ef == 0u && sig <= 0x400000u);
    return up ? (int)ef - 126 : (int)ef - 127;
}

// E4M3 byte of x * 2^-e (RNE, satfinite).
__device__ __forceinline__ unsigned char mx_e4m3(float x, float inv_scale) {
    return (unsigned char)__nv_cvt_float_to_fp8(x * inv_scale, __NV_SATFINITE, __NV_E4M3);
}

// 2^-e, exact over E8M0's whole range (e in [-127, 127]).
__device__ __forceinline__ float mx_inv_scale(int e) {
    return ldexpf(1.0f, -e);
}

// The exponent of a 32-value block spread over 4 consecutive lanes (8 values
// each, any positions): `local_amax` is this lane's max of
// `mx_abs_clamped`, combined across the quad by two xor-shuffles. Every lane
// of the warp must call it (full-mask shuffles).
__device__ __forceinline__ int mx_quad_exponent(float local_amax) {
    float amax = fmaxf(local_amax, __shfl_xor_sync(0xffffffffu, local_amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    return mx_e8m0_exponent(amax);
}

// Quantize 8 consecutive values of one 32-value block held by 4 consecutive
// lanes (lane % 4 = which 8): this lane's 8 E4M3 bytes packed little-endian
// into a uint2. Every lane of the warp must call it; returns the E8M0 byte.
__device__ __forceinline__ unsigned char mx_quant8_quad(const float v[8], uint2& packed) {
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, mx_abs_clamped(v[j]));
    const int e = mx_quad_exponent(amax);
    const float inv = mx_inv_scale(e);
    unsigned int w[2] = {0u, 0u};
    #pragma unroll
    for (int j = 0; j < 8; ++j) w[j / 4] |= (unsigned int)mx_e4m3(v[j], inv) << (8 * (j % 4));
    packed = make_uint2(w[0], w[1]);
    return (unsigned char)(e + 127);
}

// Zero the padding scale columns [inner, inner_pad) of row `outer`.
__device__ __forceinline__ void mx_sf_zero_pad(
    unsigned char* sf, unsigned int outer, unsigned int inner, unsigned int inner_pad) {
    for (unsigned int c = inner; c < inner_pad; ++c) sf[mx_sf_offset(outer, c, inner_pad)] = 0;
}
