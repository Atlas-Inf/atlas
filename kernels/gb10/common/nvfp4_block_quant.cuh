// SPDX-License-Identifier: AGPL-3.0-only
//
// NVFP4 block quantization of activations, bit for bit the quantizer of the
// engine NVIDIA's ModelOpt FP4 cards name (vLLM b12x: flashinfer b12x
// `_lib/intrinsics.py` `quantize_block_fp4` + `quantize_and_pack_16`, the
// non-fast path both its A4 prefill quantizers take):
//
//   sf   = e4m3(min(amax * g / 6, 448))       cvt.rn.satfinite, g the global
//                                               scale (1 / input_scale)
//   vs   = decode(sf) / g                      0 when decode(sf) or g is 0
//   code = e2m1 of x on the grid vs * {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to
//          the even code (the thresholds vs * {0.25, 0.75, ...} below, each
//          a rounded product, compared against |x|), sign only on a nonzero
//          magnitude; every code 0 when vs is 0.
//
// The thresholds are compared against |x| itself -- not x * g / sf rounded
// to E2M1 -- which is the reference's arithmetic, and on a tie (|x| equal to
// a rounded threshold) it decides differently from that product. The
// divisions and products are the IEEE round-to-nearest forms whatever the
// translation unit's fast-math flags.

#pragma once
#include <cuda_fp16.h>

// float -> E4M3 byte, round to nearest even, saturating (cvt.rn.satfinite).
__device__ __forceinline__ unsigned int nvfp4q_e4m3(float v) {
    unsigned short pair;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(pair) : "f"(0.0f), "f"(v));
    return pair & 0xFFu;
}

// E4M3 byte -> float, exact (subnormals and signed zero included).
__device__ __forceinline__ float nvfp4q_e4m3_value(unsigned int b) {
    unsigned int pair;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(pair) : "h"((unsigned short)(b & 0xFFu)));
    return __half2float(__ushort_as_half((unsigned short)(pair & 0xFFFFu)));
}

// The E2M1 decision grid of one block: |x| against vs * {0.25, 0.75, 1.25,
// 1.75, 2.5, 3.5, 5}.
struct Nvfp4qGrid {
    float t025, t075, t125, t175, t250, t350, t500;
    bool zero;  // vs == 0: every code 0
};

// The scale byte of a block with absolute maximum `amax` under global scale
// `g`, and its decision grid.
__device__ __forceinline__ unsigned int nvfp4q_block_scale(float amax, float g, Nvfp4qGrid* grid) {
    const float sf = fminf(__fdiv_rn(__fmul_rn(amax, g), 6.0f), 448.0f);
    const unsigned int byte = nvfp4q_e4m3(sf);
    const float dec = nvfp4q_e4m3_value(byte);
    grid->zero = dec == 0.0f || g == 0.0f;
    const float vs = grid->zero ? 0.0f : __fdiv_rn(dec, g);
    grid->t025 = __fmul_rn(vs, 0.25f);
    grid->t075 = __fmul_rn(vs, 0.75f);
    grid->t125 = __fmul_rn(vs, 1.25f);
    grid->t175 = __fmul_rn(vs, 1.75f);
    grid->t250 = __fmul_rn(vs, 2.5f);
    grid->t350 = __fmul_rn(vs, 3.5f);
    grid->t500 = __fmul_rn(vs, 5.0f);
    return byte;
}

// The E2M1 code (sign in bit 3) of `x` on `grid`.
__device__ __forceinline__ unsigned int nvfp4q_code(float x, const Nvfp4qGrid& grid) {
    if (grid.zero) return 0u;
    const float m = fabsf(x);
    unsigned int c = 0u;
    if (m > grid.t025 && m < grid.t075) c = 1u;
    else if (m >= grid.t075 && m <= grid.t125) c = 2u;
    else if (m > grid.t125 && m < grid.t175) c = 3u;
    else if (m >= grid.t175 && m <= grid.t250) c = 4u;
    else if (m > grid.t250 && m < grid.t350) c = 5u;
    else if (m >= grid.t350 && m <= grid.t500) c = 6u;
    else if (m > grid.t500) c = 7u;
    return (c != 0u && x < 0.0f) ? (c | 8u) : c;
}

// Two values -> one packed byte (`lo` in the low nibble).
__device__ __forceinline__ unsigned int nvfp4q_pack2(float lo, float hi, const Nvfp4qGrid& grid) {
    return nvfp4q_code(lo, grid) | (nvfp4q_code(hi, grid) << 4);
}
