// SPDX-License-Identifier: AGPL-3.0-only
// Vendored from vcruz305/exllamav3 @ 047ce72 (fork of turboderp-org/exllamav3, MIT,
// Copyright (c) 2025 Turboderp); see ../NOTICE.md.
// Source: exllamav3_ext/quant/exl3_gemv_kernel.cuh lines 32-155 (namespace exl3_gemv_ns: mma_ab_h,
// the dp4a mul1 pair decode and the register window extractors), verbatim. The cooperative GEMV
// kernel itself (exl3_gemv_kernel) is not vendored; only these helpers are used by gemv_tile.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>

#include "../codebook.cuh"
#include "../exl3_dq.cuh"
#include "../half_uint16.cuh"
#include "../ptx_bits.cuh"
#include "../ptx_frag.cuh"

namespace exl3_gemv_ns {

// mma.m16n8k16 with the A operand supplied as two FragB halves, fp16 accumulate
__device__ __forceinline__ void mma_ab_h(const FragB& a01, const FragB& a23, const FragB& b, FragC_h& c)
{
    const uint32_t* a0 = reinterpret_cast<const uint32_t*>(&a01);
    const uint32_t* a1 = reinterpret_cast<const uint32_t*>(&a23);
    const uint32_t* bb = reinterpret_cast<const uint32_t*>(&b);
    uint32_t* cc = reinterpret_cast<uint32_t*>(&c);
    asm
    (
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
        "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
        : "+r"(cc[0]), "+r"(cc[1])
        :  "r"(a0[0]), "r"(a0[1]), "r"(a1[0]), "r"(a1[1]),
           "r"(bb[0]), "r"(bb[1])
    );
}

// mul1 codebook pair decode via dp4a byte sum (bit-identical to the vabsdiff4 form)
__device__ __forceinline__ half2 decode_pair_cb2_dp4a_(uint32_t x0, uint32_t x1)
{
    x0 *= 0x83DCD12Du;
    x1 *= 0x83DCD12Du;
    uint32_t sum0 = __dp4a(x0, 0x01010101u, 0x6400u);
    uint32_t sum1 = __dp4a(x1, 0x01010101u, 0x6400u);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half_uint16 h0((uint16_t) sum0);
    half_uint16 h1((uint16_t) sum1);
    return __hfma2(__halves2half2(h0.as_half, h1.as_half), k_inv_h2, k_bias_h2);
}

template <int cb>
__device__ __forceinline__ void decode8(uint32_t w0, uint32_t w1, uint32_t w2, uint32_t w3,
    uint32_t w4, uint32_t w5, uint32_t w6, uint32_t w7, FragB& f0, FragB& f1)
{
    if constexpr (cb == 2)
    {
        f0[0] = decode_pair_cb2_dp4a_(w0, w1);
        f0[1] = decode_pair_cb2_dp4a_(w2, w3);
        f1[0] = decode_pair_cb2_dp4a_(w4, w5);
        f1[1] = decode_pair_cb2_dp4a_(w6, w7);
    }
    else
    {
        f0[0] = decode_3inst_2<cb>(w0, w1);
        f0[1] = decode_3inst_2<cb>(w2, w3);
        f1[0] = decode_3inst_2<cb>(w4, w5);
        f1[1] = decode_3inst_2<cb>(w6, w7);
    }
}

// Window extraction from two already-loaded words, same order as dq8_aligned_4bits
template <int cb>
__device__ __forceinline__ void dq8_regs_4bits(uint32_t a, uint32_t b, FragB& f0, FragB& f1)
{
    uint32_t s, w0, w1, w2, w3, w4, w5, w6, w7;
    FSHF_IMM(s, b, a, 20);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 4);
    BFE16_IMM(w5, b, 8);
    BFE16_IMM(w4, b, 12);
    BFE16_IMM(w3, b, 16);
    w2 = s & 0xffff;
    BFE16_IMM(w1, s, 4);
    BFE16_IMM(w0, s, 8);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8_aligned_2bits: the two words and the funnel shift are lane-dependent
template <int cb>
__device__ __forceinline__ void dq8_regs_2bits(uint32_t a, uint32_t b, int t_offset, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    b = fshift(b, a, ((~t_offset) & 8) << 1);
    w7 = b & 0xffff;
    BFE16_IMM(w6, b, 2);
    BFE16_IMM(w5, b, 4);
    BFE16_IMM(w4, b, 6);
    BFE16_IMM(w3, b, 8);
    BFE16_IMM(w2, b, 10);
    BFE16_IMM(w1, b, 12);
    BFE16_IMM(w0, b, 14);
    decode8<cb>(w0, w1, w2, w3, w4, w5, w6, w7, f0, f1);
}

// Register form of dq8<3, cb, 4> with the per-lane funnel alignment precomputed
template <int cb>
__device__ __forceinline__ void dq8_regs_3bits(uint32_t a, uint32_t b, int s2, FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b, a, s2);
    w6 = w7 >> 3;
    w5 = w6 >> 3;
    w4 = w5 >> 3;
    w3 = fshift(b, a, s2 + 12);
    w2 = w3 >> 3;
    w1 = w2 >> 3;
    w0 = w1 >> 3;
    decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

// Register form of dq8_half<KA, cb> (exl3_dq.cuh): the two window groups' word pairs and funnel alignments are
// lane constants, so the caller resolves the four words by shuffle or from the staged tile
template <int KA, int cb>
__device__ __forceinline__ void dq8_regs_half(uint32_t a7, uint32_t b7, int s7, uint32_t a3, uint32_t b3, int s3,
                                              FragB& f0, FragB& f1)
{
    uint32_t w0, w1, w2, w3, w4, w5, w6, w7;
    w7 = fshift(b7, a7, s7);
    w6 = w7 >> (KA + 1);
    w5 = w6 >> KA;
    w4 = w5 >> (KA + 1);
    w3 = fshift(b3, a3, s3);
    w2 = w3 >> (KA + 1);
    w1 = w2 >> KA;
    w0 = w1 >> (KA + 1);
    decode8<cb>(w0 & 0xffff, w1 & 0xffff, w2 & 0xffff, w3 & 0xffff,
                w4 & 0xffff, w5 & 0xffff, w6 & 0xffff, w7 & 0xffff, f0, f1);
}

}  // namespace exl3_gemv_ns
