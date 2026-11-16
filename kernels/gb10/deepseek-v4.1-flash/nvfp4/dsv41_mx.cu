// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: a batch of activations as the block-scaled
// tensor-core GEMM reads them (`spark_runtime::cublaslt::
// fp8_gemm_act_weight_t_mx`): E4M3 codes and one UE8M0 scale per 32 values,
// the scales in cuBLASLt's tiled layout (`mxfp8_act_quant.cuh`).
//
// The rounding is this model's (`dsv41_fp8_quant_dequant`): a block's scale
// is the power of two at or above max(amax, 1e-4) / 448, and a value is the
// nearest E4M3 value under it. So the codes times their scales are the
// values the row-at-a-time projections multiply, and a GEMM over them
// differs from those only in the order of its FP32 sum.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "../../common/mxfp8_act_quant.cuh"
#include "dsv41_common.cuh"

// q[m, k] = the code of x[m, k], sf the scales. Row m of x starts `a_stride`
// values after row m - 1; an input the ranks gathered is in pieces of `piece`
// values a row, the next piece `piece_stride` values on (`f4_at` in
// `dsv41_tc.cuh`), and one in a single run has `piece` 0. q is [M, K]
// bytes; K % 32 == 0. One thread a block of 32 values.
//
// For the experts' own kernel (`dsv41_mxg.cu`) `inner_pad` is 0 and the
// scales are as they come, [M, K / 32].
extern "C" __global__ void dsv41_mx_quant(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf,
    unsigned int M,
    unsigned int K,
    unsigned int a_stride,
    unsigned int piece,
    unsigned long long piece_stride,
    unsigned int inner_pad) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int blocks = K / 32u;
    if (t >= (size_t)M * blocks) return;
    const unsigned int m = (unsigned int)(t / blocks), b = (unsigned int)(t % blocks);
    const unsigned int kb = b * 32u;
    const __nv_bfloat16* src = x + (size_t)m * a_stride +
                               (piece ? (size_t)(kb / piece) * piece_stride + kb % piece : kb);
    // A block's 64 bytes in four loads and its 32 codes out in two stores:
    // a value or a code at a time, every instruction of a warp went to 16
    // lines of memory for two bytes of each.
    float v[32];
    if ((((size_t)src) & 15u) == 0u) {
        const uint4* in = (const uint4*)src;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint4 w = in[j];
            dsv41_unpack2(w.x, v[8 * j], v[8 * j + 1]);
            dsv41_unpack2(w.y, v[8 * j + 2], v[8 * j + 3]);
            dsv41_unpack2(w.z, v[8 * j + 4], v[8 * j + 5]);
            dsv41_unpack2(w.w, v[8 * j + 6], v[8 * j + 7]);
        }
    } else {
        for (int i = 0; i < 32; ++i) v[i] = __bfloat162float(src[i]);
    }
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const float scale = dsv41_fp8_scale(amax);
    unsigned int codes[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
    for (int i = 0; i < 32; ++i) {
        const float scaled = fminf(fmaxf(v[i] / scale, -448.0f), 448.0f);
        const unsigned int code = (unsigned char)__nv_cvt_float_to_fp8(
            dsv41_round_e4m3(scaled), __NV_SATFINITE, __NV_E4M3);
        codes[i / 4] |= code << (8 * (i % 4));
    }
    // q is whole blocks of 32 from an allocation's start.
    uint4* dst = (uint4*)(q + (size_t)m * K + kb);
    dst[0] = make_uint4(codes[0], codes[1], codes[2], codes[3]);
    dst[1] = make_uint4(codes[4], codes[5], codes[6], codes[7]);
    const unsigned char code = (unsigned char)(dsv41_exponent(scale) + 127);
    if (inner_pad == 0u) {
        sf[t] = code;
        return;
    }
    sf[mx_sf_offset(m, b, inner_pad)] = code;
    if (b + 1u == blocks) mx_sf_zero_pad(sf, m, blocks, inner_pad);
}

// Codes the ranks gathered as one GEMM operand. Each rank rounds its own
// part of a batch (this file's `dsv41_mx_quant`, scales as they come) and the
// ranks gather the codes and scales, half the bytes of the values; this lays
// them out as the GEMM reads them: q [M, K] a row's K codes together, and sf
// tiled as `dsv41_mx_quant` tiles it.
//
// A part is `region` bytes after the last: its `groups` pieces' codes, each
// [M, piece], then their scales, each [M, piece / 32]. K = parts * groups *
// piece. One thread a block of 32 codes.
extern "C" __global__ void dsv41_mx_gathered(
    const unsigned char* __restrict__ parts,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf,
    unsigned int M,
    unsigned int K,
    unsigned int piece,
    unsigned int groups,
    unsigned long long region,
    unsigned int inner_pad) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int blocks = K / 32u;
    if (t >= (size_t)M * blocks) return;
    const unsigned int m = (unsigned int)(t / blocks), b = (unsigned int)(t % blocks);
    const unsigned int kb = b * 32u, of = kb / piece, in = kb % piece;
    const unsigned char* part = parts + (size_t)(of / groups) * region;
    const size_t group = of % groups;
    const uint4* from = (const uint4*)(part + (group * M + m) * piece + in);
    uint4* to = (uint4*)(q + (size_t)m * K + kb);
    to[0] = from[0];
    to[1] = from[1];
    const unsigned char* scales = part + (size_t)groups * M * piece;
    sf[mx_sf_offset(m, b, inner_pad)] = scales[(group * M + m) * (piece / 32u) + in / 32u];
    if (b + 1u == blocks) mx_sf_zero_pad(sf, m, blocks, inner_pad);
}

// Scales the ranks gathered, a part each, as one run a row: out[m, part *
// blocks + b] = sf[part * piece_stride + m * blocks + b], where a part's
// `blocks` scales of a row are `piece_stride` bytes after the part's before.
// One thread a scale of the output.
extern "C" __global__ void dsv41_mx_scales_rows(
    const unsigned char* __restrict__ sf,
    unsigned char* __restrict__ out,
    unsigned int M,
    unsigned int blocks,
    unsigned int parts,
    unsigned long long piece_stride) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per = blocks * parts;
    if (t >= (size_t)M * per) return;
    const size_t m = t / per;
    const unsigned int at = (unsigned int)(t % per);
    out[t] = sf[(size_t)(at / blocks) * piece_stride + m * blocks + at % blocks];
}

// The MoE's output from its experts' when a GEMM left them in pair order:
// y[row, d] = BF16(sum over a row's chosen experts of out[pair_of[row, slot],
// d], then shared[row, d]), added in FP32 in that order, as
// `dsv41_slots_sum` adds them. out [pairs, n] BF16, pair_of [rows, top_k],
// shared [rows, n] BF16.
extern "C" __global__ void dsv41_pairs_sum(
    const __nv_bfloat16* __restrict__ out,
    const int* __restrict__ pair_of,
    const __nv_bfloat16* __restrict__ shared,
    __nv_bfloat16* __restrict__ y,
    unsigned int top_k,
    unsigned int n,
    unsigned int rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * n) return;
    const size_t row = i / n, d = i % n;
    float total = 0.0f;
    for (unsigned int slot = 0; slot < top_k; ++slot)
        total += __bfloat162float(out[(size_t)pair_of[row * top_k + slot] * n + d]);
    total += __bfloat162float(shared[i]);
    y[i] = __float2bfloat16(total);
}
