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
// For the grouped GEMM (`dsv41_slab_scales` below) the rows are gathered and
// their scales set apart by group: with `a_rows`, row m of q is row
// a_rows[m] of x; with `sf_rows`, its scales are row sf_rows[m] of sf, each
// group starting on a multiple of 128 rows, which is where a tile starts.
extern "C" __global__ void dsv41_mx_quant(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf,
    unsigned int M,
    unsigned int K,
    unsigned int a_stride,
    unsigned int piece,
    unsigned long long piece_stride,
    unsigned int inner_pad,
    const int* __restrict__ a_rows,
    const unsigned int* __restrict__ sf_rows) {
    const size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int blocks = K / 32u;
    if (t >= (size_t)M * blocks) return;
    const unsigned int m = (unsigned int)(t / blocks), b = (unsigned int)(t % blocks);
    const unsigned int kb = b * 32u;
    const size_t from = a_rows ? (size_t)a_rows[m] : m;
    const __nv_bfloat16* src = x + from * a_stride +
                               (piece ? (size_t)(kb / piece) * piece_stride + kb % piece : kb);
    float v[32];
    float amax = 0.0f;
    for (int i = 0; i < 32; ++i) {
        v[i] = __bfloat162float(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float scale = dsv41_fp8_scale(amax);
    unsigned char* dst = q + (size_t)m * K + kb;
    for (int i = 0; i < 32; ++i) {
        const float scaled = fminf(fmaxf(v[i] / scale, -448.0f), 448.0f);
        dst[i] = (unsigned char)__nv_cvt_float_to_fp8(dsv41_round_e4m3(scaled), __NV_SATFINITE,
                                                      __NV_E4M3);
    }
    const unsigned int sf_row = sf_rows ? sf_rows[m] : m;
    sf[mx_sf_offset(sf_row, b, inner_pad)] = (unsigned char)(dsv41_exponent(scale) + 127);
    if (b + 1u == blocks) mx_sf_zero_pad(sf, sf_row, blocks, inner_pad);
}

// The scales of one part of every routed expert of a layer, as the grouped
// GEMM reads them (`atlas_cutlass_mx84_grouped`): each projection's E8M0
// scales, row-major [n, k / 32] in the slab (`dsv41_expert.cu`: expert e at
// slab + e * stride, w1, w3 and w2 each `data` bytes then its scales), laid
// out again in tiles. tiled [experts, in, in, out], where `in` is the tiled
// size of a [n_in, k_in] projection (w1, w3) and `out` that of [n_out,
// k_out] (w2). One thread a byte of the output.
extern "C" __global__ void dsv41_slab_scales(
    const unsigned char* __restrict__ slab,
    unsigned char* __restrict__ tiled,
    unsigned int experts,
    unsigned long long stride,
    unsigned int data,
    unsigned int scales,
    unsigned int n_in,
    unsigned int k_in,
    unsigned int n_out,
    unsigned int k_out) {
    const unsigned int pad_in = (k_in / 32u + 3u) / 4u * 4u, pad_out = (k_out / 32u + 3u) / 4u * 4u;
    const size_t in = (size_t)((n_in + 127u) / 128u * 128u) * pad_in;
    const size_t out = (size_t)((n_out + 127u) / 128u * 128u) * pad_out;
    const size_t per = 2 * in + out;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= experts * per) return;
    const size_t expert = i / per, within = i % per;
    const unsigned int projection = within < in ? 0u : within < 2 * in ? 1u : 2u;
    const size_t at = within - (projection == 2u ? 2 * in : projection * in);
    const unsigned int n = projection == 2u ? n_out : n_in;
    const unsigned int inner = (projection == 2u ? k_out : k_in) / 32u;
    const unsigned int pad = projection == 2u ? pad_out : pad_in;
    // The scale this byte of a tile holds: `mx_sf_offset` inverted.
    const size_t tile = at / 512u;
    const unsigned int o = (unsigned int)(at % 512u);
    const unsigned int row = (unsigned int)(tile / (pad / 4u)) * 128u + ((o % 16u) / 4u) * 32u + o / 16u;
    const unsigned int col = (unsigned int)(tile % (pad / 4u)) * 4u + o % 4u;
    const unsigned char* src =
        slab + expert * stride + (size_t)projection * (data + scales) + data;
    tiled[i] = (row < n && col < inner) ? src[(size_t)row * inner + col] : (unsigned char)0;
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
