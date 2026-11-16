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
    const unsigned char code = (unsigned char)(dsv41_exponent(scale) + 127);
    if (inner_pad == 0u) {
        sf[t] = code;
        return;
    }
    sf[mx_sf_offset(m, b, inner_pad)] = code;
    if (b + 1u == blocks) mx_sf_zero_pad(sf, m, blocks, inner_pad);
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
