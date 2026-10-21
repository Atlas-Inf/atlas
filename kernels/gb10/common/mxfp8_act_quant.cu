// SPDX-License-Identifier: AGPL-3.0-only
// Ported from Enntity/atlas (branch `sparkqwen` @ 92dd543, AGPL-3.0-only),
// `kernels/gb10/common/mxfp8_act_quant.cu`, with Atlas-Inf/atlas#198's integration changes.
//
// MXFP8 operands for the cuBLASLt W8A8 prefill GEMM
// (`spark_runtime::cublaslt::fp8_gemm_act_weight_t_mx`); rounding and scale
// layout are documented in `mxfp8_act_quant.cuh`.
//
//   mxfp8_quant_act_bf16: BF16 [M, K] (row stride lda elements) -> E4M3
//     [M, K] (contiguous) + UE8M0 scales in cuBLASLt's tiled layout. One
//     thread per 8 values (16 B load, 8 B store), a quad per 32-value block.
//     K % 32 == 0, lda % 8 == 0. Grid: (ceil(M*K/8 / 256), 1, 1) Block: 256.
//
//   mxfp8_lt_scale_tile: a row-major UE8M0 scale matrix [rows, inner] (row
//     stride src_ld bytes, so a TP rank's K columns are `src + col0` with
//     the full row's stride) -> the tiled layout, every destination byte
//     written (padding 0): one thread per destination byte over
//     round_up(rows, 128) x inner_pad. The checkpoint's weight_scale is
//     row-major [n, k/32]; this is its one-time relayout at load.

#include <cuda_bf16.h>
#include "mxfp8_act_quant.cuh"

extern "C" __global__ void __launch_bounds__(256) mxfp8_quant_act_bf16(
    const __nv_bfloat16* __restrict__ x,
    unsigned char* __restrict__ q,
    unsigned char* __restrict__ sf,
    const unsigned int M,
    const unsigned int K,
    const unsigned int lda,
    const unsigned int inner_pad
) {
    const unsigned long long t = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int per_row = K / 8u;
    const unsigned long long total = (unsigned long long)M * per_row;
    // Lanes past the end still join the quad shuffle (on zeros), then stop.
    const bool live = t < total;
    const unsigned int r = live ? (unsigned int)(t / per_row) : 0u;
    const unsigned int c8 = live ? (unsigned int)(t % per_row) : 0u;
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (live) raw = *reinterpret_cast<const uint4*>(x + (size_t)r * lda + (size_t)c8 * 8u);
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
    float v[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(h[j]);
        v[2 * j] = f.x;
        v[2 * j + 1] = f.y;
    }
    uint2 packed;
    const unsigned char s = mx_quant8_quad(v, packed);
    if (!live) return;
    *reinterpret_cast<uint2*>(q + (size_t)r * K + (size_t)c8 * 8u) = packed;
    if ((c8 & 3u) == 0u) {
        const unsigned int blk = c8 / 4u;
        sf[mx_sf_offset(r, blk, inner_pad)] = s;
        if (blk + 1u == K / 32u) mx_sf_zero_pad(sf, r, blk + 1u, inner_pad);
    }
}

extern "C" __global__ void __launch_bounds__(256) mxfp8_lt_scale_tile(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    const unsigned int rows,
    const unsigned int inner,
    const unsigned int src_ld,
    const unsigned int inner_pad
) {
    const unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int rows_pad = (rows + 127u) / 128u * 128u;
    if (i >= (unsigned long long)rows_pad * inner_pad) return;
    // Destination-major: invert `mx_sf_offset` -- in-tile byte o holds
    // outer % 32 = o / 16, (outer % 128) / 32 = (o % 16) / 4, inner % 4 = o % 4.
    const unsigned int tile = (unsigned int)(i / 512u);
    const unsigned int o = (unsigned int)(i % 512u);
    const unsigned int tiles_per_row = inner_pad / 4u;
    const unsigned int outer = (tile / tiles_per_row) * 128u + ((o % 16u) / 4u) * 32u + o / 16u;
    const unsigned int col = (tile % tiles_per_row) * 4u + o % 4u;
    dst[i] = (outer < rows && col < inner) ? src[(size_t)outer * src_ld + col] : (unsigned char)0;
}
