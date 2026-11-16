// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: a projection whose input is split in four parts.
//
// The wide projections that sum over their input (an expert's w2, the
// attention's wo_b) are cut into four column parts, each a contiguous
// quarter of the input. A part's sum is an FP32 partial. The four partials
// are added in part order and rounded once:
//
//   out = BF16(((p0 + p1) + p2) + p3)
//
// One Spark computes all four; four Sparks compute one each and gather them.
// Either way the same kernels see the same operands and the sum has the same
// order, so the bytes are the same.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

// out[slot, d] = BF16(sum over parts of parts[part, slot, d]). A part is
// [slots, n] FP32 and the next starts `stride` values on; out [slots, n] BF16.
extern "C" __global__ void dsv41_parts_round(
    const float* __restrict__ parts,
    __nv_bfloat16* __restrict__ out,
    unsigned int slots,
    unsigned int n,
    unsigned int stride) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= slots * n) return;
    out[i] = __float2bfloat16(dsv41_parts_sum(parts, stride, i));
}

// The MoE's output: every slot (the chosen experts in expert order, then the
// shared expert) rounded to BF16 as above, then the slots added in FP32 in
// slot order and rounded: y[d] = BF16(sum over slots of BF16(sum over parts)).
extern "C" __global__ void dsv41_moe_combine(
    const float* __restrict__ parts,
    __nv_bfloat16* __restrict__ y,
    unsigned int slots,
    unsigned int n,
    unsigned int stride) {
    const unsigned int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= n) return;
    y[d] = __float2bfloat16(dsv41_parts_combine(parts, stride, slots, n, d));
}

// dst[row, 0..width) = src[row, first..first + width), bytes: one part's
// columns of a row-major weight or scale grid. Grid over rows * width.
extern "C" __global__ void dsv41_slice_columns(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    unsigned int rows,
    unsigned int src_stride,
    unsigned int first,
    unsigned int width) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * width) return;
    const size_t row = i / width, col = i % width;
    dst[i] = src[row * src_stride + first + col];
}
