// SPDX-License-Identifier: AGPL-3.0-only
//
// DeepSeek-V4.1-Flash: the projections in four parts.
//
// Most of the wide projections are cut by output rows: a part computes a
// quarter of the output from the whole input, and the ranks gather the
// quarters. Two are cut by input columns: the Engram's, each part reading
// the table rows of a quarter of the hash columns, and for a few rows the
// attention's last (wo_b), each part reading the low-rank output it has just
// computed, so that nothing is gathered between the two output projections.
// A column part's sum is an FP32 partial, and the four are added in part
// order and rounded once:
//
//   out = BF16(((p0 + p1) + p2) + p3)
//
// One Spark computes all four; four Sparks compute one each and gather them.
// Either way the same kernels see the same operands and the sum has the same
// order, so the bytes are the same.

#include <cuda_bf16.h>

#include "dsv41_common.cuh"

// Value d of row `first` + r of out = BF16(sum over parts of parts[part, r,
// d]). A part is [rows, n] FP32 and the next starts `stride` values on. out
// is as the ranks leave a projection cut by rows, which is how a site reads
// a sublayer's output (`dsv41_hc_gathered`): four parts, each a quarter of
// the n values of every row, with room for `out_rows` rows.
extern "C" __global__ void dsv41_parts_round(
    const float* __restrict__ parts,
    __nv_bfloat16* __restrict__ out,
    unsigned int rows,
    unsigned int n,
    unsigned int stride,
    unsigned int first,
    unsigned int out_rows) {
    atlas_pdl_enter();
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * n) return;
    const unsigned int r = i / n, d = i % n, quarter = n / DSV41_PARTS;
    out[((size_t)(d / quarter) * out_rows + first + r) * quarter + d % quarter] =
        __float2bfloat16(dsv41_parts_sum(parts, stride, i));
}

// The MoE's output from its experts': y[row, d] = BF16(sum over slots of
// out[row, slot, d]), added in FP32 in slot order (the chosen experts in
// expert order, then the shared expert). out [rows, slots, n] BF16. With
// `shared` ([rows, n] BF16), the last slot is read from there.
extern "C" __global__ void dsv41_slots_sum(
    const __nv_bfloat16* __restrict__ out,
    __nv_bfloat16* __restrict__ y,
    unsigned int slots,
    unsigned int n,
    unsigned int rows,
    const __nv_bfloat16* __restrict__ shared) {
    atlas_pdl_enter();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * n) return;
    const size_t row = i / n, d = i % n;
    const unsigned int own = shared ? slots - 1u : slots;
    float total = 0.0f;
    for (unsigned int slot = 0; slot < own; ++slot)
        total += __bfloat162float(out[(row * slots + slot) * n + d]);
    if (shared) total += __bfloat162float(shared[i]);
    y[i] = __float2bfloat16(total);
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
    atlas_pdl_enter();
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * width) return;
    const size_t row = i / width, col = i % width;
    dst[i] = src[row * src_stride + first + col];
}
