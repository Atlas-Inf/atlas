// SPDX-License-Identifier: AGPL-3.0-only
//
// Decode-path GEMV with on-the-fly MLX 8-bit dequantization:
//
//   y[n] = sum_k W_dequant[n, k] * x[k]
//
// Each threadgroup handles 4 consecutive output rows (`ROWS_PER_TG`).
// One simdgroup (32 threads) per row keeps the cross-row reduction
// trivial: each row's partial sums collapse with a single
// `simd_sum`, no threadgroup barrier required.
//
// Two tiers of optimisation vs the prior byte-loop kernel:
// 1. Iterate over packed uint32 *words* (4 weight bytes each), not
//    bytes — eliminates 3/4 of the redundant packed/scale/bias loads
//    that the byte loop emitted on consecutive iterations.
// 2. Vector-load 4 consecutive `x` values via `bfloat4` so the
//    hardware coalesces them into one 8-byte transaction.
// 3. 4 rows per threadgroup → x[] reads are shared across 4 rows
//    via the L2 cache, cutting input-side bandwidth by 4×.
//
// Layout (matches `mlx_int8_dequant`):
//   packed : uint32  [N, K / 4]
//   scales : bfloat  [N, K / group_size]
//   biases : bfloat  [N, K / group_size]
//   x      : bfloat  [K]
//   y      : bfloat  [N]
//
// Preconditions: K % 4 == 0, group_size % 4 == 0, group_size <= K.
// All hold for every MLX 8-bit linear in Qwen3.5 (group_size=64).
// Caller dispatches `ceil(N/4)` threadgroups, 128 threads each
// (`ROWS_PER_TG * SIMDGROUP_SIZE = 4 * 32`).

#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG    = 4u;
constant uint SIMDGROUP_SIZE = 32u;

kernel void mlx_int8_gemv(
    constant uint &N          [[buffer(0)]],
    constant uint &K          [[buffer(1)]],
    constant uint &group_size [[buffer(2)]],
    device const uint   *packed [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]],
    device const bfloat *biases [[buffer(5)]],
    device const bfloat *x      [[buffer(6)]],
    device bfloat       *y      [[buffer(7)]],
    uint   tg_idx          [[threadgroup_position_in_grid]],
    uint   simd_lane_id    [[thread_index_in_simdgroup]],
    uint   simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint K4              = K >> 2u;            // packed words per row
    const uint groups_per_row  = K / group_size;
    const uint group_words     = group_size >> 2u;

    device const bfloat4 *x4 = reinterpret_cast<device const bfloat4*>(x);
    device const uint    *prow = packed + row * K4;
    device const bfloat  *srow = scales + row * groups_per_row;
    device const bfloat  *brow = biases + row * groups_per_row;

    float acc = 0.0f;
    // Each simdgroup lane (32 wide) strides over packed words.
    for (uint k4 = simd_lane_id; k4 < K4; k4 += SIMDGROUP_SIZE) {
        const uint   word = prow[k4];
        const uint   g    = k4 / group_words;
        const float  s    = float(srow[g]);
        const float  b    = float(brow[g]);
        const bfloat4 xv  = x4[k4];

        // Same scale/bias for all four bytes — group_size ≥ 4.
        const float w0 = float((word >>  0) & 0xFFu) * s + b;
        const float w1 = float((word >>  8) & 0xFFu) * s + b;
        const float w2 = float((word >> 16) & 0xFFu) * s + b;
        const float w3 = float((word >> 24) & 0xFFu) * s + b;

        acc += w0 * float(xv.x) + w1 * float(xv.y)
             + w2 * float(xv.z) + w3 * float(xv.w);
    }

    // simd_sum collapses the 32 lane partials in-place; no
    // threadgroup barrier needed because each row stays inside its
    // own simdgroup.
    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum);
    }
}
