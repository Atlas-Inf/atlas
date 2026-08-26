// SPDX-License-Identifier: AGPL-3.0-only
//
// Dual-output GEMV with on-the-fly MLX 8-bit dequantization:
//
//   gate_y[n] = sum_k gate_W_dequant[n, k] * x[k]
//   up_y[n]   = sum_k   up_W_dequant[n, k] * x[k]
//
// Both projections share the input vector `x`, so dispatching them
// in a single kernel halves the x-side memory bandwidth (the
// dequantised weights are still read once each — that part is
// bandwidth-irreducible) and removes one Metal kernel launch per FFN
// per layer (32× per token at 32 layers).
//
// Threadgroup layout matches the base `mlx_int8_gemv` kernel:
//   - 4 rows per threadgroup, one simdgroup (32 lanes) per row.
//   - 32 lanes stride over the K/4 packed words; per-row simd_sum
//     collapses the partials.
//   - Each simdgroup writes both its `gate_y[row]` and `up_y[row]`
//     output entries.
//
// Layout (matches `mlx_int8_dequant`):
//   gate_packed : uint32  [N, K / 4]
//   gate_scales : bfloat  [N, K / group_size]
//   gate_biases : bfloat  [N, K / group_size]
//   up_packed   : uint32  [N, K / 4]
//   up_scales   : bfloat  [N, K / group_size]
//   up_biases   : bfloat  [N, K / group_size]
//   x           : bfloat  [K]
//   gate_y      : bfloat  [N]
//   up_y        : bfloat  [N]
//
// Preconditions: `gate_proj` and `up_proj` must share the same
// (N, K, group_size). All Qwen3.5 SwiGLU FFNs satisfy this.

#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG_GU    = 4u;
constant uint SIMDGROUP_SIZE_GU = 32u;

kernel void mlx_int8_gemv_gate_up(
    constant uint &N           [[buffer(0)]],
    constant uint &K           [[buffer(1)]],
    constant uint &group_size  [[buffer(2)]],
    device const uint   *gate_packed [[buffer(3)]],
    device const bfloat *gate_scales [[buffer(4)]],
    device const bfloat *gate_biases [[buffer(5)]],
    device const uint   *up_packed   [[buffer(6)]],
    device const bfloat *up_scales   [[buffer(7)]],
    device const bfloat *up_biases   [[buffer(8)]],
    device const bfloat *x           [[buffer(9)]],
    device bfloat       *gate_y      [[buffer(10)]],
    device bfloat       *up_y        [[buffer(11)]],
    uint   tg_idx          [[threadgroup_position_in_grid]],
    uint   simd_lane_id    [[thread_index_in_simdgroup]],
    uint   simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_GU + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint K4              = K >> 2u;
    const uint groups_per_row  = K / group_size;
    const uint group_words     = group_size >> 2u;

    device const bfloat4 *x4    = reinterpret_cast<device const bfloat4*>(x);
    device const uint    *gprow = gate_packed + row * K4;
    device const bfloat  *gsrow = gate_scales + row * groups_per_row;
    device const bfloat  *gbrow = gate_biases + row * groups_per_row;
    device const uint    *uprow = up_packed   + row * K4;
    device const bfloat  *usrow = up_scales   + row * groups_per_row;
    device const bfloat  *ubrow = up_biases   + row * groups_per_row;

    float acc_gate = 0.0f;
    float acc_up   = 0.0f;
    for (uint k4 = simd_lane_id; k4 < K4; k4 += SIMDGROUP_SIZE_GU) {
        const uint    g  = k4 / group_words;
        const bfloat4 xv = x4[k4];

        // gate_proj
        {
            const uint  word = gprow[k4];
            const float s    = float(gsrow[g]);
            const float b    = float(gbrow[g]);
            const float w0 = float((word >>  0) & 0xFFu) * s + b;
            const float w1 = float((word >>  8) & 0xFFu) * s + b;
            const float w2 = float((word >> 16) & 0xFFu) * s + b;
            const float w3 = float((word >> 24) & 0xFFu) * s + b;
            acc_gate += w0 * float(xv.x) + w1 * float(xv.y)
                      + w2 * float(xv.z) + w3 * float(xv.w);
        }
        // up_proj
        {
            const uint  word = uprow[k4];
            const float s    = float(usrow[g]);
            const float b    = float(ubrow[g]);
            const float w0 = float((word >>  0) & 0xFFu) * s + b;
            const float w1 = float((word >>  8) & 0xFFu) * s + b;
            const float w2 = float((word >> 16) & 0xFFu) * s + b;
            const float w3 = float((word >> 24) & 0xFFu) * s + b;
            acc_up += w0 * float(xv.x) + w1 * float(xv.y)
                    + w2 * float(xv.z) + w3 * float(xv.w);
        }
    }

    const float gate_sum = simd_sum(acc_gate);
    const float up_sum   = simd_sum(acc_up);
    if (simd_lane_id == 0) {
        gate_y[row] = bfloat(gate_sum);
        up_y[row]   = bfloat(up_sum);
    }
}
