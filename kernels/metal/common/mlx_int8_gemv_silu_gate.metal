// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused GEMV + SwiGLU input activation:
//
//   y[n] = sum_k W_dequant[n, k] * (silu(gate[k]) * up[k])
//
// Replaces the unfused `silu_gate → mlx_int8_gemv(down_proj)` pair on
// the FFN hot path. Eliminates the INTERMEDIATE-sized `ffn_act`
// staging buffer (write then re-read) and one kernel launch per
// FFN per layer.
//
// Layout matches `mlx_int8_gemv` exactly except for the input side:
//   packed : uint32  [N, K / 4]        — down_proj packed weights
//   scales : bfloat  [N, K / group_size]
//   biases : bfloat  [N, K / group_size]
//   gate   : bfloat  [K]               — gate_proj output
//   up     : bfloat  [K]               — up_proj output
//   y      : bfloat  [N]               — final FFN residual contribution
//
// Threadgroup layout = 4 rows × 32 lanes (one simdgroup per row),
// same as the base `mlx_int8_gemv` kernel.

#include <metal_stdlib>
using namespace metal;

constant uint ROWS_PER_TG_SG    = 4u;
constant uint SIMDGROUP_SIZE_SG = 32u;

kernel void mlx_int8_gemv_silu_gate(
    constant uint &N          [[buffer(0)]],
    constant uint &K          [[buffer(1)]],
    constant uint &group_size [[buffer(2)]],
    device const uint   *packed [[buffer(3)]],
    device const bfloat *scales [[buffer(4)]],
    device const bfloat *biases [[buffer(5)]],
    device const bfloat *gate   [[buffer(6)]],
    device const bfloat *up     [[buffer(7)]],
    device bfloat       *y      [[buffer(8)]],
    uint   tg_idx          [[threadgroup_position_in_grid]],
    uint   simd_lane_id    [[thread_index_in_simdgroup]],
    uint   simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_SG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint K4              = K >> 2u;
    const uint groups_per_row  = K / group_size;
    const uint group_words     = group_size >> 2u;

    device const bfloat4 *gate4 = reinterpret_cast<device const bfloat4*>(gate);
    device const bfloat4 *up4   = reinterpret_cast<device const bfloat4*>(up);
    device const uint    *prow  = packed + row * K4;
    device const bfloat  *srow  = scales + row * groups_per_row;
    device const bfloat  *brow  = biases + row * groups_per_row;

    float acc = 0.0f;
    for (uint k4 = simd_lane_id; k4 < K4; k4 += SIMDGROUP_SIZE_SG) {
        const uint    word = prow[k4];
        const uint    g    = k4 / group_words;
        const float   s    = float(srow[g]);
        const float   b    = float(brow[g]);
        const bfloat4 gv   = gate4[k4];
        const bfloat4 uv   = up4[k4];

        // Dequantise four packed weights with shared scale/bias.
        const float w0 = float((word >>  0) & 0xFFu) * s + b;
        const float w1 = float((word >>  8) & 0xFFu) * s + b;
        const float w2 = float((word >> 16) & 0xFFu) * s + b;
        const float w3 = float((word >> 24) & 0xFFu) * s + b;

        // SwiGLU input: silu(g) * u, computed in FP32 to keep the
        // sigmoid stable for large negative gate magnitudes.
        const float g0 = float(gv.x), u0 = float(uv.x);
        const float g1 = float(gv.y), u1 = float(uv.y);
        const float g2 = float(gv.z), u2 = float(uv.z);
        const float g3 = float(gv.w), u3 = float(uv.w);
        const float f0 = (g0 / (1.0f + exp(-g0))) * u0;
        const float f1 = (g1 / (1.0f + exp(-g1))) * u1;
        const float f2 = (g2 / (1.0f + exp(-g2))) * u2;
        const float f3 = (g3 / (1.0f + exp(-g3))) * u3;

        acc += w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3;
    }

    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum);
    }
}

// Variant that also folds the residual stream addition:
//   y[n] = x_resid[n] + sum_k W[n, k] * (silu(gate[k]) * up[k])
//
// Eliminates the trailing `bf16_add` kernel on the FFN exit path —
// one fewer launch and one fewer HIDDEN-size write+read per layer.
kernel void mlx_int8_gemv_silu_gate_resid(
    constant uint &N          [[buffer(0)]],
    constant uint &K          [[buffer(1)]],
    constant uint &group_size [[buffer(2)]],
    device const uint   *packed  [[buffer(3)]],
    device const bfloat *scales  [[buffer(4)]],
    device const bfloat *biases  [[buffer(5)]],
    device const bfloat *gate    [[buffer(6)]],
    device const bfloat *up      [[buffer(7)]],
    device const bfloat *x_resid [[buffer(8)]],
    device bfloat       *y       [[buffer(9)]],
    uint   tg_idx          [[threadgroup_position_in_grid]],
    uint   simd_lane_id    [[thread_index_in_simdgroup]],
    uint   simd_group_id   [[simdgroup_index_in_threadgroup]])
{
    const uint row = tg_idx * ROWS_PER_TG_SG + simd_group_id;
    if (row >= N) {
        return;
    }

    const uint K4              = K >> 2u;
    const uint groups_per_row  = K / group_size;
    const uint group_words     = group_size >> 2u;

    device const bfloat4 *gate4 = reinterpret_cast<device const bfloat4*>(gate);
    device const bfloat4 *up4   = reinterpret_cast<device const bfloat4*>(up);
    device const uint    *prow  = packed + row * K4;
    device const bfloat  *srow  = scales + row * groups_per_row;
    device const bfloat  *brow  = biases + row * groups_per_row;

    float acc = 0.0f;
    for (uint k4 = simd_lane_id; k4 < K4; k4 += SIMDGROUP_SIZE_SG) {
        const uint    word = prow[k4];
        const uint    g    = k4 / group_words;
        const float   s    = float(srow[g]);
        const float   b    = float(brow[g]);
        const bfloat4 gv   = gate4[k4];
        const bfloat4 uv   = up4[k4];

        const float w0 = float((word >>  0) & 0xFFu) * s + b;
        const float w1 = float((word >>  8) & 0xFFu) * s + b;
        const float w2 = float((word >> 16) & 0xFFu) * s + b;
        const float w3 = float((word >> 24) & 0xFFu) * s + b;

        const float g0 = float(gv.x), u0 = float(uv.x);
        const float g1 = float(gv.y), u1 = float(uv.y);
        const float g2 = float(gv.z), u2 = float(uv.z);
        const float g3 = float(gv.w), u3 = float(uv.w);
        const float f0 = (g0 / (1.0f + exp(-g0))) * u0;
        const float f1 = (g1 / (1.0f + exp(-g1))) * u1;
        const float f2 = (g2 / (1.0f + exp(-g2))) * u2;
        const float f3 = (g3 / (1.0f + exp(-g3))) * u3;

        acc += w0 * f0 + w1 * f1 + w2 * f2 + w3 * f3;
    }

    const float row_sum = simd_sum(acc);
    if (simd_lane_id == 0) {
        y[row] = bfloat(row_sum + float(x_resid[row]));
    }
}
