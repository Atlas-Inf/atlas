// SPDX-License-Identifier: AGPL-3.0-only
//
// LayerNorm with bias:
//
//   out[i] = ((x[i] - mean(x)) / sqrt(var(x) + eps)) * weight[i]
//                                                   + bias[i]
//
// Distinct from `rms_norm`: subtracts the mean before normalizing
// AND multiplies-then-adds bias. ViT-style vision blocks (Qwen3.5-VL
// `vision_tower.blocks.*.norm1/norm2`) use this.
//
// One threadgroup per token; same two-stage reduction (simdgroup →
// cross-simdgroup via threadgroup memory) as `rms_norm`. FP32
// accumulation throughout.
//
// Layout:
//   x      : bfloat [num_tokens, hidden_size]
//   weight : bfloat [hidden_size]
//   bias   : bfloat [hidden_size]
//   out    : bfloat [num_tokens, hidden_size]

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void layer_norm(
    constant uint  &hidden_size [[buffer(0)]],
    constant float &eps         [[buffer(1)]],
    device const bfloat *x      [[buffer(2)]],
    device const bfloat *weight [[buffer(3)]],
    device const bfloat *bias   [[buffer(4)]],
    device bfloat       *out    [[buffer(5)]],
    uint   tok_idx [[threadgroup_position_in_grid]],
    uint   tid     [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];
    threadgroup float shared_mean;
    threadgroup float shared_var;

    // ── 1. Mean reduction ───────────────────────────────────────
    float local_sum = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        local_sum += float(x[tok_idx * hidden_size + i]);
    }
    float simd_sum_v = simd_sum(local_sum);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            shared_mean = v / float(hidden_size);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mean = shared_mean;

    // ── 2. Variance reduction (E[(x - mean)²]) ──────────────────
    float local_var = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float c = float(x[tok_idx * hidden_size + i]) - mean;
        local_var += c * c;
    }
    float simd_var = simd_sum(local_var);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_var;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            shared_var = v / float(hidden_size);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float var = shared_var;

    // ── 3. Normalize + scale + shift ────────────────────────────
    float inv_std = rsqrt(var + eps);
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float xi = float(x[tok_idx * hidden_size + i]);
        float w  = float(weight[i]);
        float b  = float(bias[i]);
        out[tok_idx * hidden_size + i] = bfloat((xi - mean) * inv_std * w + b);
    }
}
