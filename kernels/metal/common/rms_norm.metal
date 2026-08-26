// SPDX-License-Identifier: AGPL-3.0-only
//
// RMSNorm: out[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]
//
// Layout:
//   x      : bfloat [num_tokens, hidden_size]
//   weight : bfloat [hidden_size]
//   out    : bfloat [num_tokens, hidden_size]
//
// One threadgroup per token. Threads in the group cooperate to
// compute the sum-of-squares via a simdgroup reduction followed by
// a per-simdgroup-leader threadgroup reduction in shared memory.
// Same numeric profile as the CUDA `rms_norm` kernel (FP32
// accumulation, BF16 inputs/outputs).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void rms_norm(
    constant uint  &hidden_size [[buffer(0)]],
    constant float &eps         [[buffer(1)]],
    device const bfloat *x      [[buffer(2)]],
    device const bfloat *weight [[buffer(3)]],
    device bfloat       *out    [[buffer(4)]],
    uint   tok_idx [[threadgroup_position_in_grid]],
    uint   tid     [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];

    // Sum of squares across this token's hidden vector (FP32 accum).
    float local_ssq = 0.0;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float v = float(x[tok_idx * hidden_size + i]);
        local_ssq += v * v;
    }

    // Reduce within each simdgroup, then publish lane 0's value to
    // shared memory for a final cross-simdgroup reduction.
    float simd_sum_val = simd_sum(local_ssq);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_sum_val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction in simdgroup 0 across all simdgroup leaders.
    uint num_simds = (tg_size + 31u) / 32u;
    float total_ssq = 0.0;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0;
        v = simd_sum(v);
        if (tid == 0) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total_ssq = partial[0];

    // Apply the normalization. Use rsqrt for the conventional
    // 1/sqrt(mean + eps) formulation.
    float inv_rms = rsqrt(total_ssq / float(hidden_size) + eps);
    for (uint i = tid; i < hidden_size; i += tg_size) {
        float xi = float(x[tok_idx * hidden_size + i]);
        float wi = float(weight[i]);
        out[tok_idx * hidden_size + i] = bfloat(xi * inv_rms * wi);
    }
}
