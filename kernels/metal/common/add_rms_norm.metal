// SPDX-License-Identifier: AGPL-3.0-only
//
// Fused residual-add + RMSNorm:
//
//   x_resid[i] = a[i] + b[i]
//   out[i]     = (x_resid[i] / sqrt(mean(x_resid^2) + eps)) * weight[i]
//
// Replaces the unfused `bf16_add → rms_norm` pair on the post-attention
// and post-FFN paths in the decoder layer. Saves one launch + one
// HIDDEN-sized BF16 store/load between the two ops.
//
// One threadgroup per token. Threads cooperate on the sum-of-squares
// reduction (FP32 accumulation), then re-read the buffered residual
// in the second pass and emit BF16 outputs.
//
// Layout:
//   a       : bfloat [num_tokens, hidden_size]
//   b       : bfloat [num_tokens, hidden_size]
//   weight  : bfloat [hidden_size]
//   x_resid : bfloat [num_tokens, hidden_size]   (in/out — staging)
//   out     : bfloat [num_tokens, hidden_size]
//
// Caller is responsible for sizing `x_resid` so the layer chain has
// access to the residual stream after this kernel runs (the next
// layer's input).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS_ARN = 32;

kernel void add_rms_norm(
    constant uint  &hidden_size [[buffer(0)]],
    constant float &eps         [[buffer(1)]],
    device const bfloat *a       [[buffer(2)]],
    device const bfloat *b       [[buffer(3)]],
    device const bfloat *weight  [[buffer(4)]],
    device bfloat       *x_resid [[buffer(5)]],
    device bfloat       *out     [[buffer(6)]],
    uint   tok_idx       [[threadgroup_position_in_grid]],
    uint   tid           [[thread_position_in_threadgroup]],
    uint   tg_size       [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS_ARN];

    const uint base = tok_idx * hidden_size;

    // Pass 1: residual add, write x_resid, accumulate sum-of-squares.
    float local_ssq = 0.0f;
    for (uint i = tid; i < hidden_size; i += tg_size) {
        const float ai = float(a[base + i]);
        const float bi = float(b[base + i]);
        const float xi = ai + bi;
        x_resid[base + i] = bfloat(xi);
        local_ssq += xi * xi;
    }

    // Reduce across simdgroups.
    const float simd_ssq = simd_sum(local_ssq);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_ssq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            partial[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total_ssq = partial[0];

    // Pass 2: normalize. Re-read x_resid (now contains a+b in BF16).
    const float inv_rms = rsqrt(total_ssq / float(hidden_size) + eps);
    for (uint i = tid; i < hidden_size; i += tg_size) {
        const float xi = float(x_resid[base + i]);
        const float wi = float(weight[i]);
        out[base + i] = bfloat(xi * inv_rms * wi);
    }
}
