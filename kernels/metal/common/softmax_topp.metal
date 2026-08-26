// SPDX-License-Identifier: AGPL-3.0-only
//
// Top-p (nucleus) sampler. Combines softmax-with-temperature, sorted
// cumulative probability truncation at threshold `p`, and a single
// CDF-based draw using a host-supplied uniform `[0, 1)` sample.
//
// One threadgroup processes the full vocab. Approach:
//   1. Find max(logits) for numerical stability.
//   2. Compute z[i] = exp((logits[i] - max) / temp); accumulate sum.
//   3. Find the smallest threshold T such that
//        sum_{i where z[i] >= T} z[i]  >=  p * total_sum
//      via a binary search on T over the [0, max(z)] range — this
//      avoids needing a sorted vocab in shared memory (vocab is too
//      big).
//   4. Renormalise the surviving tokens, then pick by inverse-CDF
//      lookup of `uniform * surviving_sum`.
//
// Output is a single token id in `result[0]`.
//
// Layout:
//   logits : bfloat [vocab]
//   result : uint32 [1]

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;
// Binary search precision — 24 iterations halves the bracket 2^-24
// times, deeper than BF16 mantissa precision warrants.
constant uint BSEARCH_ITERS = 24;

kernel void softmax_topp(
    constant uint  &vocab    [[buffer(0)]],
    constant float &temp     [[buffer(1)]],
    constant float &p        [[buffer(2)]],
    constant float &uniform  [[buffer(3)]],
    device const bfloat *logits [[buffer(4)]],
    device uint         *result [[buffer(5)]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS];
    threadgroup float shared_max;
    threadgroup float shared_total;
    threadgroup float shared_threshold;
    threadgroup float shared_surviving_sum;
    threadgroup uint  shared_pick;

    // ── 1. max(logits) ──────────────────────────────────────────
    float local_max = -INFINITY;
    for (uint i = tid; i < vocab; i += tg_size) {
        float v = float(logits[i]);
        if (v > local_max) local_max = v;
    }
    float simd_max_v = simd_max(local_max);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_max_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        uint num_simds = (tg_size + 31u) / 32u;
        float v = (tid < num_simds) ? partial[tid] : -INFINITY;
        v = simd_max(v);
        if (tid == 0) shared_max = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float mx = shared_max;

    // ── 2. total_sum = sum_i z[i] where z = exp((l - mx)/temp) ──
    float local_sum = 0.0f;
    for (uint i = tid; i < vocab; i += tg_size) {
        float z = exp((float(logits[i]) - mx) / temp);
        local_sum += z;
    }
    float simd_sum_v = simd_sum(local_sum);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_group_id == 0) {
        uint num_simds = (tg_size + 31u) / 32u;
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) shared_total = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = shared_total;
    float p_target = p * total;

    // ── 3. Binary search for threshold T over [0, 1] (post-mx
    //    normalisation z values lie in (0, 1] since mx is the max).
    if (tid == 0) {
        float lo = 0.0f;
        float hi = 1.0f;
        for (uint iter = 0; iter < BSEARCH_ITERS; ++iter) {
            float mid = 0.5f * (lo + hi);
            float surviving = 0.0f;
            for (uint i = 0; i < vocab; ++i) {
                float z = exp((float(logits[i]) - mx) / temp);
                if (z >= mid) {
                    surviving += z;
                }
            }
            if (surviving >= p_target) {
                lo = mid; // we can afford a tighter (higher) threshold
            } else {
                hi = mid;
            }
        }
        shared_threshold = lo;
        // Compute surviving_sum at the chosen threshold.
        float surv = 0.0f;
        for (uint i = 0; i < vocab; ++i) {
            float z = exp((float(logits[i]) - mx) / temp);
            if (z >= lo) surv += z;
        }
        shared_surviving_sum = surv;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── 4. Inverse-CDF pick over surviving tokens ───────────────
    if (tid == 0) {
        float threshold = shared_threshold;
        float target = uniform * shared_surviving_sum;
        float run = 0.0f;
        uint pick = 0;
        for (uint i = 0; i < vocab; ++i) {
            float z = exp((float(logits[i]) - mx) / temp);
            if (z >= threshold) {
                run += z;
                if (run >= target) {
                    pick = i;
                    break;
                }
                pick = i; // fallback: last surviving index
            }
        }
        shared_pick = pick;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        result[0] = shared_pick;
    }
}
