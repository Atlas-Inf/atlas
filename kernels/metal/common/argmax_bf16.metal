// SPDX-License-Identifier: AGPL-3.0-only
//
// Argmax over a bfloat vector — returns the index of the largest
// element. Used by greedy sampling on the decode hot path.
//
// One threadgroup walks the full `n`-element vector. Each thread
// holds its local (max_val, max_idx) pair, then a simdgroup
// reduction selects the greater value within each simdgroup. The
// final cross-simdgroup reduction lives in threadgroup memory so
// we don't need atomic float operations (Metal doesn't expose
// atomic max on float).
//
// Layout:
//   logits : bfloat [n]
//   result : uint32 [1]

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS = 32;

kernel void argmax_bf16(
    constant uint &n           [[buffer(0)]],
    device const bfloat *logits [[buffer(1)]],
    device uint         *result [[buffer(2)]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]],
    uint  simd_lane_id  [[thread_index_in_simdgroup]],
    uint  simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial_val[MAX_SIMDGROUPS];
    threadgroup uint  partial_idx[MAX_SIMDGROUPS];

    // Per-thread argmax over a strided slice of `logits`.
    float best_val = -INFINITY;
    uint  best_idx = 0;
    for (uint i = tid; i < n; i += tg_size) {
        float v = float(logits[i]);
        if (v > best_val) {
            best_val = v;
            best_idx = i;
        }
    }

    // Reduce within the simdgroup using simd_shuffle. Argmax over
    // simd_size lanes via a tournament; ties favour the smaller
    // index to match the CUDA `argmax_bf16` kernel's behavior.
    for (uint offset = 16u; offset > 0u; offset >>= 1u) {
        float other_val = simd_shuffle_xor(best_val, offset);
        uint  other_idx = simd_shuffle_xor(best_idx, offset);
        bool other_wins = other_val > best_val ||
                          (other_val == best_val && other_idx < best_idx);
        best_val = other_wins ? other_val : best_val;
        best_idx = other_wins ? other_idx : best_idx;
    }

    if (simd_lane_id == 0) {
        partial_val[simd_group_id] = best_val;
        partial_idx[simd_group_id] = best_idx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Final reduction across simdgroup leaders, performed by
    // simdgroup 0 only.
    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial_val[tid] : -INFINITY;
        uint  i = (tid < num_simds) ? partial_idx[tid] : 0u;
        for (uint offset = 16u; offset > 0u; offset >>= 1u) {
            float other_v = simd_shuffle_xor(v, offset);
            uint  other_i = simd_shuffle_xor(i, offset);
            bool other_wins = other_v > v ||
                              (other_v == v && other_i < i);
            v = other_wins ? other_v : v;
            i = other_wins ? other_i : i;
        }
        if (tid == 0) {
            result[0] = i;
        }
    }
}
