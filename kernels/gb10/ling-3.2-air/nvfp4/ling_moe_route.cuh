// SPDX-License-Identifier: AGPL-3.0-only
#pragma once

#include <cuda_bf16.h>

// Ling-3.2-air routed-MoE routing arithmetic: the DeepSeek-style
// group-limited sigmoid top-k of BailingMoeV3Gate
// (context/modeling_bailing_moe_v3_vl.py), after the GLM routing chain
// (kernels/gb10/glm-5.3-flash/nvfp4/glm_moe_route.cu). Reference, per row:
//
//   logits = x_f32 @ gate.weight_f32^T                  (Gate.forward, :486)
//   scores = sigmoid(logits), fp32                      (:487)
//   scores_for_routing = scores + expert_bias           (:488)
//   group_scores = per-group top-2 of scores_for_routing, summed (:472)
//   group_idx = top-k(group_scores, topk_group)         (:473)
//   masked = scores_for_routing masked outside the selected groups (:479)
//   probs, top_indices = top-k(masked, top_k)           (:477)
//   weights = gather(scores, top_indices)               (:490) — the RAW
//   sigmoid scores; the bias only steers selection
//   weights = scores / (scores.sum() + 1e-20)           (:492, top_k > 1)
//   weights *= routed_scaling_factor                    (:494)
//
// Ties in every round go to the lowest index (strict `>` in each thread's
// ascending scan, lowest index on equal values in every reduction), the
// torch.topk convention the GLM kernels encode. Groups are contiguous
// row-major slices of the expert axis (scores.view(tokens, n_group, -1)):
// group g covers experts [g * (E / n_group), (g + 1) * (E / n_group)).
// All arithmetic is fp32. Supported shapes: top_k <= 16, n_group <= 16,
// gsize = E / n_group a multiple of the warp width.

// Lane-local top-2 pairwise merge: two sorted descending pairs into one.
__device__ __forceinline__ void ling_top2_merge(float& m1, float& m2, float b1, float b2) {
    const float hi = fmaxf(m1, b1);
    const float lo_hi = fminf(m1, b1);
    m2 = fmaxf(lo_hi, fmaxf(m2, b2));
    m1 = hi;
}

// Full-warp top-2 by pairwise butterfly merges (the result is uniform).
__device__ __forceinline__ void ling_route_top2_warp(float& m1, float& m2) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float b1 = __shfl_xor_sync(0xFFFFFFFFu, m1, off);
        const float b2 = __shfl_xor_sync(0xFFFFFFFFu, m2, off);
        ling_top2_merge(m1, m2, b1, b2);
    }
}

// Select one row's top-k on one warp. `sig`/`sel` are this warp's shared
// slots ([E] floats each: sigmoid scores and scores + bias; `sel` is
// destroyed). Writes topk_ids[row * top_k + t] and topk_weights[...]; a
// dead row (row_live null-or-zero) gets ids E and weights 0 so it sorts to
// the tail partition and contributes nothing downstream.
__device__ __forceinline__ void ling_route_topk_row(
    const float* __restrict__ logits_row,
    const float* __restrict__ s_bias,
    float* sig, float* sel,
    unsigned int* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    const unsigned int* __restrict__ row_live,
    unsigned int row, unsigned int E, unsigned int top_k,
    unsigned int n_group, unsigned int topk_group,
    float routed_scaling_factor
) {
    const unsigned int lane = threadIdx.x % 32u;
    const unsigned int k = top_k;

    // The fused launcher's winner CTA reads the router CTAs' logits through
    // L2 (`__ldcg`): they were written by other CTAs of the same launch.
    for (unsigned int i = lane; i < E; i += 32u) {
        const float logit = __ldcg(logits_row + i);
        const float s = 1.0f / (1.0f + __expf(-logit));
        sig[i] = s;
        sel[i] = s + s_bias[i];
    }
    __syncwarp();

    if (row_live != nullptr && row_live[row] == 0u) {
        if (lane == 0) {
            for (unsigned int t = 0; t < k; ++t) {
                topk_ids[(unsigned long long)row * top_k + t] = E;
                topk_weights[(unsigned long long)row * top_k + t] = 0.0f;
            }
        }
        __syncwarp();
        return;
    }

    // Group scores: sum of the top-2 selection scores inside each group
    // (group_limited_topk's `topk(2).sum`), one group at a time, each lane
    // scanning its stride-32 experts of the group then a warp-wide top-2.
    const unsigned int gsize = E / n_group;
    float gsel[16];
    for (unsigned int g = 0; g < n_group; ++g) {
        float m1 = -1e30f, m2 = -1e30f;
        for (unsigned int j = lane; j < gsize; j += 32u) {
            const float v = sel[g * gsize + j];
            if (v > m1) { m2 = m1; m1 = v; }
            else if (v > m2) m2 = v;
        }
        ling_route_top2_warp(m1, m2);
        gsel[g] = m1 + m2;
    }

    // Lane 0 picks the top `topk_group` groups (ties -> lowest group index)
    // and masks the experts of the rest out of the selection scan.
    if (lane == 0) {
        unsigned int picked = 0u;
        for (unsigned int t = 0; t < topk_group; ++t) {
            unsigned int best_g = 0u;
            float best = -1e30f;
            for (unsigned int g = 0; g < n_group; ++g)
                if (!(picked & (1u << g)) && gsel[g] > best) {
                    best = gsel[g];
                    best_g = g;
                }
            picked |= (1u << best_g);
        }
        for (unsigned int g = 0; g < n_group; ++g)
            if (!(picked & (1u << g)))
                for (unsigned int j = 0; j < gsize; ++j) sel[g * gsize + j] = -1e30f;
    }
    __syncwarp();

    // Top-k among the surviving experts by selection score (ties -> lowest
    // index): glm_moe_route_topk_row's round structure.
    float vals[16];
    unsigned int top[16];
    for (unsigned int t = 0; t < k && t < E; ++t) {
        float best = -1e30f;
        unsigned int at = 0;
        for (unsigned int i = lane; i < E; i += 32u) {
            const float v = sel[i];
            if (v > best) { best = v; at = i; }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, best, off);
            const unsigned int other_at = __shfl_xor_sync(0xFFFFFFFFu, at, off);
            if (other > best || (other == best && other_at < at)) {
                best = other;
                at = other_at;
            }
        }
        top[t] = at;
        vals[t] = 0.0f;
        if (lane == 0) sel[at] = -1e30f;
        __syncwarp();
    }

    if (lane == 0) {
        float topk_sum = 0.0f;
        for (unsigned int t = 0; t < k && t < E; ++t) {
            vals[t] = sig[top[t]];
            topk_sum += vals[t];
        }
        for (unsigned int t = 0; t < k && t < E; ++t) {
            const float w = vals[t] / (topk_sum + 1e-20f) * routed_scaling_factor;
            topk_ids[(unsigned long long)row * top_k + t] = top[t];
            topk_weights[(unsigned long long)row * top_k + t] = w;
        }
    }
    __syncwarp();
}
