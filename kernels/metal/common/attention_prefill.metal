// SPDX-License-Identifier: AGPL-3.0-only
//
// Prefill-path scaled-dot-product attention with causal masking.
// Multi-token query attends to a contiguous K/V history in a single
// kernel launch (no KV-cache-page indirection — that's the paged
// variant).
//
//   scores[m, s] = (Q[m, h] · K[s, h_kv]) / sqrt(head_dim)
//   masked: scores[m, s] = -∞ if s > m   (causal)
//   softmax over s
//   out[m, h, d] = sum_s(softmax_s * V[s, h_kv, d])
//
// One threadgroup per (query token, head) pair. Per-row score vector
// in threadgroup memory (cap MAX_SEQ_PREFILL).
//
// Layout:
//   q   : bfloat [num_tokens, num_heads,   head_dim]
//   k   : bfloat [seq_len,    num_kv_heads, head_dim]
//   v   : bfloat [seq_len,    num_kv_heads, head_dim]
//   out : bfloat [num_tokens, num_heads,   head_dim]

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_PREFILL = 4096;

kernel void attention_prefill(
    constant uint  &num_tokens   [[buffer(0)]],
    constant uint  &seq_len      [[buffer(1)]],
    constant uint  &num_heads    [[buffer(2)]],
    constant uint  &num_kv_heads [[buffer(3)]],
    constant uint  &head_dim     [[buffer(4)]],
    constant float &scale        [[buffer(5)]],
    device const bfloat *q       [[buffer(6)]],
    device const bfloat *k       [[buffer(7)]],
    device const bfloat *v       [[buffer(8)]],
    device bfloat       *out     [[buffer(9)]],
    uint  tg_idx  [[threadgroup_position_in_grid]],
    uint  tid     [[thread_position_in_threadgroup]],
    uint  tg_size [[threads_per_threadgroup]])
{
    threadgroup float scores[MAX_SEQ_PREFILL];
    threadgroup float max_score;
    threadgroup float sum_exp;

    // Flat 1-D grid dispatch: caller sends `num_heads * num_tokens`
    // threadgroups; we decode (m, h) here. Using uint3 builtins for
    // a 2-D grid would also work but Metal forbids mixing scalar
    // and vector position attributes in one entry point.
    uint h = tg_idx % num_heads;
    uint m = tg_idx / num_heads;
    if (m >= num_tokens || h >= num_heads) {
        return;
    }
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;
    // Causal cutoff: this query can attend to keys at positions
    // [0, m] inclusive — assumes Q occupies positions [0, num_tokens)
    // and K/V cover the same range.
    uint cutoff = m + 1u;

    // Stage 1: scores. Mask everything past the causal cutoff to -∞
    // so the softmax exp drives them to 0.
    for (uint s = tid; s < seq_len && s < MAX_SEQ_PREFILL; s += tg_size) {
        if (s >= cutoff) {
            scores[s] = -INFINITY;
            continue;
        }
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            float qv = float(q[(m * num_heads + h) * head_dim + d]);
            float kvv = float(k[(s * num_kv_heads + kv_h) * head_dim + d]);
            dot += qv * kvv;
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: max reduction.
    if (tid == 0) {
        float mx = -INFINITY;
        for (uint s = 0; s < seq_len; ++s) {
            if (scores[s] > mx) mx = scores[s];
        }
        max_score = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 3: exp(score - max).
    for (uint s = tid; s < seq_len; s += tg_size) {
        scores[s] = exp(scores[s] - max_score);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 4: sum.
    if (tid == 0) {
        float sum = 0.0f;
        for (uint s = 0; s < seq_len; ++s) {
            sum += scores[s];
        }
        sum_exp = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 5: out[m, h, d] = sum_s(softmax_s * V[s, kv_h, d]).
    float inv_sum = 1.0f / sum_exp;
    for (uint d = tid; d < head_dim; d += tg_size) {
        float acc = 0.0f;
        for (uint s = 0; s < seq_len; ++s) {
            float vv = float(v[(s * num_kv_heads + kv_h) * head_dim + d]);
            acc += scores[s] * inv_sum * vv;
        }
        out[(m * num_heads + h) * head_dim + d] = bfloat(acc);
    }
}
