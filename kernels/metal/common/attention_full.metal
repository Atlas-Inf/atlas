// SPDX-License-Identifier: AGPL-3.0-only
//
// Non-causal full self-attention. Used by ViT-style vision towers and
// any encoder-only block where every query attends to every key.
// Identical structure to `attention_prefill` but without the
// `s > m → -∞` mask — every (m, h) attends to all `seq_len` keys.
//
// Layout:
//   q   : bfloat [num_tokens, num_heads,   head_dim]
//   k   : bfloat [seq_len,    num_kv_heads, head_dim]
//   v   : bfloat [seq_len,    num_kv_heads, head_dim]
//   out : bfloat [num_tokens, num_heads,   head_dim]
//
// One threadgroup per (token, head); flat 1-D grid (Metal forbids
// mixing scalar and uint3 position attributes in one kernel).

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SEQ_FULL = 4096;

kernel void attention_full(
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
    threadgroup float scores[MAX_SEQ_FULL];
    threadgroup float max_score;
    threadgroup float sum_exp;

    uint h = tg_idx % num_heads;
    uint m = tg_idx / num_heads;
    if (m >= num_tokens || h >= num_heads) {
        return;
    }
    uint group = num_heads / num_kv_heads;
    uint kv_h  = h / group;

    // Stage 1: scores. No causal mask — every query sees every key.
    for (uint s = tid; s < seq_len && s < MAX_SEQ_FULL; s += tg_size) {
        float dot = 0.0f;
        for (uint d = 0; d < head_dim; ++d) {
            float qv = float(q[(m * num_heads + h) * head_dim + d]);
            float kvv = float(k[(s * num_kv_heads + kv_h) * head_dim + d]);
            dot += qv * kvv;
        }
        scores[s] = dot * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: max for numerical-stable softmax.
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
