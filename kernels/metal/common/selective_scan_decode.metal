// SPDX-License-Identifier: AGPL-3.0-only
//
// Mamba-2 / SSD selective-scan decode step (single-token recurrence).
// The basic state-space update that underlies Qwen3.5's GDN
// linear-attention layers — Qwen3.5's specific GDN variant adds
// gating on top of this skeleton, but the per-token state update
// follows the same arithmetic.
//
// Per state-head h (size `num_heads`):
//   A_eff[h]  = -exp(A_log[h])
//   dt[h]     = softplus(dt_raw[h] + dt_bias[h])
//   decay[h]  = exp(dt[h] * A_eff[h])
//
// Per (head h, channel c) state cell:
//   state_new[h, c] = state_old[h, c] * decay[h]
//                   + dt[h] * B[h] * x[c]
//
// Per channel c output:
//   y[c] = sum_h(state_new[h, c] * C[h])
//
// Layout:
//   A_log    : float  [num_heads]
//   dt_bias  : bfloat [num_heads]
//   dt_raw   : bfloat [num_heads]    (per-token dt projection)
//   B        : bfloat [num_heads]    (per-token B projection)
//   C        : bfloat [num_heads]    (per-token C projection)
//   x        : bfloat [num_channels]
//   state    : bfloat [num_heads, num_channels]    (in-place update)
//   y        : bfloat [num_channels]
//
// Grid: one threadgroup per channel. Threads inside the group
// cooperate over the heads-axis reduction for the output sum.
// FP32 accumulation throughout.

#include <metal_stdlib>
using namespace metal;

constant uint MAX_SIMDGROUPS_SSM = 32;

kernel void selective_scan_decode(
    constant uint  &num_heads    [[buffer(0)]],
    constant uint  &num_channels [[buffer(1)]],
    device const float  *A_log   [[buffer(2)]],
    device const bfloat *dt_bias [[buffer(3)]],
    device const bfloat *dt_raw  [[buffer(4)]],
    device const bfloat *B       [[buffer(5)]],
    device const bfloat *C       [[buffer(6)]],
    device const bfloat *x       [[buffer(7)]],
    device bfloat       *state   [[buffer(8)]],
    device bfloat       *y       [[buffer(9)]],
    uint   ch_idx [[threadgroup_position_in_grid]],
    uint   tid    [[thread_position_in_threadgroup]],
    uint   tg_size [[threads_per_threadgroup]],
    uint   simd_lane_id  [[thread_index_in_simdgroup]],
    uint   simd_group_id [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial[MAX_SIMDGROUPS_SSM];

    if (ch_idx >= num_channels) {
        return;
    }
    float xc = float(x[ch_idx]);

    // Each thread processes a stride of heads; computes its slice of
    // (state update + output partial sum) in FP32 then reduces.
    float local_y = 0.0f;
    for (uint h = tid; h < num_heads; h += tg_size) {
        float a_eff = -exp(A_log[h]);
        float dt_pre = float(dt_raw[h]) + float(dt_bias[h]);
        // softplus: numerically-stable log1p(exp(x))
        float dt = (dt_pre > 20.0f) ? dt_pre : log(1.0f + exp(dt_pre));
        float decay = exp(dt * a_eff);
        float bv = float(B[h]);
        float cv = float(C[h]);

        uint  state_off = h * num_channels + ch_idx;
        float old_s = float(state[state_off]);
        float new_s = old_s * decay + dt * bv * xc;
        state[state_off] = bfloat(new_s);

        local_y += new_s * cv;
    }

    // Reduce local_y across the threadgroup: simdgroup → cross-simd.
    float simd_sum_v = simd_sum(local_y);
    if (simd_lane_id == 0) {
        partial[simd_group_id] = simd_sum_v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint num_simds = (tg_size + 31u) / 32u;
    if (simd_group_id == 0) {
        float v = (tid < num_simds) ? partial[tid] : 0.0f;
        v = simd_sum(v);
        if (tid == 0) {
            y[ch_idx] = bfloat(v);
        }
    }
}
