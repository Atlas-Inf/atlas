// SPDX-License-Identifier: AGPL-3.0-only
//
// Per-head and element-wise helper kernels for the GDN
// (gated delta rule) forward path.
//
// All kernels follow the cuda-side conventions: input/output
// pointers + scalar dims via `constant uint &`. Internal FP32
// accumulation; BF16 in/out where applicable.

#include <metal_stdlib>
using namespace metal;

// ── gdn_compute_gate ────────────────────────────────────────────
//
// Per-head gate computation (matches Mamba-2 SSD form):
//
//   dt[h]    = softplus(dt_raw[h] + dt_bias[h])
//   A[h]     = -exp(A_log[h])
//   gate[h]  = exp(dt[h] * A[h])             (always in (0, 1))
//
// Layout:
//   dt_raw  : bfloat [num_heads]
//   dt_bias : bfloat [num_heads]
//   A_log   : float  [num_heads]
//   gate    : float  [num_heads]               (FP32 — fed into GDN
//                                                decode which expects
//                                                float gate)
kernel void gdn_compute_gate(
    constant uint &num_heads     [[buffer(0)]],
    device const bfloat *dt_raw  [[buffer(1)]],
    device const bfloat *dt_bias [[buffer(2)]],
    device const float  *A_log   [[buffer(3)]],
    device float        *gate    [[buffer(4)]],
    uint h [[thread_position_in_grid]])
{
    if (h >= num_heads) {
        return;
    }
    float dt_pre = float(dt_raw[h]) + float(dt_bias[h]);
    // Numerically-stable softplus.
    float dt = (dt_pre > 20.0f) ? dt_pre : log(1.0f + exp(dt_pre));
    float a_eff = -exp(A_log[h]);
    gate[h] = exp(dt * a_eff);
}

// ── sigmoid_bf16 → float ────────────────────────────────────────
//
//   out[i] = 1 / (1 + exp(-in[i]))
//
// Input BF16, output FP32 (the GDN decode kernel expects beta as
// float). FP32-internal pipeline regardless.
//
// Layout:
//   x   : bfloat [n]
//   out : float  [n]
kernel void sigmoid_bf16_to_f32(
    constant uint &n      [[buffer(0)]],
    device const bfloat *x [[buffer(1)]],
    device float        *out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float v = float(x[gid]);
    out[gid] = 1.0f / (1.0f + exp(-v));
}

// ── silu_apply (in-place) ───────────────────────────────────────
//
//   out[i] = x[i] * sigmoid(x[i])
//
// Same activation as inside `silu_gate` but standalone, for the
// GDN gate path where `silu(z)` is computed before being
// element-wise-multiplied with the SSM output.
//
// Layout:
//   x, out : bfloat [n]
kernel void silu_apply(
    constant uint &n         [[buffer(0)]],
    device const bfloat *x   [[buffer(1)]],
    device bfloat       *out [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    float v = float(x[gid]);
    float sig = 1.0f / (1.0f + exp(-v));
    out[gid] = bfloat(v * sig);
}

// ── bf16_mul (element-wise) ─────────────────────────────────────
//
//   out[i] = a[i] * b[i]
//
// Complement to `bf16_add`. FP32-internal so the
// silu(z) * y_norm products in the GDN gate path don't lose tail
// precision through a BF16 intermediate.
//
// Layout:
//   a, b, out : bfloat [n]
kernel void bf16_mul(
    constant uint &n         [[buffer(0)]],
    device const bfloat *a   [[buffer(1)]],
    device const bfloat *b   [[buffer(2)]],
    device bfloat       *out [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) {
        return;
    }
    out[gid] = bfloat(float(a[gid]) * float(b[gid]));
}
