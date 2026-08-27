// SPDX-License-Identifier: AGPL-3.0-only

//! Gated-delta-net reference — the recurrence, in the order that matters.

use super::{gdn_delta_step, linear, sigmoid};

/// Gated-delta-net weights for one linear-attention layer.
pub struct GdnWeights<'a> {
    /// `[2*key_dim + value_dim, hidden]` — q, k and v fused.
    pub in_proj_qkv: &'a [f32],
    /// `[value_dim, hidden]` — the output gate.
    pub in_proj_z: &'a [f32],
    /// `[num_v_heads, hidden]`
    pub in_proj_a: &'a [f32],
    pub in_proj_b: &'a [f32],
    /// `[2*key_dim + value_dim, kernel]`, depthwise causal.
    pub conv1d: &'a [f32],
    /// `[num_v_heads]`
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    /// `[value_head_dim]` — a PLAIN multiplier here, not an offset from 1.
    /// `Qwen4ExpTextRMSNormGated` differs from `Qwen4ExpTextRMSNorm` in exactly
    /// this, and they sit a few lines apart in the reference.
    pub norm: &'a [f32],
    /// `[hidden, value_dim]`
    pub out_proj: &'a [f32],
}

pub struct GdnDims {
    pub hidden: usize,
    pub num_k_heads: usize,
    pub key_head_dim: usize,
    pub num_v_heads: usize,
    pub value_head_dim: usize,
    pub conv_kernel: usize,
    pub eps: f32,
    /// `true` for `output_gate_type = "sigmoid"`, `false` for SiLU.
    pub sigmoid_gate: bool,
}

fn softplus(x: f32) -> f32 {
    // Numerically stable: log1p(exp(-|x|)) + max(x, 0).
    (-x.abs()).exp().ln_1p() + x.max(0.0)
}

fn l2norm(x: &mut [f32], eps: f32) {
    let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f32>() + eps).sqrt();
    for v in x {
        *v *= inv;
    }
}

/// One gated-delta-net layer over a whole sequence, from a zero state.
///
/// `hidden` is `[seq, hidden]`; the result is the same shape.
pub fn gdn_forward(dims: &GdnDims, w: &GdnWeights<'_>, hidden: &[f32]) -> Vec<f32> {
    let h = dims.hidden;
    let seq = hidden.len() / h;
    let key_dim = dims.num_k_heads * dims.key_head_dim;
    let value_dim = dims.num_v_heads * dims.value_head_dim;
    let conv_dim = key_dim * 2 + value_dim;
    let group = dims.num_v_heads / dims.num_k_heads;
    let (kd, vd) = (dims.key_head_dim, dims.value_head_dim);

    // Project, then a depthwise causal conv with SiLU over the fused q/k/v.
    let mut projected = vec![0f32; seq * conv_dim];
    for t in 0..seq {
        projected[t * conv_dim..(t + 1) * conv_dim].copy_from_slice(&linear(
            &hidden[t * h..(t + 1) * h],
            w.in_proj_qkv,
            conv_dim,
            h,
        ));
    }
    let mut mixed = vec![0f32; seq * conv_dim];
    for t in 0..seq {
        for c in 0..conv_dim {
            let mut acc = 0f32;
            for tap in 0..dims.conv_kernel {
                // Left-padded: tap `kernel-1` is the current position.
                if let Some(src) = t.checked_sub(dims.conv_kernel - 1 - tap) {
                    acc += w.conv1d[c * dims.conv_kernel + tap] * projected[src * conv_dim + c];
                }
            }
            mixed[t * conv_dim + c] = acc * sigmoid(acc); // SiLU
        }
    }

    // Per-head gates. `g` is negative: it becomes a decay through exp().
    let mut recurrent = vec![0f32; dims.num_v_heads * kd * vd];
    let mut context = vec![0f32; seq * value_dim];
    for t in 0..seq {
        let x = &hidden[t * h..(t + 1) * h];
        let a = linear(x, w.in_proj_a, dims.num_v_heads, h);
        let b = linear(x, w.in_proj_b, dims.num_v_heads, h);
        let row = &mixed[t * conv_dim..(t + 1) * conv_dim];
        let (q_all, rest) = row.split_at(key_dim);
        let (k_all, v_all) = rest.split_at(key_dim);

        for head in 0..dims.num_v_heads {
            // q and k are shared across `group` value heads.
            let k_head = head / group;
            let mut q: Vec<f32> = q_all[k_head * kd..(k_head + 1) * kd].to_vec();
            let mut k: Vec<f32> = k_all[k_head * kd..(k_head + 1) * kd].to_vec();
            l2norm(&mut q, 1e-6);
            l2norm(&mut k, 1e-6);
            let scale = 1.0 / (kd as f32).sqrt();
            for value in &mut q {
                *value *= scale;
            }
            let v = &v_all[head * vd..(head + 1) * vd];

            let decay = (-w.a_log[head].exp() * softplus(a[head] + w.dt_bias[head])).exp();
            let beta = sigmoid(b[head]);
            let state = &mut recurrent[head * kd * vd..(head + 1) * kd * vd];
            let out = &mut context[t * value_dim + head * vd..t * value_dim + (head + 1) * vd];
            out.copy_from_slice(&gdn_delta_step(state, &q, &k, v, decay, beta));
        }
    }

    // Gated RMS norm per value head, then the output projection.
    let mut out = vec![0f32; seq * h];
    for t in 0..seq {
        let z = linear(&hidden[t * h..(t + 1) * h], w.in_proj_z, value_dim, h);
        let mut gated = vec![0f32; value_dim];
        for head in 0..dims.num_v_heads {
            let span = head * vd..(head + 1) * vd;
            let x = &context[t * value_dim + span.start..t * value_dim + span.end];
            let mean_square = x.iter().map(|v| v * v).sum::<f32>() / vd as f32;
            let inv = 1.0 / (mean_square + dims.eps).sqrt();
            for (i, slot) in gated[span.clone()].iter_mut().enumerate() {
                let g = z[span.start + i];
                let activated = if dims.sigmoid_gate {
                    sigmoid(g)
                } else {
                    g * sigmoid(g)
                };
                // Plain multiplier, not (1 + w).
                *slot = x[i] * inv * w.norm[i] * activated;
            }
        }
        out[t * h..(t + 1) * h].copy_from_slice(&linear(&gated, w.out_proj, h, value_dim));
    }
    out
}
