// SPDX-License-Identifier: AGPL-3.0-only

//! Gated-Q full-attention reference, and the decode step extracted from it.

use super::{grouped_rms_norm, linear, sigmoid};

/// Full-attention weights for one layer. The indexer is absent on purpose:
/// it is a no-op below `indexer_budget + compress_ratio - 1` tokens of
/// context, which is where a first bring-up lives.
pub struct AttnWeights<'a> {
    /// `[2 * num_heads * head_dim, hidden]` — query and its gate, interleaved
    /// per head.
    pub q_proj: &'a [f32],
    /// `[num_kv_heads * head_dim, hidden]`
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    /// `[hidden, num_heads * head_dim]` — consumes only the query half.
    pub o_proj: &'a [f32],
    /// `[head_dim]`, offsets from 1, applied per head.
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

pub struct AttnDims {
    pub hidden: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    /// `head_dim * partial_rotary_factor`. Only this prefix of each head is
    /// rotated; the rest carries through untouched.
    pub rotary_dim: usize,
    pub eps: f32,
}

/// Rotate the first `rotary_dim` of one head in place, rotate-half style.
fn apply_rope(head: &mut [f32], cos: &[f32], sin: &[f32]) {
    let rotary = cos.len();
    let half = rotary / 2;
    let original: Vec<f32> = head[..rotary].to_vec();
    for i in 0..rotary {
        // rotate_half: [-x2, x1]
        let rotated = if i < half {
            -original[i + half]
        } else {
            original[i - half]
        };
        head[i] = original[i] * cos[i] + rotated * sin[i];
    }
}

/// Causal multi-head attention over a whole sequence.
///
/// `cos` and `sin` are `[seq, rotary_dim]` — supplied rather than derived, so
/// this stays independent of how MRoPE builds them. `hidden` is
/// `[seq, hidden]`; the result is the same shape.
pub fn attention_forward(
    dims: &AttnDims,
    w: &AttnWeights<'_>,
    hidden: &[f32],
    cos: &[f32],
    sin: &[f32],
) -> Vec<f32> {
    let (h, hd, rot) = (dims.hidden, dims.head_dim, dims.rotary_dim);
    let seq = hidden.len() / h;
    let q_dim = dims.num_heads * hd;
    let kv_dim = dims.num_kv_heads * hd;

    let mut queries = vec![0f32; seq * q_dim];
    let mut gates = vec![0f32; seq * q_dim];
    let mut keys = vec![0f32; seq * kv_dim];
    let mut values = vec![0f32; seq * kv_dim];

    for t in 0..seq {
        let x = &hidden[t * h..(t + 1) * h];
        let (c, s) = (&cos[t * rot..(t + 1) * rot], &sin[t * rot..(t + 1) * rot]);

        // q_proj emits [query | gate] per head, hence the 2x head stride.
        let fused = linear(x, w.q_proj, q_dim * 2, h);
        for head in 0..dims.num_heads {
            let src = head * hd * 2;
            let dst = t * q_dim + head * hd;
            let mut q = grouped_rms_norm(&fused[src..src + hd], hd, w.q_norm, dims.eps);
            apply_rope(&mut q, c, s);
            queries[dst..dst + hd].copy_from_slice(&q);
            gates[dst..dst + hd].copy_from_slice(&fused[src + hd..src + hd * 2]);
        }

        let k = linear(x, w.k_proj, kv_dim, h);
        for head in 0..dims.num_kv_heads {
            let src = head * hd;
            let dst = t * kv_dim + head * hd;
            let mut kh = grouped_rms_norm(&k[src..src + hd], hd, w.k_norm, dims.eps);
            apply_rope(&mut kh, c, s);
            keys[dst..dst + hd].copy_from_slice(&kh);
        }
        values[t * kv_dim..(t + 1) * kv_dim].copy_from_slice(&linear(x, w.v_proj, kv_dim, h));
    }

    let mut out = vec![0f32; seq * h];
    for t in 0..seq {
        let context = attention_decode_step(
            dims,
            &queries[t * q_dim..(t + 1) * q_dim],
            &gates[t * q_dim..(t + 1) * q_dim],
            &keys[..(t + 1) * kv_dim],
            &values[..(t + 1) * kv_dim],
        );
        out[t * h..(t + 1) * h].copy_from_slice(&linear(&context, w.o_proj, h, q_dim));
    }
    out
}

/// The attention arithmetic for ONE query position, up to but not including
/// `o_proj`: causal softmax over the K/V written so far, then the elementwise
/// sigmoid gate.
///
/// Split out of `attention_forward` so the GPU decode kernel is checked against
/// the same code that matches HuggingFace at 8.0e-7, rather than against a
/// second transcription of the same equations. `keys`/`values` are
/// `[past_len, num_kv_heads * head_dim]` and their length is what sets the
/// causal window -- the caller passes exactly the positions this query may see.
///
/// `query` must already be normed and rotated; `gate` is the raw pre-sigmoid
/// half of `q_proj`'s per-head `[query | gate]` pair.
pub fn attention_decode_step(
    dims: &AttnDims,
    query: &[f32],
    gate: &[f32],
    keys: &[f32],
    values: &[f32],
) -> Vec<f32> {
    let hd = dims.head_dim;
    let q_dim = dims.num_heads * hd;
    let kv_dim = dims.num_kv_heads * hd;
    let group = dims.num_heads / dims.num_kv_heads;
    let scale = 1.0 / (hd as f32).sqrt();
    let past = keys.len() / kv_dim;
    assert_eq!(query.len(), q_dim, "query is num_heads * head_dim");
    assert_eq!(gate.len(), q_dim, "gate is num_heads * head_dim");
    assert_eq!(
        values.len(),
        past * kv_dim,
        "keys and values agree on length"
    );
    assert!(past > 0, "a query attends to at least its own position");

    let mut context = vec![0f32; q_dim];
    for head in 0..dims.num_heads {
        let kv_head = head / group;
        let q = &query[head * hd..(head + 1) * hd];

        let mut scores = Vec::with_capacity(past);
        for p in 0..past {
            let k = &keys[p * kv_dim + kv_head * hd..p * kv_dim + (kv_head + 1) * hd];
            scores.push(q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale);
        }
        let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0f32;
        for score in &mut scores {
            *score = (*score - max).exp();
            total += *score;
        }
        for (p, weight) in scores.iter().enumerate() {
            let v = &values[p * kv_dim + kv_head * hd..p * kv_dim + (kv_head + 1) * hd];
            let slot = &mut context[head * hd..(head + 1) * hd];
            for (c, value) in slot.iter_mut().zip(v) {
                *c += weight / total * value;
            }
        }
    }
    // Elementwise sigmoid gate, applied before the output projection.
    for (c, g) in context.iter_mut().zip(gate) {
        *c *= sigmoid(*g);
    }
    context
}
