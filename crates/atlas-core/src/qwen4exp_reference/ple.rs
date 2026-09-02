// SPDX-License-Identifier: AGPL-3.0-only

//! PLE tower reference — the gate, the dilated depthwise conv, and their dims.

use super::{grouped_rms_norm, linear, sigmoid, silu};

/// Everything the tower needs that is not an activation.
pub struct PleWeights<'a> {
    /// `[hc_count * hidden, ple_embed_dim]`
    pub key_proj: &'a [f32],
    /// `[hidden, ple_embed_dim]`
    pub value_proj: &'a [f32],
    /// `[hc_count * hidden]`, each an offset from 1
    pub norm_key: &'a [f32],
    pub norm_query: &'a [f32],
    pub norm_conv: &'a [f32],
    /// `[hc_count * hidden, kernel]`, depthwise
    pub conv1d: &'a [f32],
}

pub struct PleDims {
    pub hidden: usize,
    pub hc_count: usize,
    pub ple_embed_dim: usize,
    pub kernel: usize,
    /// Conv dilation, which the reference sets to `ngram_size`.
    pub dilation: usize,
    pub eps: f32,
}

impl PleDims {
    pub fn wide(&self) -> usize {
        self.hidden * self.hc_count
    }
    /// Left context the dilated conv needs: `(kernel - 1) * dilation`.
    pub fn conv_state_len(&self) -> usize {
        (self.kernel - 1) * self.dilation
    }
}

/// One PLE forward over a whole sequence, starting from a fresh conv state.
///
/// `embeddings` is `[seq, ple_embed_dim]` — the gathered, dequantized n-gram
/// rows. `hidden_states` is `[seq, hc_count * hidden]`. Returns the same shape
/// as `hidden_states`.
pub fn ple_forward(
    dims: &PleDims,
    w: &PleWeights<'_>,
    embeddings: &[f32],
    hidden_states: &[f32],
) -> Vec<f32> {
    let (wide, hidden, e) = (dims.wide(), dims.hidden, dims.ple_embed_dim);
    let seq = embeddings.len() / e;
    assert_eq!(hidden_states.len(), seq * wide, "hidden_states shape");

    // Per position: project, norm, gate, scale the shared value.
    let mut gated = vec![0f32; seq * wide];
    for t in 0..seq {
        let emb = &embeddings[t * e..(t + 1) * e];
        let key = linear(emb, w.key_proj, wide, e);
        let key_normed = grouped_rms_norm(&key, hidden, w.norm_key, dims.eps);
        let value = linear(emb, w.value_proj, hidden, e);
        let query_normed = grouped_rms_norm(
            &hidden_states[t * wide..(t + 1) * wide],
            hidden,
            w.norm_query,
            dims.eps,
        );

        for stream in 0..dims.hc_count {
            let span = stream * hidden..(stream + 1) * hidden;
            let dot: f32 = key_normed[span.clone()]
                .iter()
                .zip(&query_normed[span.clone()])
                .map(|(k, q)| k * q)
                .sum::<f32>()
                / (hidden as f32).sqrt();
            // Signed square root, floored so the gradient at zero is finite.
            let gate = dot.abs().max(1e-6).sqrt() * if dot < 0.0 { -1.0 } else { 1.0 };
            let gate = sigmoid(gate);
            for h in 0..hidden {
                gated[t * wide + stream * hidden + h] = gate * value[h];
            }
        }
    }

    // The conv runs on a normalised copy; the residual adds the un-normalised one.
    let mut normed = Vec::with_capacity(seq * wide);
    for t in 0..seq {
        normed.extend(grouped_rms_norm(
            &gated[t * wide..(t + 1) * wide],
            hidden,
            w.norm_conv,
            dims.eps,
        ));
    }

    // Depthwise, dilated, causal: zero-padded on the left by (kernel-1)*dilation.
    let mut out = gated;
    for t in 0..seq {
        for channel in 0..wide {
            let mut acc = 0f32;
            for tap in 0..dims.kernel {
                let back = (dims.kernel - 1 - tap) * dims.dilation;
                if let Some(source) = t.checked_sub(back) {
                    acc += w.conv1d[channel * dims.kernel + tap] * normed[source * wide + channel];
                }
            }
            out[t * wide + channel] += silu(acc);
        }
    }
    out
}
