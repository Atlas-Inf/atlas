// SPDX-License-Identifier: AGPL-3.0-only

//! CPU reference for the `qwen4_exp` PLE tower.
//!
//! Not the serving path — this is the thing a GPU implementation gets checked
//! against. The PLE tower is the most novel of the layers this model needs, and
//! several steps in it are easy to get plausibly wrong:
//!
//! * its RMSNorm is **grouped** (each of `hc_count` streams normalises over its
//!   own `hidden_size` slice) and its weight is an **offset from 1**
//!   (`x * (1 + w)`), not a plain scale;
//! * the gate takes a **signed square root** — `sign(g) * sqrt(max(|g|, 1e-6))`;
//! * the depthwise conv is **dilated by `ngram_size`**, so its state is
//!   `(kernel - 1) * ngram_size` wide rather than `kernel - 1`.
//!
//! Checked against HuggingFace's own `Qwen4ExpTextPLELayer` at real weights.

/// Grouped RMS norm. `weight` is an offset from 1, applied across the full row.
pub fn grouped_rms_norm(x: &[f32], group: usize, weight: &[f32], eps: f32) -> Vec<f32> {
    assert_eq!(x.len() % group, 0, "row must divide into groups");
    assert_eq!(weight.len(), x.len(), "norm weight must match the row");
    let mut out = vec![0f32; x.len()];
    for (chunk_index, chunk) in x.chunks_exact(group).enumerate() {
        let mean_square = chunk.iter().map(|v| v * v).sum::<f32>() / group as f32;
        let scale = 1.0 / (mean_square + eps).sqrt();
        let base = chunk_index * group;
        for (offset, value) in chunk.iter().enumerate() {
            out[base + offset] = value * scale * (1.0 + weight[base + offset]);
        }
    }
    out
}

/// `y = W x` for a row-major `[out_dim, in_dim]` weight.
pub fn linear(x: &[f32], weight: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_dim);
    assert_eq!(weight.len(), out_dim * in_dim);
    (0..out_dim)
        .map(|row| {
            weight[row * in_dim..(row + 1) * in_dim]
                .iter()
                .zip(x)
                .map(|(w, v)| w * v)
                .sum()
        })
        .collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

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
