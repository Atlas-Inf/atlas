// SPDX-License-Identifier: AGPL-3.0-only

//! CPU references for the `qwen4_exp` layers that Atlas does not have yet.
//!
//! Not the serving path — these are what a GPU implementation gets checked
//! against. Each is checked here against HuggingFace's own module at real
//! weights, because each has steps that are easy to get plausibly wrong, and
//! "plausibly wrong" in a language model means fluent output that is subtly
//! not the model you loaded.
//!
//! In the PLE tower:
//!
//! * its RMSNorm is **grouped** (each of `hc_count` streams normalises over its
//!   own `hidden_size` slice) and its weight is an **offset from 1**
//!   (`x * (1 + w)`), not a plain scale;
//! * the gate takes a **signed square root** — `sign(g) * sqrt(max(|g|, 1e-6))`;
//! * the depthwise conv is **dilated by `ngram_size`**, so its state is
//!   `(kernel - 1) * ngram_size` wide rather than `kernel - 1`.
//!
//! In the hyper-connection block:
//!
//! * the mixing gate divides by `hc_count` BEFORE its activation, twice — once
//!   into the low-rank SiLU and again into the injection sigmoid;
//! * the block output is a **mean** across streams, not a sum or a concat, so
//!   it comes back `hidden` wide from a `hc_count * hidden` input;
//! * the injection weights are `2 * sigmoid(...)`, centred on 1 rather than
//!   0.5.

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

/// One hyper-connection block's weights. Widths are `hc_count * hidden` — the
/// residual is that many streams concatenated, not `hidden` with a gate.
pub struct HyperConnectionWeights<'a> {
    /// `[hc_count * hidden]`, an offset from 1
    pub hc_norm: &'a [f32],
    /// `[hc_lowrank, hc_count * hidden]`
    pub mix_down: &'a [f32],
    /// `[hc_count * hidden, hc_lowrank]`
    pub mix_up: &'a [f32],
    /// `[hc_count, hc_count * hidden]`. `None` on the trunk and MTP mixers,
    /// which mix without injecting.
    pub block_inject: Option<&'a [f32]>,
}

/// What a hyper-connection block hands the layer that follows it.
pub struct HyperConnectionOut {
    /// `[hidden]` — the mixed input the block actually computes on.
    pub mixed: Vec<f32>,
    /// `[hc_count]` — per-stream injection gains, centred on 1. Empty when the
    /// block has no `block_inject_weight`.
    pub injection: Vec<f32>,
}

/// One position through a hyper-connection block.
pub fn hyper_connection_forward(
    dims: &PleDims,
    w: &HyperConnectionWeights<'_>,
    lowrank: usize,
    hyper_input: &[f32],
) -> HyperConnectionOut {
    let (wide, hidden, hc) = (dims.wide(), dims.hidden, dims.hc_count);
    assert_eq!(
        hyper_input.len(),
        wide,
        "hyper input must be hc_count * hidden"
    );

    let normed = grouped_rms_norm(hyper_input, hidden, w.hc_norm, dims.eps);
    // Divided by hc_count BEFORE the activation, not after.
    let down: Vec<f32> = linear(&normed, w.mix_down, lowrank, wide)
        .into_iter()
        .map(|v| {
            let scaled = v / hc as f32;
            scaled * sigmoid(scaled)
        })
        .collect();
    let gate: Vec<f32> = linear(&down, w.mix_up, wide, lowrank)
        .into_iter()
        .map(sigmoid)
        .collect();

    // Mean across streams: `hc_count * hidden` in, `hidden` out.
    let mut mixed = vec![0f32; hidden];
    for stream in 0..hc {
        for h in 0..hidden {
            let index = stream * hidden + h;
            mixed[h] += gate[index] * normed[index];
        }
    }
    for value in &mut mixed {
        *value /= hc as f32;
    }

    let injection = match w.block_inject {
        None => Vec::new(),
        // Centred on 1, not 0.5.
        Some(inject) => linear(&normed, inject, hc, wide)
            .into_iter()
            .map(|v| 2.0 * sigmoid(v / hc as f32))
            .collect(),
    };
    HyperConnectionOut { mixed, injection }
}
