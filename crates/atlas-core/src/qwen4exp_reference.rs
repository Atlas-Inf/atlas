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

/// Rows below which threading costs more than it saves.
const PARALLEL_ROW_THRESHOLD: usize = 512;

/// `y = W x` for a row-major `[out_dim, in_dim]` weight.
///
/// Parallel over output rows above a threshold. This is a reference, not a
/// kernel — the point is that a 180 B-parameter model can be checked end to end
/// in minutes rather than an hour, not that it competes with a GPU.
pub fn linear(x: &[f32], weight: &[f32], out_dim: usize, in_dim: usize) -> Vec<f32> {
    assert_eq!(x.len(), in_dim);
    assert_eq!(weight.len(), out_dim * in_dim);

    let row = |r: usize| -> f32 {
        weight[r * in_dim..(r + 1) * in_dim]
            .iter()
            .zip(x)
            .map(|(w, v)| w * v)
            .sum()
    };

    if out_dim < PARALLEL_ROW_THRESHOLD {
        return (0..out_dim).map(row).collect();
    }

    let threads = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(out_dim);
    if threads <= 1 {
        return (0..out_dim).map(row).collect();
    }

    let mut out = vec![0f32; out_dim];
    let chunk = out_dim.div_ceil(threads);
    std::thread::scope(|scope| {
        for (index, slice) in out.chunks_mut(chunk).enumerate() {
            let base = index * chunk;
            scope.spawn(move || {
                for (offset, slot) in slice.iter_mut().enumerate() {
                    *slot = weight[(base + offset) * in_dim..(base + offset + 1) * in_dim]
                        .iter()
                        .zip(x)
                        .map(|(w, v)| w * v)
                        .sum();
                }
            });
        }
    });
    out
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

/// Standard RoPE tables, `[seq, rotary_dim]` each.
///
/// MRoPE collapses to this for text: the three grids carry identical position
/// ids, so `apply_interleaved_mrope` copies each section onto itself. Only a
/// multimodal prompt makes the grids differ.
pub fn rope_tables(seq: usize, rotary_dim: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = rotary_dim / 2;
    let mut cos = vec![0f32; seq * rotary_dim];
    let mut sin = vec![0f32; seq * rotary_dim];
    for t in 0..seq {
        for i in 0..half {
            let freq = t as f32 / theta.powf(2.0 * i as f32 / rotary_dim as f32);
            // emb = cat(freqs, freqs), so each frequency appears twice.
            for slot in [i, i + half] {
                cos[t * rotary_dim + slot] = freq.cos();
                sin[t * rotary_dim + slot] = freq.sin();
            }
        }
    }
    (cos, sin)
}

/// Scatter a block's `hidden`-wide output back across the `hc_count` residual
/// streams, scaled per stream.
pub fn broadcast_inject(mixed: &[f32], injection: &[f32], hidden: usize) -> Vec<f32> {
    let mut out = vec![0f32; injection.len() * hidden];
    for (stream, gain) in injection.iter().enumerate() {
        for (slot, value) in out[stream * hidden..(stream + 1) * hidden]
            .iter_mut()
            .zip(mixed)
        {
            *slot = value * gain;
        }
    }
    out
}

/// One gated-delta-net recurrence step for one head.
///
/// `q` must already be L2-normalised AND scaled by `1/sqrt(key_head_dim)`;
/// `k` L2-normalised. `decay` is `exp(g_t)`, `beta` is `sigmoid(b_t)`. The
/// state is `[key_head_dim, value_head_dim]` with the value dim contiguous,
/// which is the layout Atlas's `gated_delta_rule_decode` kernel expects. Both
/// dims are read off `k` and `v` rather than passed in.
///
/// The order matters and is not the obvious one: decay the state FIRST, then
/// measure how wrong its recall of `v` is, then correct by that error scaled by
/// beta. Accumulating `k v^T` instead -- the natural reading of "linear
/// attention" -- gives a model that still produces text.
pub fn gdn_delta_step(
    state: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    decay: f32,
    beta: f32,
) -> Vec<f32> {
    // Both dims are already pinned by the slices, so taking them as parameters
    // only creates a way for a caller to disagree with its own data.
    let key_dim = k.len();
    let value_dim = v.len();
    assert_eq!(q.len(), key_dim, "q and k are both key_head_dim");
    assert_eq!(state.len(), key_dim * value_dim);
    for value in state.iter_mut() {
        *value *= decay;
    }
    let mut recall = vec![0f32; value_dim];
    for (ki, kv) in k.iter().enumerate() {
        for (vi, slot) in recall.iter_mut().enumerate() {
            *slot += state[ki * value_dim + vi] * kv;
        }
    }
    let delta: Vec<f32> = v
        .iter()
        .zip(&recall)
        .map(|(target, got)| (target - got) * beta)
        .collect();
    for (ki, kv) in k.iter().enumerate() {
        for (vi, d) in delta.iter().enumerate() {
            state[ki * value_dim + vi] += kv * d;
        }
    }
    let mut out = vec![0f32; value_dim];
    for (ki, qv) in q.iter().enumerate() {
        for (vi, slot) in out.iter_mut().enumerate() {
            *slot += state[ki * value_dim + vi] * qv;
        }
    }
    out
}

// Each block lives in its own file: the repo caps a .rs at 500 LoC and this
// module was 783. The split is BY BLOCK, which is also how the port is
// verified — one oracle per block, each checked against HuggingFace at real
// weights — so a reader looking for the PLE tower opens ple.rs instead of
// scrolling past the MoE.
#[path = "qwen4exp_reference/attn.rs"]
mod attn;
#[path = "qwen4exp_reference/gdn.rs"]
mod gdn;
#[path = "qwen4exp_reference/hc.rs"]
mod hc;
#[path = "qwen4exp_reference/moe.rs"]
mod moe;
#[path = "qwen4exp_reference/ple.rs"]
mod ple;

// Re-exported flat, so `atlas_core::qwen4exp_reference::ple_forward` still
// resolves for every caller — the examples, the microtest and the GPU parity
// gates all name these paths.
pub use attn::*;
pub use gdn::*;
pub use hc::*;
pub use moe::*;
pub use ple::*;
