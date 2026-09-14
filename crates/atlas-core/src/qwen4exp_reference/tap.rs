// SPDX-License-Identifier: AGPL-3.0-only

//! Reference-side highway taps, written in the SAME layout the GPU writes.
//!
//! `spark-model`'s `layers::ple::dump` taps the `hc_count`-wide residual
//! highway per sublayer under `ATLAS_QWEN4EXP_DUMP` and writes
//! `L{layer:02}_{tag}.bin` as raw little-endian FP32. That side has existed
//! since the bisect that found the GDN gated norm gates with sigmoid; what has
//! been missing is a reference to diff it against WITHOUT Python, because the
//! golden generators import `torch` and a GB10 dev box need not have it.
//!
//! This writes the reference's own taps into a second directory under
//! `ATLAS_QWEN4EXP_REF_DUMP`, so the comparison is Rust against Rust:
//!
//! ```text
//! # GPU: one request, one prefill
//! ATLAS_QWEN4EXP_DUMP=/tmp/gpu   spark serve ...
//! # CPU: the same token ids
//! ATLAS_QWEN4EXP_REF_DUMP=/tmp/ref  cargo run --example qwen4exp_forward -- <ckpt>
//! python3 scripts/dev/qwen4exp_tap_diff.py /tmp/gpu /tmp/ref
//! ```
//!
//! The GPU writes BF16 taps as `L{n}_{tag}.bf16.bin`; this writes everything
//! as FP32 and the diff tool widens the BF16 side, since the point is to find
//! the first sublayer that DISAGREES, not to match rounding.
//!
//! One-shot per tag like the GPU side, and for the same reason: the GPU's SSM
//! layer counter never resets, so only the first prefill of a run is labelled
//! consistently. Send one request.

/// Directory from `ATLAS_QWEN4EXP_REF_DUMP`, resolved once.
fn dir() -> Option<&'static str> {
    static DIR: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    DIR.get_or_init(|| {
        let d = std::env::var("ATLAS_QWEN4EXP_REF_DUMP")
            .ok()
            .filter(|s| !s.is_empty());
        if let Some(ref path) = d {
            let _ = std::fs::create_dir_all(path);
            eprintln!("ATLAS_QWEN4EXP_REF_DUMP={path}: taping the reference highway");
        }
        d
    })
    .as_deref()
}

/// Write one FP32 tap, refusing to overwrite an existing file.
///
/// `layer` must be the index the GPU would use for the SAME point, which for
/// the SSM sublayers is the count of GDN layers seen so far and NOT the model
/// layer index — 36 of the 48 layers are GDN, so the two diverge from layer 3
/// onwards. `ssm_index_of` does that mapping.
pub fn tap(layer: usize, tag: &str, values: &[f32]) {
    let Some(dir) = dir() else {
        return;
    };
    let path = format!("{dir}/L{layer:02}_{tag}.bin");
    if std::path::Path::new(&path).exists() {
        return;
    }
    let mut raw = Vec::with_capacity(values.len() * 4);
    for v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&path, &raw) {
        eprintln!("reference tap {path}: {e}");
    }
}

/// Dump the reference's final logits, for a KL comparison against the GPU's.
///
/// The GPU appends FP32 rows of `vocab_size` to `logits_fetch.bin` under
/// `ATLAS_DUMP_LOGITS_PATH` (see `spark_runtime::sampler`); this writes the
/// same layout to `ref_logits.bin` so the two can be compared directly.
///
/// Logits are the right place to measure drift: cosine on a hidden state says
/// how similar two vectors are, while KL over the softmax says how differently
/// the two models would SAMPLE — which is what actually reaches a user. A
/// hidden-state cosine of 0.96 can be harmless or fatal depending on how it
/// lands on the vocabulary, and only this tells you which.
pub fn tap_logits(rows: &[f32], vocab: usize) {
    let Some(dir) = dir() else {
        return;
    };
    let path = format!("{dir}/ref_logits.bin");
    let mut raw = Vec::with_capacity(rows.len() * 4);
    for v in rows {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    if let Err(e) = std::fs::write(&path, &raw) {
        eprintln!("reference logit tap {path}: {e}");
        return;
    }
    eprintln!(
        "reference logits: {} rows x {vocab} -> {path}",
        rows.len() / vocab.max(1)
    );
}

/// Whether any reference tap is enabled — lets a caller skip the work of
/// materializing a tap's values when nothing will be written.
pub fn enabled() -> bool {
    dir().is_some()
}

/// The GPU's SSM-layer index for model layer `layer`, given the layer types.
///
/// The GPU labels its SSM taps with a counter over LINEAR-ATTENTION layers
/// only, so model layer 4 (the second GDN layer after a full-attention one) is
/// `L03` there. Getting this wrong compares two different layers and reports a
/// divergence that is really an off-by-one.
pub fn ssm_index_of(layer_types: &[crate::config::LayerType], layer: usize) -> Option<usize> {
    if layer_types.get(layer)? != &crate::config::LayerType::LinearAttention {
        return None;
    }
    Some(
        layer_types[..layer]
            .iter()
            .filter(|t| **t == crate::config::LayerType::LinearAttention)
            .count(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::LayerType::{FullAttention, LinearAttention};

    /// The mapping the GPU's counter implies, on the real 3:1 interleave.
    #[test]
    fn ssm_index_counts_only_linear_layers() {
        let types = [
            LinearAttention,
            LinearAttention,
            LinearAttention,
            FullAttention,
            LinearAttention,
        ];
        assert_eq!(ssm_index_of(&types, 0), Some(0));
        assert_eq!(ssm_index_of(&types, 2), Some(2));
        // The full-attention layer has no SSM tap at all.
        assert_eq!(ssm_index_of(&types, 3), None);
        // ...and the next GDN layer is L03, NOT L04. This is the off-by-one
        // that would make a tap diff blame the wrong sublayer.
        assert_eq!(ssm_index_of(&types, 4), Some(3));
    }
}
