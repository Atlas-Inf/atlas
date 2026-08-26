// SPDX-License-Identifier: AGPL-3.0-only

//! What a checkpoint must contain, derived from its config.
//!
//! A weight loader's first job is knowing which tensors to ask for and how
//! wide they are. Writing that down separately from the loading itself buys
//! two things a loader alone does not: it can be checked against a published
//! checkpoint's index without downloading the weights, and a shape that
//! disagrees fails here rather than at inference.
//!
//! That is not hypothetical. `qwen4_exp`'s `q_proj` is `[12288, 2560]` while
//! `num_attention_heads * head_dim` is 6144 — the projection carries an
//! interleaved gate. Read as ungated, a loader takes the gate half as query
//! values, halves the model's effective attention, and nothing fails.

use crate::config::ModelConfig;
use anyhow::Result;

/// One tensor a checkpoint is expected to carry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedTensor {
    pub name: String,
    pub shape: Vec<usize>,
}

impl ExpectedTensor {
    fn new(name: impl Into<String>, shape: impl Into<Vec<usize>>) -> Self {
        Self {
            name: name.into(),
            shape: shape.into(),
        }
    }
}

/// How a checkpoint's actual tensors compare against the manifest.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct ManifestDiff {
    /// Expected but absent.
    pub missing: Vec<String>,
    /// Present but not expected.
    pub unexpected: Vec<String>,
    /// Present with the wrong shape: `(name, expected, actual)`.
    pub mismatched: Vec<(String, Vec<usize>, Vec<usize>)>,
}

impl ManifestDiff {
    pub fn is_clean(&self) -> bool {
        self.missing.is_empty() && self.unexpected.is_empty() && self.mismatched.is_empty()
    }
}

/// Diff a manifest against what a checkpoint actually holds.
///
/// `actual` is `(name, shape)` pairs, typically read from a safetensors header
/// or index. Both sides are compared as sets, so ordering never matters.
pub fn diff<'a>(
    expected: &[ExpectedTensor],
    actual: impl IntoIterator<Item = (&'a str, &'a [usize])>,
) -> ManifestDiff {
    use std::collections::HashMap;
    let want: HashMap<&str, &[usize]> = expected
        .iter()
        .map(|t| (t.name.as_str(), t.shape.as_slice()))
        .collect();
    let mut seen = std::collections::HashSet::new();
    let mut out = ManifestDiff::default();

    for (name, shape) in actual {
        seen.insert(name);
        match want.get(name) {
            None => out.unexpected.push(name.to_string()),
            Some(w) if *w != shape => {
                out.mismatched
                    .push((name.to_string(), w.to_vec(), shape.to_vec()))
            }
            Some(_) => {}
        }
    }
    for t in expected {
        if !seen.contains(t.name.as_str()) {
            out.missing.push(t.name.clone());
        }
    }
    out.missing.sort();
    out.unexpected.sort();
    out.mismatched.sort();
    out
}

mod qwen4_exp;
pub use qwen4_exp::qwen4_exp_manifest;

/// The manifest for a config, dispatched on `model_type`.
///
/// `Ok(None)` means no manifest is written for that family yet — the ordinary
/// case, since this exists to support new ports rather than to re-describe
/// loaders that already work.
pub fn manifest_for(config: &ModelConfig) -> Result<Option<Vec<ExpectedTensor>>> {
    match config.model_type.as_str() {
        "qwen4_exp" => Ok(Some(qwen4_exp_manifest(config)?)),
        _ => Ok(None),
    }
}
