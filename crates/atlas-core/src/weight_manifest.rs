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

    /// Total discrepancies, for picking the closer of two candidate layouts.
    pub fn count(&self) -> usize {
        self.missing.len() + self.unexpected.len() + self.mismatched.len()
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

/// Does an HF-style module glob match a module path?
///
/// ModelOpt writes patterns like `*.self_attn.*`, `mtp.*` and `*hyper_connection*`,
/// where `*` spans dots — `*.self_attn.*` is meant to match
/// `model.language_model.layers.3.self_attn.q_proj`. HF's own native-FP8 lists
/// carry no globs at all (Qwen3.8-Flash-Next-FP8 spells out all 943 modules),
/// so both forms have to work.
pub fn module_glob_matches(pattern: &str, module: &str) -> bool {
    let mut parts = pattern.split('*');
    let Some(first) = parts.next() else {
        return pattern == module;
    };
    if !module.starts_with(first) {
        return false;
    }
    if !pattern.contains('*') {
        return pattern == module;
    }
    let mut rest = &module[first.len()..];
    let mut trailing = "";
    for part in parts {
        trailing = part;
        if part.is_empty() {
            continue;
        }
        match rest.find(part) {
            Some(at) => rest = &rest[at + part.len()..],
            None => return false,
        }
    }
    // A pattern not ending in `*` must consume to the end.
    trailing.is_empty() || rest.is_empty()
}

/// Scale siblings a quantized checkpoint carries alongside the logical weights.
///
/// Separate from the base manifest on purpose: the same architecture ships in
/// several quantizations (Qwen3.8-Flash-Next has an FP8 release and at least two
/// NVFP4 repacks) and they differ only here. Keeping the split means a repack
/// does not fork the architecture description.
///
/// Only block-scaled FP8 (`weight_scale_inv`) is implemented. ModelOpt NVFP4
/// uses a different sibling set (`weight_scale`, `weight_scale_2`,
/// `input_scale`) and returns `Ok(None)` rather than a wrong guess.
pub fn quantization_siblings(
    config: &ModelConfig,
    manifest: &[ExpectedTensor],
) -> Result<Option<Vec<ExpectedTensor>>> {
    let Some(quant) = config.quantization_config.as_ref() else {
        return Ok(None);
    };
    if quant.quant_method != "fp8" || quant.weight_block_size.len() != 2 {
        return Ok(None);
    }
    let (br, bc) = (quant.weight_block_size[0], quant.weight_block_size[1]);
    anyhow::ensure!(br > 0 && bc > 0, "weight_block_size must be positive");

    let is_ignored = |module: &str| {
        quant
            .ignore_modules
            .iter()
            .any(|p| module_glob_matches(p, module))
    };
    // A group carrying its own `weight_scale` is quantized PER TENSOR, not per
    // block, and takes no `weight_scale_inv`. The n-gram embedding is the case
    // that matters: its 128 shards are FP8 but share one BF16 scale, and they
    // are absent from `modules_to_not_convert` because they are converted --
    // just by a different scheme. Keyed off the manifest rather than off the
    // name, so any future per-tensor group is handled for free.
    let per_tensor: std::collections::HashSet<&str> = manifest
        .iter()
        .filter_map(|t| t.name.strip_suffix(".weight_scale"))
        .collect();

    Ok(Some(
        manifest
            .iter()
            .filter_map(|tensor| {
                // Only 2-D linear weights are block-quantized. Norms, biases,
                // integer buffers and the 3-D conv kernels are not, and none of
                // them appear in the ignore list either -- which is why the
                // rank check has to be here and not left to the list.
                let module = tensor.name.strip_suffix(".weight")?;
                if tensor.shape.len() != 2 || is_ignored(module) {
                    return None;
                }
                // `a.b.shard_0` -> `a.b`; skip when that group scales per tensor.
                if let Some((group, _)) = module.rsplit_once('.')
                    && per_tensor.contains(group)
                {
                    return None;
                }
                Some(ExpectedTensor::new(
                    format!("{module}.weight_scale_inv"),
                    [tensor.shape[0].div_ceil(br), tensor.shape[1].div_ceil(bc)],
                ))
            })
            .collect(),
    ))
}

/// The full on-disk tensor set for a quantized release: logical weights with
/// their storage shapes, plus every scale sibling.
///
/// Distinct from [`quantization_siblings`], which only adds. NVFP4 also
/// *rewrites* the weight it quantizes — a `[2560, 640]` projection is stored as
/// U8 `[2560, 320]`, two FP4 values per byte — so a caller that only appended
/// siblings would still expect the wrong shape for the weight itself.
///
/// `Ok(None)` for an unquantized checkpoint or a scheme not described here.
pub fn quantized_manifest(
    config: &ModelConfig,
    base: &[ExpectedTensor],
) -> Result<Option<Vec<ExpectedTensor>>> {
    let Some(quant) = config.quantization_config.as_ref() else {
        return Ok(None);
    };

    // FP8 block-scaled: weights keep their shape, one scale per tile.
    if quant.quant_method == "fp8" && quant.weight_block_size.len() == 2 {
        let Some(siblings) = quantization_siblings(config, base)? else {
            return Ok(None);
        };
        let mut out = base.to_vec();
        out.extend(siblings);
        return Ok(Some(out));
    }

    // ModelOpt NVFP4: packed weights plus a three-tensor scale set.
    if quant.quant_algo == "NVFP4" {
        let group = quant.group_size;
        anyhow::ensure!(group > 0, "NVFP4 requires a non-zero group_size");
        let is_ignored = |module: &str| {
            quant
                .ignore_modules
                .iter()
                .any(|p| module_glob_matches(p, module))
        };
        let per_tensor: std::collections::HashSet<&str> = base
            .iter()
            .filter_map(|t| t.name.strip_suffix(".weight_scale"))
            .collect();

        let mut out = Vec::with_capacity(base.len() * 2);
        for tensor in base {
            let quantizable = tensor
                .name
                .strip_suffix(".weight")
                .filter(|module| tensor.shape.len() == 2 && !is_ignored(module))
                .filter(|module| {
                    !module
                        .rsplit_once('.')
                        .is_some_and(|(group, _)| per_tensor.contains(group))
                });
            let Some(module) = quantizable else {
                out.push(tensor.clone());
                continue;
            };
            let (rows, cols) = (tensor.shape[0], tensor.shape[1]);
            anyhow::ensure!(
                cols.is_multiple_of(2) && cols.is_multiple_of(group),
                "NVFP4 needs an even, group-aligned input dim; {} has {cols}",
                tensor.name
            );
            // Two FP4 values per byte along the input dim.
            out.push(ExpectedTensor::new(tensor.name.clone(), [rows, cols / 2]));
            out.push(ExpectedTensor::new(
                format!("{module}.weight_scale"),
                [rows, cols / group],
            ));
            // Both are scalars: a per-tensor second-level scale and the
            // activation scale.
            out.push(ExpectedTensor::new(format!("{module}.weight_scale_2"), []));
            out.push(ExpectedTensor::new(format!("{module}.input_scale"), []));
        }
        return Ok(Some(out));
    }

    Ok(None)
}

mod preflight;
mod qwen4_exp;
pub use preflight::{read_checkpoint, read_shard_header, verify_checkpoint};
pub use qwen4_exp::{ExpertLayout, Qwen4ExpLayout, qwen4_exp_manifest, qwen4_exp_manifest_with};

/// The manifest for a config, dispatched on `model_type`.
///
/// `Ok(None)` means no manifest is written for that family yet — the ordinary
/// case, since this exists to support new ports rather than to re-describe
/// loaders that already work.
pub fn manifest_for(config: &ModelConfig) -> Result<Option<Vec<ExpectedTensor>>> {
    manifest_for_with(config, Qwen4ExpLayout::default())
}

/// [`manifest_for`] with per-release layout overrides.
pub fn manifest_for_with(
    config: &ModelConfig,
    layout: Qwen4ExpLayout,
) -> Result<Option<Vec<ExpectedTensor>>> {
    match config.model_type.as_str() {
        "qwen4_exp" => Ok(Some(qwen4_exp_manifest_with(config, layout)?)),
        _ => Ok(None),
    }
}
