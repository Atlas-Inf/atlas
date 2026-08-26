// SPDX-License-Identifier: AGPL-3.0-only

//! Check a checkpoint on disk against its manifest, before loading a byte of it.
//!
//! This is the loader's first move. A checkpoint that is missing a tensor, or
//! carries one at the wrong shape, should be refused by name here rather than
//! discovered halfway through a 135 GB load — or worse, not discovered, because
//! a wrong-but-plausible shape reads real numbers from the wrong place.
//!
//! Header parsing goes through [`crate::safetensors::tensor_span`], so a
//! crafted or truncated file is rejected on the same terms as the loading path
//! rather than on looser ones.

use super::{ExpectedTensor, ManifestDiff, Qwen4ExpLayout, diff, manifest_for_with};
use crate::config::ModelConfig;
use anyhow::{Context, Result, bail};
use std::collections::BTreeMap;
use std::path::Path;

/// Largest safetensors header we will parse. Real ones run to a few MB; this is
/// third-party data, and an unbounded length prefix is an allocation primitive.
const MAX_HEADER_BYTES: u64 = 256 * 1024 * 1024;

/// Tensor names and shapes read from one shard's header.
pub fn read_shard_header(path: &Path) -> Result<BTreeMap<String, Vec<usize>>> {
    use std::io::{Read, Seek, SeekFrom};

    let mut file =
        std::fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let file_len = file.metadata()?.len();

    let mut prefix = [0u8; 8];
    file.read_exact(&mut prefix)
        .with_context(|| format!("{}: too short for a safetensors header", path.display()))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES || header_len.saturating_add(8) > file_len {
        bail!(
            "{}: header length {header_len} is implausible for a {file_len}-byte file",
            path.display()
        );
    }

    let mut raw = vec![0u8; header_len as usize];
    file.seek(SeekFrom::Start(8))?;
    file.read_exact(&mut raw)
        .with_context(|| format!("{}: truncated header", path.display()))?;
    let header: serde_json::Value = serde_json::from_slice(&raw)
        .with_context(|| format!("{}: header is not JSON", path.display()))?;
    let entries = header
        .as_object()
        .with_context(|| format!("{}: header is not an object", path.display()))?;

    let data_start = 8 + header_len;
    let mut out = BTreeMap::new();
    for (name, meta) in entries {
        if name == "__metadata__" {
            continue;
        }
        // Validate the span even though we only want the shape: a header that
        // lies about offsets is not one whose shapes we should trust either.
        let offsets = meta
            .get("data_offsets")
            .with_context(|| format!("{}: tensor {name} has no data_offsets", path.display()))?;
        crate::safetensors::tensor_span(name, offsets, data_start, file_len)?;

        let shape = meta
            .get("shape")
            .and_then(|s| s.as_array())
            .with_context(|| format!("{}: tensor {name} has no shape", path.display()))?
            .iter()
            .map(|v| {
                v.as_u64()
                    .map(|n| n as usize)
                    .with_context(|| format!("tensor {name}: non-integer dimension"))
            })
            .collect::<Result<Vec<_>>>()?;
        out.insert(name.clone(), shape);
    }
    Ok(out)
}

/// Every tensor across every `*.safetensors` shard in `dir`.
pub fn read_checkpoint(dir: &Path) -> Result<BTreeMap<String, Vec<usize>>> {
    let mut shards: Vec<_> = std::fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
        .collect();
    shards.sort();
    if shards.is_empty() {
        bail!("{} holds no .safetensors shards", dir.display());
    }

    let mut out = BTreeMap::new();
    for shard in &shards {
        for (name, shape) in read_shard_header(shard)? {
            if let Some(previous) = out.insert(name.clone(), shape)
                && previous != out[&name]
            {
                bail!("tensor {name} appears in more than one shard with different shapes");
            }
        }
    }
    Ok(out)
}

/// Vision lives in its own config block and its own tower; the manifest does
/// not describe it, so it is filtered rather than reported as unexpected.
fn is_vision(name: &str) -> bool {
    name.starts_with("model.visual.")
}

/// Diff a checkpoint directory against the manifest for `config`.
///
/// Expert layout is a packaging choice rather than an architectural one — the
/// same model ships split per expert or stacked depending on whether a
/// quantizer has been through it — so both are tried and the closer fit is
/// reported, along with which one it was.
pub fn verify_checkpoint(
    dir: &Path,
    config: &ModelConfig,
) -> Result<(ManifestDiff, Qwen4ExpLayout)> {
    let actual = read_checkpoint(dir)?;
    let actual: Vec<(&str, &[usize])> = actual
        .iter()
        .filter(|(name, _)| !is_vision(name))
        .map(|(name, shape)| (name.as_str(), shape.as_slice()))
        .collect();

    let mut best: Option<(ManifestDiff, Qwen4ExpLayout)> = None;
    for layout in Qwen4ExpLayout::candidates() {
        let Some(base) = manifest_for_with(config, layout)? else {
            bail!("no manifest for model_type {}", config.model_type);
        };
        let expected: Vec<ExpectedTensor> = match super::quantized_manifest(config, &base)? {
            Some(quantized) => quantized,
            None => base,
        };
        let found = diff(&expected, actual.iter().copied());
        let better = best.as_ref().is_none_or(|(b, _)| found.count() < b.count());
        if better {
            let clean = found.is_clean();
            best = Some((found, layout));
            if clean {
                break;
            }
        }
    }
    Ok(best.expect("at least one candidate layout"))
}
