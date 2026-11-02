// SPDX-License-Identifier: AGPL-3.0-only

//! EXL3 ships tensors in `*.safetensors` files that `model.safetensors.index.json`
//! does not list: `ngram_embedding.safetensors` (the PLE n-gram table, every
//! build), `vision_k6.safetensors` (vision linears, 4.05 bpw) and
//! `mtp_hyper_connection_mixer_patch.safetensors` (3.05 bpw). exllamav3 loads
//! every safetensor file in the directory; Atlas reads only the indexed ones and
//! filters each shard through the `weight_map`, so those tensors would silently
//! be absent from the store. Merging them into the map here keeps every
//! downstream consumer (per-shard filter, pre-flight estimate, n-gram deferral)
//! unchanged.

use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Adds `tensor -> file` entries for every tensor in an unindexed `*.safetensors`
/// file of an EXL3 checkpoint.
/// Returns the side files merged (for the log). Indexed names always win;
/// `extra_weights.safetensors` is left to its own existing path. No-op unless
/// `model_dir/config.json` declares `quantization_config.quant_method == "exl3"`
/// (case-insensitive).
pub(crate) fn merge_exl3_side_files(
    model_dir: &Path,
    weight_map: &mut HashMap<String, String>,
) -> Result<Vec<String>> {
    if !is_exl3_checkpoint(model_dir)? {
        return Ok(Vec::new());
    }
    let indexed: HashSet<String> = weight_map.values().cloned().collect();
    let mut side_files: Vec<std::path::PathBuf> = Vec::new();
    for entry in
        std::fs::read_dir(model_dir).with_context(|| format!("read {}", model_dir.display()))?
    {
        let path = entry?.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        if !name.ends_with(".safetensors") || name == "extra_weights.safetensors" {
            continue;
        }
        if indexed.contains(name) {
            continue;
        }
        side_files.push(path);
    }
    side_files.sort();

    let mut merged = Vec::with_capacity(side_files.len());
    for path in side_files {
        let file_name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("side file name is not UTF-8")?
            .to_string();
        let mut added = 0usize;
        let mut shadowed = 0usize;
        for (tensor, _shape, _dtype) in super::loader::read_safetensor_header(&path)? {
            match weight_map.entry(tensor) {
                std::collections::hash_map::Entry::Vacant(v) => {
                    v.insert(file_name.clone());
                    added += 1;
                }
                std::collections::hash_map::Entry::Occupied(_) => shadowed += 1,
            }
        }
        tracing::info!("EXL3 side file {file_name}: added {added} tensors to the weight map");
        if shadowed > 0 {
            tracing::warn!(
                "EXL3 side file {file_name}: {shadowed} tensors already mapped by the index \
                 (indexed shard wins)"
            );
        }
        merged.push(file_name);
    }
    Ok(merged)
}

/// True when `model_dir/config.json` declares `quantization_config.quant_method`
/// == "exl3" (case-insensitive). A missing config is not an EXL3 checkpoint; a
/// corrupt one is an error, never a silent false.
fn is_exl3_checkpoint(model_dir: &Path) -> Result<bool> {
    let path = model_dir.join("config.json");
    if !path.exists() {
        return Ok(false);
    }
    let text = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read {}", path.display()))?;
    let config: serde_json::Value = serde_json::from_str(&text)
        .with_context(|| format!("Failed to parse {}", path.display()))?;
    let method = config["quantization_config"]["quant_method"].as_str();
    Ok(method.is_some_and(|m| m.eq_ignore_ascii_case("exl3")))
}

#[cfg(test)]
#[path = "side_files_tests.rs"]
mod tests;
