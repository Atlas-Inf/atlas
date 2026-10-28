// SPDX-License-Identifier: AGPL-3.0-only

//! Safetensors header parsing and shard discovery for
//! [`super::FastSafetensorsLoader`].

use crate::weights::WeightDtype;
use anyhow::{Context, Result, bail};
use serde_json::Value;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

/// Index file format: `{ "weight_map": { "tensor_name": "shard_filename" } }`.
#[derive(serde::Deserialize)]
struct SafetensorsIndex {
    weight_map: HashMap<String, String>,
}

/// Parsed entry from a safetensor header.
pub(super) struct TensorMeta {
    pub(super) name: String,
    pub(super) dtype: WeightDtype,
    /// True when the shard stores this tensor as IEEE F16. The tensor is
    /// staged as BF16 (`dtype` above) and the copy loop converts the bytes —
    /// same 2-byte element size, different bit layout.
    pub(super) from_f16: bool,
    pub(super) shape: Vec<usize>,
    /// Absolute byte offset in the file where the tensor bytes start.
    pub(super) abs_offset: u64,
    /// Tensor byte length.
    pub(super) len: usize,
}

/// The tensors an index routes to `shard_path`.
/// Discover which safetensor files to load. Mirrors the resolution order in
/// `crate::weights::SafetensorsLoader::load`.
///
/// Returns `(shard_files, Some(tensor→shard))` when an index is present and
/// `(shard_files, None)` when we're loading everything in the listed files.
#[allow(clippy::type_complexity)]
pub(super) fn resolve_shards(
    model_dir: &Path,
) -> Result<(Vec<PathBuf>, Option<HashMap<String, String>>)> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let consolidated_index = model_dir.join("consolidated.safetensors.index.json");

    let actual_index = if index_path.exists() {
        Some(index_path)
    } else if consolidated_index.exists() {
        Some(consolidated_index)
    } else {
        None
    };

    if let Some(ip) = actual_index {
        let json = std::fs::read_to_string(&ip)
            .with_context(|| format!("Failed to read {}", ip.display()))?;
        let mut index: SafetensorsIndex = serde_json::from_str(&json)?;
        // EXL3 tensors in unindexed side files join the map (no-op otherwise),
        // so they become shard files below.
        let _merged = crate::weights::merge_exl3_side_files(model_dir, &mut index.weight_map)?;
        let mut shards: Vec<String> = index
            .weight_map
            .values()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        shards.sort();
        let files: Vec<PathBuf> = shards.iter().map(|s| model_dir.join(s)).collect();
        return Ok((files, Some(index.weight_map)));
    }

    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok((vec![single], None));
    }

    let mut shards: Vec<_> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                (n.starts_with("model.safetensors-") || n.starts_with("consolidated-"))
                    && n.ends_with(".safetensors")
            })
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        bail!(
            "No safetensor files found in {}. Expected model.safetensors*, \
             consolidated.safetensors*, or consolidated-*-of-*.safetensors",
            model_dir.display()
        );
    }
    Ok((shards, None))
}

/// Parse the safetensors header of an open file. Tensor offsets in the header
/// are relative to the start of the data section; we convert to absolute file
/// offsets up-front so downstream callers only need the fd.
pub(super) fn parse_header(file: &mut File) -> Result<Vec<TensorMeta>> {
    let file_len = file.metadata()?.len();
    let mut size_buf = [0u8; 8];
    file.read_exact(&mut size_buf)?;
    let header_size = u64::from_le_bytes(size_buf) as usize;
    if header_size > 64 * 1024 * 1024 {
        bail!("safetensor header too large ({} bytes)", header_size);
    }
    let mut header_buf = vec![0u8; header_size];
    file.read_exact(&mut header_buf)?;
    let data_start: u64 = 8 + header_size as u64;

    let json: Value = serde_json::from_slice(&header_buf)?;
    let obj = json
        .as_object()
        .context("invalid safetensor header (not an object)")?;

    let mut out = Vec::with_capacity(obj.len());
    for (name, info) in obj {
        if name == "__metadata__" {
            continue;
        }
        let dtype_str = info["dtype"].as_str().unwrap_or("BF16");
        let (dtype, from_f16) = match dtype_str {
            // F16 is not store-legal (WeightDtype is closed to store dtypes):
            // stage as BF16 and mark for byte conversion in the copy loop.
            // centml modelopt W4A4 exports ship all unquantized tensors as F16.
            // EXL3 scale vectors are the exception: keep the raw bytes (FP16).
            "F16" => {
                if crate::weights::keeps_raw_f16(name) {
                    (WeightDtype::FP16, false)
                } else {
                    (WeightDtype::BF16, true)
                }
            }
            // CSF component streams (slab exception words; csf_32x2_v1 page
            // origins, prefixes and records). Never stored: the copy loop
            // hands every CSF part to `CsfAssembler`, which re-derives the
            // element width from byte length against shape. `UInt8` is only
            // the staging container here; U16/U32 anywhere else is still
            // refused below.
            "U16" | "U32" if crate::weights::nvfp4_csf::csf_part(name).is_some() => {
                (WeightDtype::UInt8, false)
            }
            // Every other dtype maps exactly as on the buffered and RDMA paths:
            // one table (`from_safetensors_str`), so the loaders cannot drift.
            other => (
                WeightDtype::from_safetensors_str(other)
                    .with_context(|| format!("tensor {name}"))?,
                false,
            ),
        };
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();
        // Shared with the RDMA manifest builder: a reversed or past-EOF
        // `data_offsets` pair is rejected here rather than wrapping into a
        // `u64::MAX`-ish `len` that becomes a pread window below.
        let span = atlas_core::safetensors::tensor_span(
            name,
            &info["data_offsets"],
            data_start,
            file_len,
        )?;
        out.push(TensorMeta {
            name: name.clone(),
            dtype,
            from_f16,
            shape,
            abs_offset: span.abs_offset,
            len: span.len as usize,
        });
    }
    // Sort by offset so the reader does sequential disk access.
    out.sort_by_key(|t| t.abs_offset);
    Ok(out)
}

#[cfg(test)]
#[path = "header_tests.rs"]
mod tests;
