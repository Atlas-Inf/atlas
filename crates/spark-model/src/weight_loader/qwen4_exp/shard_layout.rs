// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE table's on-disk shard layout, read from safetensors headers.
//!
//! Test-only, and split from `qwen4_exp.rs` for the 500-LoC cap.

use anyhow::{Context, Result};

/// The PLE table's shard layout, read straight from a checkpoint's
/// safetensors header: `(backing file, byte offset of first row)` per shard,
/// the rows each shard holds, and the shards' safetensors dtype string
/// (`"BF16"` / `"F8_E4M3"`) — which decides the row stride, and is 1 byte per
/// element on the RadixArk NVFP4 conversion.
///
/// Exists so a test can rebuild the segmented row cache WITHOUT loading a
/// 75 GB model — the gather is the one part of PLE whose failure is invisible
/// downstream, so it needs a cheap isolated arm.
///
/// Returns a path PER SHARD because the shards are not confined to one file:
/// the RadixArk NVFP4 conversion spreads 128 shards over 10
/// `model-plefp8-*.safetensors`. An earlier version read shard 0's header and
/// looked every shard up inside it, which found nothing for the 118 shards
/// living elsewhere.
///
/// Its only caller is `ops/ple_tests.rs`, which is a GPU test and therefore
/// gated on the cuda feature — so this must be too, or a metal test build
/// trips `deny(dead_code)`.
pub fn ple_shard_layout(snapshot: &str) -> Result<(Vec<(std::path::PathBuf, u64)>, u64, String)> {
    use std::collections::HashMap;
    use std::io::{Read, Seek, SeekFrom};
    let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        std::path::Path::new(snapshot).join("model.safetensors.index.json"),
    )?)?;
    let map = idx["weight_map"].as_object().context("weight_map")?;
    let mut names: Vec<(usize, &String)> = map
        .keys()
        .filter(|k| k.contains(".ngram_embedding.shard_"))
        .map(|k| {
            let n = k
                .rsplit("shard_")
                .next()
                .and_then(|r| r.split('.').next())
                .and_then(|r| r.parse().ok())
                .unwrap_or(usize::MAX);
            (n, k)
        })
        .collect();
    names.sort();
    anyhow::ensure!(!names.is_empty(), "no PLE shards in {snapshot}");

    // One header read per DISTINCT file, cached — 10 reads, not 128.
    let mut headers: HashMap<String, (serde_json::Value, u64)> = HashMap::new();
    let mut shards = Vec::with_capacity(names.len());
    let mut rows_per = 0u64;
    let mut dtype = String::new();
    for (i, name) in &names {
        let file = map[name.as_str()].as_str().context("shard file")?;
        let path = std::path::Path::new(snapshot).join(file);
        if !headers.contains_key(file) {
            let mut fh = std::fs::File::open(&path)?;
            let mut len = [0u8; 8];
            fh.read_exact(&mut len)?;
            let hlen = u64::from_le_bytes(len);
            let mut hdr = vec![0u8; hlen as usize];
            fh.seek(SeekFrom::Start(8))?;
            fh.read_exact(&mut hdr)?;
            headers.insert(file.to_string(), (serde_json::from_slice(&hdr)?, 8 + hlen));
        }
        let (hdr, data_start) = &headers[file];
        let e = &hdr[name.as_str()];
        let off = e["data_offsets"][0]
            .as_u64()
            .with_context(|| format!("data_offsets for shard {i} in {file}"))?;
        let rows = e["shape"][0].as_u64().context("shape")?;
        let dt = e["dtype"].as_str().context("dtype")?.to_string();
        if *i == 0 {
            rows_per = rows;
            dtype = dt.clone();
        }
        anyhow::ensure!(dt == dtype, "shard {i} is {dt}, not {dtype}");
        anyhow::ensure!(
            rows == rows_per,
            "shard {i} has {rows} rows, not {rows_per}"
        );
        shards.push((path, data_start + off));
    }
    Ok((shards, rows_per, dtype))
}
