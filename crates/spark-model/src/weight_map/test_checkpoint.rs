// SPDX-License-Identifier: AGPL-3.0-only

//! One reader for the GPU tests that compare against a real checkpoint's
//! bytes on disk (`ATLAS_FP4L_CKPT`): a tensor straight from its shard,
//! through the index, with no loader in between.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

use anyhow::{Context, Result};

/// `(dtype, shape, raw bytes)` of tensor `name` in the checkpoint at `dir`.
pub(crate) fn read_checkpoint_tensor(
    dir: &Path,
    name: &str,
) -> Result<(String, Vec<usize>, Vec<u8>)> {
    // Read whole, then parsed: `from_reader` on a bare File is a read(2) per
    // byte, ~47M syscalls for FP4-Local's index on every call.
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
    let shard = index["weight_map"][name]
        .as_str()
        .with_context(|| format!("{name}: not in the index"))?;
    read_safetensors_tensor(&dir.join(shard), name)
}

/// Rows of a two-dimensional tensor too large to read whole (an Engram
/// table), read one at a time from its shard.
pub(crate) struct CheckpointRows {
    file: std::fs::File,
    start: u64,
    pub(crate) rows: usize,
    pub(crate) row_bytes: usize,
}

impl CheckpointRows {
    /// Tensor `name` of the checkpoint at `dir`, whose values are one byte.
    pub(crate) fn open(dir: &Path, name: &str) -> Result<Self> {
        let index: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("model.safetensors.index.json"))?)?;
        let shard = index["weight_map"][name]
            .as_str()
            .with_context(|| format!("{name}: not in the index"))?;
        let mut file = std::fs::File::open(dir.join(shard))?;
        let mut len = [0u8; 8];
        file.read_exact(&mut len)?;
        let len = u64::from_le_bytes(len);
        let mut header = vec![0u8; len as usize];
        file.read_exact(&mut header)?;
        let header: serde_json::Value = serde_json::from_slice(&header)?;
        let t = header
            .get(name)
            .with_context(|| format!("{name}: not in {shard}"))?;
        let shape: Vec<usize> = serde_json::from_value(t["shape"].clone())?;
        let [lo, hi]: [u64; 2] = serde_json::from_value(t["data_offsets"].clone())?;
        anyhow::ensure!(
            shape.len() == 2 && (hi - lo) as usize == shape[0] * shape[1],
            "{name}: {shape:?} is not rows of one-byte values"
        );
        Ok(Self {
            file,
            start: 8 + len + lo,
            rows: shape[0],
            row_bytes: shape[1],
        })
    }

    /// Row `row`'s bytes, appended to `out`.
    pub(crate) fn read(&mut self, row: usize, out: &mut Vec<u8>) -> Result<()> {
        anyhow::ensure!(row < self.rows, "row {row} of {}", self.rows);
        self.file
            .seek(SeekFrom::Start(self.start + (row * self.row_bytes) as u64))?;
        let at = out.len();
        out.resize(at + self.row_bytes, 0);
        Ok(self.file.read_exact(&mut out[at..])?)
    }
}

/// `(dtype, shape, raw bytes)` of tensor `name` in the safetensors file `path`.
pub(crate) fn read_safetensors_tensor(
    path: &Path,
    name: &str,
) -> Result<(String, Vec<usize>, Vec<u8>)> {
    let mut f = std::fs::File::open(path).with_context(|| path.display().to_string())?;
    let mut len = [0u8; 8];
    f.read_exact(&mut len)?;
    let len = u64::from_le_bytes(len);
    let mut header = vec![0u8; len as usize];
    f.read_exact(&mut header)?;
    let header: serde_json::Value = serde_json::from_slice(&header)?;
    let t = header
        .get(name)
        .with_context(|| format!("{name}: not in {}", path.display()))?;
    let shape = serde_json::from_value(t["shape"].clone())?;
    let [lo, hi]: [u64; 2] = serde_json::from_value(t["data_offsets"].clone())?;
    f.seek(SeekFrom::Start(8 + len + lo))?;
    let mut data = vec![0u8; (hi - lo) as usize];
    f.read_exact(&mut data)?;
    Ok((t["dtype"].as_str().unwrap_or("").to_string(), shape, data))
}
