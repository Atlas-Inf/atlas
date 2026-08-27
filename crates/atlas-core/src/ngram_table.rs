// SPDX-License-Identifier: AGPL-3.0-only

//! Row-gathered access to the n-gram embedding table, without holding it.
//!
//! Qwen3.8-Flash-Next's n-gram table is **51.2 billion parameters** —
//! `320_001_536 x 160` — larger than the rest of the model put together, and
//! ~52 GB even at FP8. On a GB10 there is nowhere to put it: the box is
//! coherent unified memory (`Addressing Mode: ATS`, ~119.6 GiB total), so CPU
//! and GPU draw on one pool and "moving it to host" frees nothing.
//!
//! It does not need to be held. The table is a pure row gather: one token
//! touches `K*(N-1)` rows — 16 on the published model — of `hidden / 16` = 160
//! bytes each, about 2.5 KB. So rows are read on demand and the kernel's page
//! cache keeps the hot set.
//!
//! `pread` rather than `mmap`, deliberately. Both go through the page cache, so
//! the caching is identical, but a page fault inside a CUDA-adjacent thread
//! stalls it invisibly, and a truncated or concurrently-modified file turns
//! into SIGBUS with `mmap` and an ordinary `io::Error` with `pread`.

use crate::config::ModelConfig;
use crate::weight_manifest::locate_checkpoint;
use anyhow::{Context, Result, bail, ensure};
use std::fs::File;
use std::path::Path;

/// Positional read at an absolute offset, without moving the file cursor —
/// `pread(2)`. The whole point is that concurrent readers do not serialise on a
/// shared cursor, which is what lets the PLE gather run without a mutex around
/// the file.
///
/// Unix spells it `FileExt::read_exact_at`; Windows spells it
/// `seek_read`, which is `OVERLAPPED` underneath and also does not move the
/// cursor — but it is a SHORT read like `read`, so it needs the loop. The
/// NVMe tiers are Linux-only in production; this arm exists so the workspace
/// builds on the Windows release leg, which a missing `cfg` broke.
#[cfg(unix)]
fn read_exact_at(file: &File, out: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::unix::fs::FileExt;
    file.read_exact_at(out, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, out: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < out.len() {
        let n = file.seek_read(&mut out[done..], offset + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read from the n-gram table",
            ));
        }
        done += n;
    }
    Ok(())
}

/// One shard of the concatenated table: a file, the byte span its rows occupy,
/// and the global row it starts at.
struct Shard {
    file: File,
    /// Absolute offset of this shard's first row in its file.
    base: u64,
    /// Global row index this shard begins at.
    first_row: u64,
    rows: u64,
}

/// The n-gram table, read a row at a time.
pub struct NgramTable {
    shards: Vec<Shard>,
    row_bytes: usize,
    dtype: String,
    total_rows: u64,
}

/// Bytes per element for the dtypes an n-gram table is published in.
fn element_bytes(dtype: &str) -> Option<usize> {
    match dtype {
        "F8_E4M3" | "F8_E5M2" | "U8" | "I8" => Some(1),
        "BF16" | "F16" => Some(2),
        "F32" | "I32" => Some(4),
        _ => None,
    }
}

impl NgramTable {
    /// Open the table for one PLE layer of a checkpoint directory.
    ///
    /// Validates that the shards tile the table exactly: `split_ngram_parts`
    /// of them, equal row counts, contiguous, summing to the padded total the
    /// config implies. A checkpoint that disagrees is refused here rather than
    /// producing rows from the wrong place.
    pub fn open(dir: &Path, config: &ModelConfig, ple_layer_index: usize) -> Result<Self> {
        let ngram = config
            .qwen4exp_ngram(ple_layer_index)?
            .context("checkpoint declares no PLE n-gram tower")?;
        let decoder_layer = config
            .ple_decoder_layer(ple_layer_index)
            .context("no decoder layer for that PLE index")?;
        let prefix = format!(
            "model.language_model.layers.{decoder_layer}.ple.ple_embedding.ngram_embedding"
        );

        let located = locate_checkpoint(dir)?;
        let parts = config.split_ngram_parts;
        ensure!(parts > 0, "split_ngram_parts must be non-zero");

        let expected_total = ngram.padded_rows(config.ngram_vocab_size_base);
        let mut shards = Vec::with_capacity(parts);
        let mut first_row = 0u64;
        let mut row_bytes = 0usize;
        let mut dtype = String::new();

        for index in 0..parts {
            let name = format!("{prefix}.shard_{index}.weight");
            let loc = located
                .get(&name)
                .with_context(|| format!("checkpoint is missing {name}"))?;
            ensure!(
                loc.shape.len() == 2 && loc.shape[1] == ngram.head_dim(),
                "{name}: expected [rows, {}], got {:?}",
                ngram.head_dim(),
                loc.shape
            );
            // The table stays FP8 in both published releases -- even the NVFP4
            // repack, which excludes `*.ple.*` -- but the width is taken from
            // the checkpoint rather than assumed, so a BF16 or F32 export (the
            // tiny development checkpoint is F32) reads correctly too.
            let element = element_bytes(&loc.dtype)
                .with_context(|| format!("{name}: unsupported dtype {}", loc.dtype))?;
            if row_bytes == 0 {
                row_bytes = ngram.head_dim() * element;
                dtype = loc.dtype.clone();
            } else {
                ensure!(
                    loc.dtype == dtype,
                    "{name}: dtype {} disagrees with shard_0's {dtype}",
                    loc.dtype
                );
            }
            let rows = loc.shape[0] as u64;
            ensure!(
                loc.span.len == rows * row_bytes as u64,
                "{name}: span {} bytes does not match {rows} x {row_bytes}",
                loc.span.len
            );
            shards.push(Shard {
                file: File::open(&loc.path)
                    .with_context(|| format!("opening {}", loc.path.display()))?,
                base: loc.span.abs_offset,
                first_row,
                rows,
            });
            first_row += rows;
        }

        if first_row != expected_total {
            bail!(
                "shards hold {first_row} rows but the config implies {expected_total}; \
                 the table does not tile"
            );
        }
        Ok(Self {
            shards,
            row_bytes,
            dtype,
            total_rows: first_row,
        })
    }

    pub fn row_bytes(&self) -> usize {
        self.row_bytes
    }

    /// On-disk element type of the table, as the checkpoint declares it.
    pub fn dtype(&self) -> &str {
        &self.dtype
    }

    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Read one global row into `out`, which must be exactly `row_bytes` long.
    pub fn read_row(&self, row: u64, out: &mut [u8]) -> Result<()> {
        ensure!(
            out.len() == self.row_bytes,
            "row buffer is {} bytes, expected {}",
            out.len(),
            self.row_bytes
        );
        ensure!(
            row < self.total_rows,
            "row {row} past the table's {} rows",
            self.total_rows
        );
        // Shards are equal-sized in every published release, but binary search
        // keeps this correct if a future one is not.
        let index = self
            .shards
            .partition_point(|s| s.first_row + s.rows <= row)
            .min(self.shards.len() - 1);
        let shard = &self.shards[index];
        let offset = shard.base + (row - shard.first_row) * self.row_bytes as u64;
        read_exact_at(&shard.file, out, offset).with_context(|| format!("reading row {row}"))?;
        Ok(())
    }

    /// Gather `ids` and dequantize into `out` (`ids.len() * head_dim` floats).
    ///
    /// This is the PLE tower's actual input: the per-head slices of one token's
    /// n-gram embedding, concatenated in head order and scaled.
    ///
    /// `weight_scale` is the single BF16 scalar the checkpoint ships next to
    /// the shards — the table is quantized PER TENSOR, not per block, which is
    /// why the 128 shards share one scale and take no `weight_scale_inv`.
    pub fn gather_dequant(&self, ids: &[u32], weight_scale: f32, out: &mut [f32]) -> Result<()> {
        let head_dim = self.head_dim();
        ensure!(
            out.len() == ids.len() * head_dim,
            "gather buffer holds {} floats, expected {}",
            out.len(),
            ids.len() * head_dim
        );
        let mut raw = vec![0u8; self.row_bytes];
        for (id, slot) in ids.iter().zip(out.chunks_exact_mut(head_dim)) {
            self.read_row(*id as u64, &mut raw)?;
            match self.dtype.as_str() {
                "F8_E4M3" => {
                    for (byte, value) in raw.iter().zip(slot.iter_mut()) {
                        *value = crate::numeric::fp8_e4m3_to_f32(*byte) * weight_scale;
                    }
                }
                "BF16" => {
                    for (pair, value) in raw.chunks_exact(2).zip(slot.iter_mut()) {
                        let bits = u16::from_le_bytes([pair[0], pair[1]]);
                        *value = f32::from_bits((bits as u32) << 16) * weight_scale;
                    }
                }
                "F32" => {
                    for (quad, value) in raw.chunks_exact(4).zip(slot.iter_mut()) {
                        *value =
                            f32::from_le_bytes([quad[0], quad[1], quad[2], quad[3]]) * weight_scale;
                    }
                }
                other => bail!("n-gram table dtype {other} has no dequant path"),
            }
        }
        Ok(())
    }

    /// Elements per row, as opposed to [`Self::row_bytes`].
    pub fn head_dim(&self) -> usize {
        match self.dtype.as_str() {
            "BF16" | "F16" => self.row_bytes / 2,
            "F32" | "I32" => self.row_bytes / 4,
            _ => self.row_bytes,
        }
    }

    /// Gather `ids` into `out`, laid out row-major (`ids.len() * row_bytes`).
    ///
    /// This is the shape the PLE tower wants: the per-head slices of one
    /// token's embedding, concatenated in head order.
    pub fn gather(&self, ids: &[u32], out: &mut [u8]) -> Result<()> {
        ensure!(
            out.len() == ids.len() * self.row_bytes,
            "gather buffer is {} bytes, expected {}",
            out.len(),
            ids.len() * self.row_bytes
        );
        for (id, slot) in ids.iter().zip(out.chunks_exact_mut(self.row_bytes)) {
            self.read_row(*id as u64, slot)?;
        }
        Ok(())
    }
}
