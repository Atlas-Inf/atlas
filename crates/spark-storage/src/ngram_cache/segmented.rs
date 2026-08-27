// SPDX-License-Identifier: AGPL-3.0-only

//! Segmented n-gram tables: shards scattered across one or MORE files, and
//! the single-scalar FP8 dequant scale that RadixArk's Qwen3.8-Flash-Next
//! conversion ships instead of a per-row scale file.
//!
//! Split out of `ngram_cache.rs` to keep that file under the 500-LoC cap. A
//! child module, so the cache's private fields are still reachable.

use anyhow::{Context, Result, bail};

use super::{BLOCK, NgramRowCache, ScaleCache, Segments, open_direct};
use crate::expert_arena::ExpertArena;
use std::path::Path;

impl NgramRowCache {
    /// As [`Self::open_at`], but for a table split across equal-sized shards
    /// at SCATTERED offsets, in one or MORE files — Qwen3.8-Flash-Next's PLE
    /// table, whose 128 shard tensors are neither consecutive nor confined to
    /// a single safetensors file. `shards[i]` is shard `i`'s backing file and
    /// the byte offset of its first row; every shard holds `rows_per_shard`
    /// rows.
    ///
    /// Each DISTINCT path is opened once, so a 128-shard table over 10 files
    /// costs 10 descriptors rather than 128.
    #[allow(clippy::too_many_arguments)]
    pub fn open_segmented(
        shards: &[(std::path::PathBuf, u64)],
        rows_per_shard: u64,
        scale_path: Option<&Path>,
        row_stride: usize,
        slots: usize,
    ) -> Result<Self> {
        if shards.is_empty() || rows_per_shard == 0 {
            bail!(
                "NgramRowCache: segmented table needs shards and rows \
                 (shards={}, rows_per_shard={rows_per_shard})",
                shards.len()
            );
        }
        let (paths, shard_file) = plan_shard_files(shards);
        let mut files = Vec::with_capacity(paths.len());
        for path in &paths {
            files.push(open_direct(path)?);
        }
        let rows_total = shards.len() as u64 * rows_per_shard;
        // `open_at` opens shard 0's file for the unsegmented `self.file`
        // field, which a segmented cache never reads — `row_loc` always
        // resolves through `segments.files`. Kept so the single-offset path
        // stays byte for byte what it was.
        let mut c = Self::open_at(paths[0], 0, scale_path, rows_total, row_stride, slots)?;
        c.segments = Some(Segments {
            bases: shards.iter().map(|(_, off)| *off).collect(),
            rows_per: rows_per_shard,
            files,
            shard_file,
        });
        Ok(c)
    }

    /// Install a SINGLE per-tensor dequant scale, filling every slot with it.
    ///
    /// For an FP8 table whose checkpoint ships one scalar rather than a
    /// per-row scale file. `batched_embed_fp8` indexes the scale array by slot
    /// and multiplies, so a constant array makes it compute `fp8 * scale` for
    /// every row — the correct dequant — with no kernel change and no extra
    /// I/O on the fault path.
    ///
    /// Must be called before the first `resolve`, and refuses to overwrite a
    /// per-row scale file, since silently ignoring one would dequantize an
    /// entire table with the wrong number.
    pub fn set_constant_scale(&mut self, scale: f32) -> Result<()> {
        if let Some(sc) = &self.scales {
            if sc.file.is_some() {
                bail!(
                    "NgramRowCache: a per-row scale file is already installed; \
                     a constant per-tensor scale would silently override it"
                );
            }
        }
        let sbytes = self.slots * 4;
        let sblocks = sbytes.div_ceil(BLOCK);
        let arena = ExpertArena::new(1, sblocks as u32, BLOCK)
            .context("NgramRowCache: constant scale arena")?;
        // SAFETY: the arena holds sblocks*BLOCK >= slots*4 bytes.
        unsafe {
            let base = arena.slot_host_ptr(0, 0)?;
            let all = std::slice::from_raw_parts_mut(base as *mut f32, self.slots);
            all.fill(scale);
        }
        self.scales = Some(ScaleCache { arena, file: None });
        Ok(())
    }
}

/// Distinct backing paths in first-use order, and `shard -> path index`.
///
/// Split out of [`NgramRowCache::open_segmented`] and kept free of any CUDA so
/// it is directly testable: a segmented table's shards may live in several
/// files, and assuming otherwise silently loses every shard outside the first
/// one. Dedupes so a 128-shard table over 10 files costs 10 descriptors.
pub(super) fn plan_shard_files(shards: &[(std::path::PathBuf, u64)]) -> (Vec<&Path>, Vec<u32>) {
    let mut paths: Vec<&Path> = Vec::new();
    let mut shard_file = Vec::with_capacity(shards.len());
    for (path, _) in shards {
        let idx = match paths.iter().position(|p| *p == path.as_path()) {
            Some(i) => i,
            None => {
                paths.push(path.as_path());
                paths.len() - 1
            }
        };
        shard_file.push(idx as u32);
    }
    (paths, shard_file)
}
