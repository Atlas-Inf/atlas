// SPDX-License-Identifier: AGPL-3.0-only

//! One sequence's DeepSeek-V4.1 attention cache on one layer, and what
//! source layers hand to the layers after them within a pass.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::cache_recipe::WINDOW;
use super::index::INDEX_DIM;
use super::ops::HEAD_DIM;
use super::shape::MAX_RATIO;
use crate::layer::LayerState;

/// The most rows of a batch whose effect on a sequence can be taken back
/// ([`SeqState::roll_back`]): a verify of that many tokens less one draft.
pub const KEPT_ROWS: usize = 16;

/// A source layer's compressed keys for one sequence, their index keys, and
/// the group its compressor is filling.
pub struct CompressedKeys {
    /// `[capacity, 512]` BF16, one row a completed group.
    pub keys: DevicePtr,
    /// `[capacity, 128]` BF16: what the indexer scores each group by.
    pub index_keys: DevicePtr,
    pub capacity: usize,
    /// Buffers these outgrew. A row of a pass that was handed one still
    /// reads it, and every group it may read is in it, so they are kept
    /// until the sequence ends.
    retired: Vec<DevicePtr>,
    /// `[ratio, 512]` FP32: the key and gate projections of the group being
    /// filled.
    pub(super) kv_state: DevicePtr,
    pub(super) score_state: DevicePtr,
    /// `[MAX_RATIO + KEPT_ROWS, 512]` FP32 each: the same projections of the
    /// group that was open before the last small batch and of the batch's
    /// tokens, from which the open group of any of its positions is put back.
    pub(super) kept_kv: DevicePtr,
    pub(super) kept_score: DevicePtr,
}

impl CompressedKeys {
    /// Room for `groups` groups, keeping the ones held.
    pub(super) fn reserve(
        &mut self,
        gpu: &dyn GpuBackend,
        groups: usize,
        stream: u64,
    ) -> Result<()> {
        if groups <= self.capacity {
            return Ok(());
        }
        // Kernels still running may be writing the buffers being copied.
        gpu.synchronize(stream)?;
        let capacity = groups.max(2 * self.capacity);
        for (buffer, width) in [
            (&mut self.keys, HEAD_DIM),
            (&mut self.index_keys, INDEX_DIM),
        ] {
            let grown = gpu.alloc(capacity * width * 2)?;
            gpu.copy_d2d(*buffer, grown, self.capacity * width * 2)?;
            self.retired.push(std::mem::replace(buffer, grown));
        }
        self.capacity = capacity;
        Ok(())
    }
}

/// One sequence's attention cache on one layer.
pub struct SeqState {
    /// `[128, 512]` BF16 ring: slot `position % 128` holds that position's key.
    pub window: DevicePtr,
    pub compressed: Option<CompressedKeys>,
    /// `[KEPT_ROWS, 512]` BF16: the window keys the last small batch
    /// displaced from the ring, in the order of its rows; and that batch's
    /// first position and rows, while it can still be taken back.
    pub(super) kept_window: DevicePtr,
    pub(super) kept: Option<(usize, usize)>,
    /// On an Engram layer: the compressed id of the token at every position
    /// so far, which the n-grams ending at the next ones are hashed from.
    pub history: Vec<u32>,
}

impl SeqState {
    /// `ratio` is the layer's compression ratio when it produces compressed
    /// keys itself, else 0. There is room for `positions` positions to start
    /// with, and it grows.
    pub fn alloc(gpu: &dyn GpuBackend, ratio: usize, positions: usize) -> Result<Self> {
        let window = gpu.alloc(WINDOW * HEAD_DIM * 2)?;
        gpu.memset(window, 0, WINDOW * HEAD_DIM * 2)?;
        let compressed = match ratio {
            0 => None,
            ratio => {
                let capacity = (positions / ratio).max(1);
                Some(CompressedKeys {
                    keys: gpu.alloc(capacity * HEAD_DIM * 2)?,
                    index_keys: gpu.alloc(capacity * INDEX_DIM * 2)?,
                    capacity,
                    retired: Vec::new(),
                    kv_state: gpu.alloc(ratio * HEAD_DIM * 4)?,
                    score_state: gpu.alloc(ratio * HEAD_DIM * 4)?,
                    kept_kv: gpu.alloc((MAX_RATIO + KEPT_ROWS) * HEAD_DIM * 4)?,
                    kept_score: gpu.alloc((MAX_RATIO + KEPT_ROWS) * HEAD_DIM * 4)?,
                })
            }
        };
        Ok(Self {
            window,
            compressed,
            kept_window: gpu.alloc(KEPT_ROWS * HEAD_DIM * 2)?,
            kept: None,
            history: Vec::new(),
        })
    }

    /// Take back all but the first `accepted` of the `rows` tokens the last
    /// batch ran: a verify whose later drafts were refused. The ring gets
    /// back the keys those rows displaced, a compressor the group that was
    /// open at the last accepted token, and an Engram's history ends there.
    /// Keys of groups the refused rows completed stay where they are: no
    /// query sees them before the tokens that complete those groups again
    /// have written them.
    pub fn roll_back(
        &mut self,
        gpu: &dyn GpuBackend,
        ratio: usize,
        accepted: usize,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        if accepted >= rows {
            return Ok(());
        }
        let Some((first, kept)) = self.kept else {
            anyhow::bail!(
                "deepseek_v41: {accepted} of {rows} rows accepted, and the batch was not kept"
            );
        };
        anyhow::ensure!(
            kept == rows,
            "deepseek_v41: a verify of {rows} rows after a batch of {kept}"
        );
        gpu.synchronize(stream)?;
        let key = HEAD_DIM * 2;
        for row in accepted..rows {
            let slot = (first + row) % WINDOW;
            let from = self.kept_window.offset(row * key);
            gpu.copy_d2d(from, self.window.offset(slot * key), key)?;
        }
        let end = first + accepted;
        if let Some(c) = self.compressed.as_ref().filter(|_| ratio > 1) {
            // The pools start at the group that was open before the batch.
            let (start, open) = (first - first % ratio, end % ratio);
            let (at, bytes) = ((end - open - start) * HEAD_DIM * 4, open * HEAD_DIM * 4);
            if bytes > 0 {
                gpu.copy_d2d(c.kept_kv.offset(at), c.kv_state, bytes)?;
                gpu.copy_d2d(c.kept_score.offset(at), c.score_state, bytes)?;
            }
        }
        self.history.truncate(end);
        self.kept = None;
        Ok(())
    }

    pub fn free(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.window)?;
        gpu.free(self.kept_window)?;
        if let Some(c) = self.compressed.take() {
            let kept = [c.kept_kv, c.kept_score];
            for p in [c.keys, c.index_keys, c.kv_state, c.score_state]
                .into_iter()
                .chain(kept)
            {
                gpu.free(p)?;
            }
            for p in c.retired {
                gpu.free(p)?;
            }
        }
        Ok(())
    }
}

impl LayerState for SeqState {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// What source layers hand down the stack for one row of a pass. Layers run
/// in order, so a source has written before its readers ask.
#[derive(Clone)]
pub(super) struct RowHandoff {
    /// The latest key source's compressed keys and index keys.
    pub(super) compressed: DevicePtr,
    pub(super) index_keys: DevicePtr,
    /// The compressed positions the latest index source chose for the row's
    /// query, in ascending order.
    pub(super) chosen: Vec<i32>,
    /// One flag per compressed position, from the candidate source: the
    /// positions the index sources after it may choose from.
    pub(super) candidates: Vec<bool>,
}

impl Default for RowHandoff {
    fn default() -> Self {
        Self {
            compressed: DevicePtr::NULL,
            index_keys: DevicePtr::NULL,
            chosen: Vec::new(),
            candidates: Vec::new(),
        }
    }
}

/// What source layers hand down the stack within one pass, per row.
pub struct Handoff {
    pub(super) rows: Vec<RowHandoff>,
    /// `[table_rows, index_topk]` i32: for the rows of a pass that went in
    /// batches, the compressed positions the latest index source chose.
    table: DevicePtr,
    table_rows: usize,
    /// `[flag_rows, flag_cols]` bytes: for the same rows, the blocks of
    /// compressed positions the candidate source left to choose from.
    flags: DevicePtr,
    flag_rows: usize,
    /// Whether the pass reads every row of the last layer, as rank 0 said at
    /// its first layer (`tail.rs`).
    pub(super) whole: bool,
}

impl Default for Handoff {
    fn default() -> Self {
        Self {
            rows: Vec::new(),
            table: DevicePtr::NULL,
            table_rows: 0,
            flags: DevicePtr::NULL,
            flag_rows: 0,
            whole: true,
        }
    }
}

/// `buffer`, which holds `held` rows of `width` bytes, with room for `rows`:
/// grown, keeping what it holds, when it has less.
fn grown(
    gpu: &dyn GpuBackend,
    buffer: &mut DevicePtr,
    held: &mut usize,
    rows: usize,
    width: usize,
    stream: u64,
) -> Result<DevicePtr> {
    if rows > *held {
        // Kernels still running may be writing the buffer being copied.
        gpu.synchronize(stream)?;
        let room = rows.max(2 * *held);
        let larger = gpu.alloc(room * width)?;
        if *held > 0 {
            gpu.copy_d2d(*buffer, larger, *held * width)?;
            gpu.free(*buffer)?;
        }
        (*buffer, *held) = (larger, room);
    }
    Ok(*buffer)
}

impl Handoff {
    /// The table of chosen positions, with room for `rows` rows of `topk`.
    pub(super) fn table(
        &mut self,
        gpu: &dyn GpuBackend,
        rows: usize,
        topk: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let (table, held) = (&mut self.table, &mut self.table_rows);
        grown(gpu, table, held, rows, topk * 4, stream)
    }

    /// The candidate blocks' flags, with room for `rows` rows of `cols`.
    pub(super) fn flags(
        &mut self,
        gpu: &dyn GpuBackend,
        rows: usize,
        cols: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let (flags, held) = (&mut self.flags, &mut self.flag_rows);
        grown(gpu, flags, held, rows, cols, stream)
    }
}
