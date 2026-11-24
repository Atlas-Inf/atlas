// SPDX-License-Identifier: AGPL-3.0-only

//! A sequence's holds on window-bounded KV (`spark_runtime::kv_cache::window`).
//!
//! Before each forward the sequence holds the window rows that forward reads
//! or writes, and lets go of the rows no later forward reads: those below
//! [`WindowPlan::first_block`] of the forward's first row. A sequence's first
//! row only moves forward (a rejected draft rewinds no further than the
//! verify's first row), so a released row is never read again.
//!
//! Snapshots pin the window they resume from. Aux collection runs with and
//! without the cache lock held, so it only queues the pin; the queue drains
//! under the lock before any hold changes, while the sequence still holds
//! every block it names.

use std::ops::Range;

use anyhow::{Result, ensure};
use parking_lot::Mutex;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{PagedKvCache, WindowPlan};

use crate::traits::SequenceState;

/// The cache's window geometry, and snapshot pins waiting for the lock.
pub(crate) struct KvWindow {
    plan: Option<(WindowPlan, usize)>,
    pins: Mutex<Vec<Vec<u32>>>,
}

impl KvWindow {
    pub(crate) fn new(kv: &PagedKvCache) -> Self {
        Self {
            plan: kv.window_plan().map(|p| (p.clone(), kv.block_size())),
            pins: Mutex::new(Vec::new()),
        }
    }

    pub(crate) fn is_active(&self) -> bool {
        self.plan.is_some()
    }

    /// Queue a pin of the window a snapshot of `seq` at its current length
    /// resumes from.
    pub(crate) fn queue_pin(&self, seq: &SequenceState) {
        let Some((plan, bs)) = &self.plan else {
            return;
        };
        let end = seq.seq_len.div_ceil(*bs).min(seq.block_table.len());
        let start = plan.first_block(seq.seq_len, *bs).min(end);
        if start < end {
            self.pins.lock().push(seq.block_table[start..end].to_vec());
        }
    }

    fn apply_pins(&self, kv: &mut PagedKvCache) {
        for blocks in self.pins.lock().drain(..) {
            kv.window_pin(&blocks);
        }
    }

    /// Hold the window rows a forward from `first_pos` through logical block
    /// `last_block` reads or writes, and release those below its window.
    /// Rows below `first_pos` (or the sequence's resume point) were written
    /// earlier and must still be mapped: a resume whose snapshot window is
    /// gone is refused, holding nothing, and no fresh (zeroed) block is ever
    /// lent for rows that exist.
    pub(crate) fn sync(
        &self,
        seq: &mut SequenceState,
        first_pos: usize,
        last_block: usize,
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some((plan, bs)) = &self.plan else {
            return Ok(());
        };
        self.apply_pins(kv);
        // rows below a resume point come from the snapshot, never computed
        let first_pos = first_pos.max(seq.marconi_skip_to);
        let len = seq.block_table.len();
        let held = clamp(&seq.window_held, len);
        let end = (last_block + 1).max(held.end).min(len);
        let want = plan.first_block(first_pos, *bs).min(end)..end;
        let written = first_pos.div_ceil(*bs);
        let gone: Vec<usize> = want
            .clone()
            .filter(|&i| i < written && !held.contains(&i))
            .filter(|&i| !kv.window_mapped(seq.block_table[i]))
            .collect();
        ensure!(
            gone.is_empty(),
            "KV window: window rows of logical blocks {gone:?} below position {first_pos} are gone"
        );
        let blocks = |range: Range<usize>, not: &Range<usize>| -> Vec<u32> {
            range
                .filter(|i| !not.contains(i))
                .map(|i| seq.block_table[i])
                .collect()
        };
        let (hold, release) = (blocks(want.clone(), &held), blocks(held, &want));
        kv.window_hold(&hold, gpu, stream)?;
        seq.window_held = want;
        kv.window_release(&release, gpu, stream)
    }

    /// After a prefill chunk: keep only the window the next forward reads.
    pub(crate) fn trim(
        &self,
        seq: &mut SequenceState,
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let last = seq.seq_len.saturating_sub(1) / kv.block_size();
        self.sync(seq, seq.seq_len, last, kv, gpu, stream)
    }

    /// Before `seq.block_table` shrinks to `len` blocks: release the holds
    /// on the blocks it drops (all of them for `len` 0).
    pub(crate) fn truncate(
        &self,
        seq: &mut SequenceState,
        len: usize,
        kv: &mut PagedKvCache,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if self.plan.is_none() {
            return Ok(());
        }
        self.apply_pins(kv);
        let held = clamp(&seq.window_held, seq.block_table.len());
        let dropped: Vec<u32> = (held.start.max(len)..held.end)
            .map(|i| seq.block_table[i])
            .collect();
        let end = held.end.min(len);
        seq.window_held = held.start.min(end)..end;
        kv.window_release(&dropped, gpu, stream)
    }
}

/// `held` within a table of `len` blocks. A hold past the table end is a
/// table that shrank without [`KvWindow::truncate`]: those holds are lost.
fn clamp(held: &Range<usize>, len: usize) -> Range<usize> {
    if held.end > len {
        tracing::error!("KV window: holds on blocks {held:?} past a {len}-block table");
    }
    held.start.min(len)..held.end.min(len)
}

impl super::types::TransformerModel {
    /// Hold the window a resume at `depth` reads; false (holding nothing
    /// new) when its rows are gone, so the prefill recomputes instead.
    pub(in crate::model) fn kv_window_adopt(
        &self,
        seq: &mut SequenceState,
        depth: usize,
        kv: &mut PagedKvCache,
        stream: u64,
    ) -> bool {
        let last = depth.saturating_sub(1) / kv.block_size();
        match self
            .kv_window
            .sync(seq, depth, last, kv, self.gpu.as_ref(), stream)
        {
            Ok(()) => true,
            Err(e) => {
                tracing::info!("prefix snapshot at {depth} declined: {e:#}");
                false
            }
        }
    }
}

#[cfg(test)]
#[path = "kv_window_tests.rs"]
mod tests;
