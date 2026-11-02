// SPDX-License-Identifier: AGPL-3.0-only

//! Slot-mapped raw tails for the sparse semantic index.
//!
//! A raw key/gate tail is only read to finalize the pool that contains the
//! token being written, so only a sequence's newest blocks ever need one.
//! Instead of one tail per physical block, a small pool of tail slots is lent
//! to fresh blocks and reclaimed once a block falls behind its sequence end.
//! Kernels translate `physical block -> tail slot` through a device map;
//! [`NO_TAIL`] makes them skip the row.

use anyhow::{Result, bail};

use super::PagedKvCache;
use crate::gpu::{DevicePtr, GpuBackend};

/// Device map entry for a block that owns no tail slot.
pub const NO_TAIL: u32 = u32::MAX;

/// How far behind a sequence's newest block a tail can still be read, and
/// how many sequences can hold tails at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TailSlotPlan {
    /// Blocks behind the newest written block whose tails stay lent: must
    /// cover one step's widest row span plus speculative rollback.
    pub lag_blocks: usize,
    pub sequences: usize,
}

impl TailSlotPlan {
    /// Lent tails per sequence: the lagging window plus the newest block
    /// and the block a new step may open.
    pub fn capacity(self) -> usize {
        self.sequences * (self.lag_blocks + 2)
    }
}

pub(super) struct TailSlots {
    pub(super) map: DevicePtr,
    host: Vec<u32>,
    free: Vec<u32>,
    plan: TailSlotPlan,
    dirty: Vec<u32>,
}

impl TailSlots {
    pub(super) fn new(
        num_blocks: usize,
        plan: TailSlotPlan,
        gpu: &dyn GpuBackend,
        placement: &super::KvPlacement,
    ) -> Result<Self> {
        let capacity = plan.capacity();
        if capacity == 0 || u32::try_from(capacity).is_err() {
            bail!("sparse index tail slot capacity {capacity} must be a positive u32");
        }
        let map_bytes = num_blocks * std::mem::size_of::<u32>();
        let map = placement.alloc(gpu, super::KvBuffer::TailMap, map_bytes)?;
        if let Err(error) = gpu.memset(map, 0xFF, num_blocks * std::mem::size_of::<u32>()) {
            let _ = gpu.free(map);
            return Err(error);
        }
        Ok(Self {
            map,
            host: vec![NO_TAIL; num_blocks],
            free: (0..capacity as u32).rev().collect(),
            plan,
            dirty: Vec::new(),
        })
    }

    pub(super) fn capacity(&self) -> usize {
        self.plan.capacity()
    }

    fn assign(&mut self, block: u32) -> Result<()> {
        if self.host[block as usize] != NO_TAIL {
            return Ok(());
        }
        let Some(slot) = self.free.pop() else {
            bail!(
                "sparse index tail slots exhausted ({} in use, plan {:?})",
                self.plan.capacity(),
                self.plan
            );
        };
        self.host[block as usize] = slot;
        self.dirty.push(block);
        Ok(())
    }

    pub(super) fn release(&mut self, block: u32) {
        let slot = std::mem::replace(&mut self.host[block as usize], NO_TAIL);
        if slot != NO_TAIL {
            self.free.push(slot);
            self.dirty.push(block);
        }
    }

    /// Publish every changed entry, one copy per run of contiguous blocks.
    /// Releases and assignments land in the same stream-ordered flush, so a
    /// reused slot is never mapped by two blocks when a kernel reads the map.
    fn flush(&mut self, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        self.dirty.sort_unstable();
        self.dirty.dedup();
        for run in self.dirty.chunk_by(|a, b| *b == *a + 1) {
            let first = run[0] as usize;
            let entries = &self.host[first..first + run.len()];
            // SAFETY: a `u32` slice viewed as its own bytes, same lifetime.
            let bytes = unsafe {
                std::slice::from_raw_parts(
                    entries.as_ptr().cast::<u8>(),
                    std::mem::size_of_val(entries),
                )
            };
            let dst = self.map.offset(first * std::mem::size_of::<u32>());
            gpu.copy_h2d_async(bytes, dst, stream)?;
        }
        self.dirty.clear();
        Ok(())
    }
}

impl PagedKvCache {
    /// Device `u32[num_blocks]` block -> tail-slot map, or NULL when every
    /// block carries its own tail.
    pub fn sparse_index_tail_map_ptr(&self) -> DevicePtr {
        self.tail_slots.as_ref().map_or(DevicePtr::NULL, |t| t.map)
    }

    /// Tails are slotted and `block` holds none: the index kernels would
    /// skip its rows (`NO_TAIL`) and never finalize its pooled keys.
    pub fn tail_slot_missing(&self, block: u32) -> bool {
        self.tail_slots
            .as_ref()
            .is_some_and(|t| t.host[block as usize] == NO_TAIL)
    }

    /// Lend each freshly allocated block a tail slot (when slotted) and
    /// publish the map, with any pending releases, before any kernel on
    /// `stream` can write the blocks.
    pub fn lend_tail_slots(
        &mut self,
        blocks: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(tails) = self.tail_slots.as_mut() else {
            return Ok(());
        };
        for &block in blocks {
            tails.assign(block)?;
        }
        tails.flush(gpu, stream)
    }

    /// Reclaim tails that lag more than `lag_blocks` behind `newest_block`
    /// (a logical index into `block_table`). Scans only the window the
    /// previous call could have left behind, so a step costs O(lag), not
    /// O(context). Published with the next [`Self::lend_tail_slots`].
    pub fn release_lagging_tail_slots(&mut self, block_table: &[u32], newest_block: usize) {
        let Some(tails) = self.tail_slots.as_mut() else {
            return;
        };
        let lag = tails.plan.lag_blocks;
        let end = newest_block.saturating_sub(lag).min(block_table.len());
        let start = end.saturating_sub(2 * lag + 2);
        for &block in &block_table[start..end] {
            tails.release(block);
        }
    }

    pub(super) fn release_tail_slot_if_freed(&mut self, block: u32) {
        if self.block_ref_counts[block as usize] == 0
            && let Some(tails) = self.tail_slots.as_mut()
        {
            tails.release(block);
        }
    }
}
