// SPDX-License-Identifier: AGPL-3.0-only

//! [`EvictedBlocks`] — what a prefix-cache eviction hands back to its caller.
//! Split out of `prefix_cache.rs` (500-LoC cap).

use super::nvme::SpillOrder;

/// Result of evicting LRU cached blocks (Phase 6.1.e).
#[derive(Debug, Clone, Default)]
pub struct EvictedBlocks {
    /// Physical block indices freed (caller calls `PagedKvCache::free_block`).
    pub physical: Vec<u32>,
    /// Parallel disk-block IDs to release (caller calls
    /// `HighSpeedSwap::dec_disk_ref`). Empty when HSS isn't in use.
    pub disk_block_ids: Vec<u32>,
    /// NVMe spill tier: blocks in `physical` whose bytes MUST be written to
    /// their record before the block is returned. Empty unless enabled.
    pub spill: Vec<SpillOrder>,
}

impl EvictedBlocks {
    pub fn is_empty(&self) -> bool {
        self.physical.is_empty()
    }

    pub fn len(&self) -> usize {
        self.physical.len()
    }
}
