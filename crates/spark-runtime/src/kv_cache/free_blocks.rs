// SPDX-License-Identifier: AGPL-3.0-only

//! Free physical blocks and block allocation.
//!
//! A latent shard (`latent_shard.rs`) stores block `b`'s latents on rank
//! `b % world`. The ranks of a pair mirror the same sequences but allocate
//! and free in rank-local order, so their physical ids differ. To keep the
//! owner of every LOGICAL block the same on both ranks, a sharded pool keeps
//! one free list per id residue and draws the block for logical index `l`
//! from list `l % world`: a table entry's id residue always equals its
//! logical index's, whatever the ids themselves. Prefix sharing is
//! positional, so a cached block is only ever reused at its own index.

use anyhow::{Result, anyhow};

use super::PagedKvCache;

pub(super) struct FreeBlocks {
    /// One list (unsharded), or one per id residue mod `world`.
    lists: Vec<Vec<u32>>,
}

impl FreeBlocks {
    /// Every block of `num_blocks` free over `lists` residues; each list pops
    /// its lowest id first.
    pub(super) fn new(num_blocks: usize, lists: usize) -> Self {
        let lists = (0..lists)
            .map(|residue| {
                (0..num_blocks as u32)
                    .rev()
                    .filter(|&b| b as usize % lists == residue)
                    .collect()
            })
            .collect();
        Self { lists }
    }

    fn pop(&mut self, logical: usize) -> Option<u32> {
        let n = self.lists.len();
        self.lists[logical % n].pop()
    }

    pub(super) fn push(&mut self, block: u32) {
        let n = self.lists.len();
        self.lists[block as usize % n].push(block);
    }

    /// Blocks any mix of logical indices can draw: every free block with one
    /// list, else `world` times the shortest list.
    fn allocatable(&self) -> usize {
        match self.lists.as_slice() {
            [only] => only.len(),
            lists => lists.len() * lists.iter().map(Vec::len).min().unwrap_or(0),
        }
    }

    fn total(&self) -> usize {
        self.lists.iter().map(Vec::len).sum()
    }

    pub(super) fn clear(&mut self) {
        self.lists.iter_mut().for_each(Vec::clear);
    }
}

impl PagedKvCache {
    /// Take the block for logical index `logical` off its free list.
    #[track_caller]
    fn take_block(&mut self, logical: usize, event: &'static str) -> Option<u32> {
        let idx = self.free_blocks.pop(logical)?;
        self.block_ref_counts[idx as usize] = 1;
        if self.trace.is_on() {
            self.trace
                .record(idx as usize, event, 1, std::panic::Location::caller());
        }
        Some(idx)
    }

    /// Allocate a free block for logical block `logical` of its sequence
    /// (under a latent shard the id's residue matches `logical`'s, see the
    /// module docs). `None` when exhausted.
    #[track_caller]
    pub fn try_alloc_block_at(&mut self, logical: usize) -> Option<u32> {
        self.take_block(logical, "try_alloc")
    }

    /// [`Self::try_alloc_block_at`], failing when exhausted.
    #[track_caller]
    pub fn alloc_block_at(&mut self, logical: usize) -> Result<u32> {
        let Some(idx) = self.take_block(logical, "alloc") else {
            // A shard's residue can run dry while the other still has blocks.
            return Err(match self.latent_shard {
                None => anyhow!("KV cache exhausted: no free blocks"),
                Some(_) => anyhow!(
                    "KV cache exhausted: no free blocks for logical block {logical} ({} free in all)",
                    self.free_blocks.total()
                ),
            });
        };
        Ok(idx)
    }

    /// Allocate a free block. Returns block index. Refused under a latent
    /// shard, whose blocks must be drawn per logical index
    /// ([`Self::alloc_block_at`]).
    #[track_caller]
    pub fn alloc_block(&mut self) -> Result<u32> {
        self.ensure_unsharded("alloc_block (no logical index)")?;
        self.alloc_block_at(0)
    }

    /// Try to allocate a free block without failing. Returns None if
    /// exhausted. Panics under a latent shard (see [`Self::alloc_block`]).
    #[track_caller]
    pub fn try_alloc_block(&mut self) -> Option<u32> {
        self.assert_unsharded("try_alloc_block (no logical index)");
        self.try_alloc_block_at(0)
    }

    /// Number of free blocks: under a latent shard, the blocks any mix of
    /// logical indices is guaranteed to get (`world` × the scarcest residue).
    pub fn num_free_blocks(&self) -> usize {
        self.free_blocks.allocatable()
    }

    /// Every free block, whatever its residue: what measures the progress of
    /// a reclaim, since an eviction that frees only the richer residue of a
    /// latent shard leaves [`Self::num_free_blocks`] where it was. The same
    /// count unsharded.
    pub fn num_free_in_all(&self) -> usize {
        self.free_blocks.total()
    }
}
