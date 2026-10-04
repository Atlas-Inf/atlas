// SPDX-License-Identifier: AGPL-3.0-only

//! Which KV buffers live in the backend's carveout
//! ([`GpuBackend::alloc_carveout`], `gpu::carveout`).
//!
//! Each layer's pools are separate allocations, so whole buffers move: a
//! pool never straddles the carveout and system memory. A placement is
//! planned once, at the largest block count a rank sizes for, and then kept:
//! the ranks agree on a block count no larger, which only shrinks every
//! buffer, so the chosen set still fits the carveout and the buffers left
//! in system memory never grow past what the rank sized for.

use std::collections::BTreeSet;

use anyhow::Result;

use super::{KvCacheConfig, PagedKvCache, SparseIndexCacheConfig, TailSlotPlan};
use crate::gpu::carveout::CarveoutArena;
use crate::gpu::{DevicePtr, GpuBackend};

/// One per-layer device buffer of a [`PagedKvCache`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum KvBuffer {
    K(usize),
    V(usize),
    IndexValues(usize),
    IndexScales(usize),
    IndexTail(usize),
    /// The slot map of lent index tails (a `u32` per block).
    TailMap,
}

/// The buffers allocated from the carveout; every other one comes from
/// system memory. The default places nothing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct KvPlacement {
    carveout: BTreeSet<KvBuffer>,
}

impl KvPlacement {
    /// Largest buffers first, each taken while its footprint still fits
    /// `capacity`. Ties go to the lower [`KvBuffer`], so every rank with the
    /// same sizes plans the same set.
    pub fn plan(buffers: &[(KvBuffer, usize)], capacity: usize) -> Self {
        let mut order: Vec<_> = buffers.iter().filter(|(_, bytes)| *bytes > 0).collect();
        order.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut left = capacity;
        let mut carveout = BTreeSet::new();
        for &&(buffer, bytes) in &order {
            let need = CarveoutArena::footprint(bytes);
            if need <= left {
                left -= need;
                carveout.insert(buffer);
            }
        }
        Self { carveout }
    }

    pub fn contains(&self, buffer: KvBuffer) -> bool {
        self.carveout.contains(&buffer)
    }

    pub fn len(&self) -> usize {
        self.carveout.len()
    }

    pub fn is_empty(&self) -> bool {
        self.carveout.is_empty()
    }

    /// Bytes of `buffers` this placement keeps out of system memory.
    pub fn carved_bytes(&self, buffers: &[(KvBuffer, usize)]) -> usize {
        buffers
            .iter()
            .filter(|(buffer, _)| self.contains(*buffer))
            .map(|(_, bytes)| bytes)
            .sum()
    }

    pub(super) fn alloc(
        &self,
        gpu: &dyn GpuBackend,
        buffer: KvBuffer,
        bytes: usize,
    ) -> Result<DevicePtr> {
        if self.contains(buffer) {
            gpu.alloc_carveout(bytes)
        } else {
            gpu.alloc(bytes)
        }
    }
}

impl PagedKvCache {
    /// The per-layer buffers [`Self::new_placed`] and
    /// [`Self::attach_sparse_index_with_tail_slots`] allocate for
    /// `num_blocks`, with their sizes.
    pub fn buffer_sizes(
        config: &KvCacheConfig,
        num_blocks: usize,
        v_aliases_k: bool,
        index: Option<SparseIndexCacheConfig>,
        tail_slots: Option<TailSlotPlan>,
    ) -> Vec<(KvBuffer, usize)> {
        let mut out = Vec::new();
        for layer in 0..config.num_layers {
            out.push((
                KvBuffer::K(layer),
                num_blocks * config.k_block_bytes_for_layer(layer),
            ));
            if !v_aliases_k {
                out.push((
                    KvBuffer::V(layer),
                    num_blocks * config.v_block_bytes_for_layer(layer),
                ));
            }
        }
        if let Some(index) = index {
            let block_size = config.block_size;
            let tails = tail_slots.map_or(num_blocks, TailSlotPlan::capacity);
            if tail_slots.is_some() {
                out.push((KvBuffer::TailMap, num_blocks * std::mem::size_of::<u32>()));
            }
            for layer in 0..config.num_layers {
                out.extend([
                    (
                        KvBuffer::IndexValues(layer),
                        num_blocks * index.values_block_bytes(block_size),
                    ),
                    (
                        KvBuffer::IndexScales(layer),
                        num_blocks * index.scales_block_bytes(block_size),
                    ),
                    (
                        KvBuffer::IndexTail(layer),
                        tails * index.tail_block_bytes(block_size),
                    ),
                ]);
            }
        }
        out
    }
}

#[cfg(test)]
#[path = "placement_tests.rs"]
mod tests;
