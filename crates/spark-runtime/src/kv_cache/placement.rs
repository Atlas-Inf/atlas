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

/// Which buffers may move to the carveout (`ATLAS_KV_CARVEOUT_ORDER`).
///
/// The GPU does not cache the carveout in L2 (`cuda_backend::display_carveout`),
/// so a buffer there costs in proportion to how often its data is re-read.
/// The sparse index buffers are the worst case: the indexer scores every
/// earlier key for every query, so each index layer there cost about 1% of a
/// 64K cold prefill on the pair. The latent (`K`/`V`) pools are read only at
/// the selected rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CarveoutOrder {
    /// Any buffer, largest first (the original order; for comparison).
    Size,
    /// Latent pools only, largest first.
    Latent,
}

impl CarveoutOrder {
    pub fn from_env() -> Self {
        match std::env::var("ATLAS_KV_CARVEOUT_ORDER").as_deref() {
            Ok("size") => Self::Size,
            _ => Self::Latent,
        }
    }

    fn takes(self, buffer: KvBuffer) -> bool {
        self == Self::Size || matches!(buffer, KvBuffer::K(_) | KvBuffer::V(_))
    }
}

impl KvPlacement {
    /// [`Self::plan_ordered`] in the order `ATLAS_KV_CARVEOUT_ORDER` selects.
    pub fn plan(buffers: &[(KvBuffer, usize)], capacity: usize) -> Self {
        Self::plan_ordered(buffers, capacity, CarveoutOrder::from_env())
    }

    /// The buffers `which` takes, largest first, each taken while its
    /// footprint still fits `capacity`. Ties go to the lower [`KvBuffer`],
    /// so every rank with the same sizes plans the same set.
    pub fn plan_ordered(
        buffers: &[(KvBuffer, usize)],
        capacity: usize,
        which: CarveoutOrder,
    ) -> Self {
        let mut order: Vec<_> = buffers
            .iter()
            .filter(|(buffer, bytes)| *bytes > 0 && which.takes(*buffer))
            .collect();
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
        Self::buffer_sizes_with_k_slots(
            config,
            num_blocks,
            num_blocks,
            v_aliases_k,
            index,
            tail_slots,
        )
    }

    /// [`Self::buffer_sizes`] with `k_slots` K-pool block slots per layer,
    /// as [`Self::new_latent_sharded_placed`] allocates them (a rank's
    /// [`super::LatentShard::local_blocks_for`]).
    pub fn buffer_sizes_with_k_slots(
        config: &KvCacheConfig,
        num_blocks: usize,
        k_slots: usize,
        v_aliases_k: bool,
        index: Option<SparseIndexCacheConfig>,
        tail_slots: Option<TailSlotPlan>,
    ) -> Vec<(KvBuffer, usize)> {
        let mut out = Vec::new();
        for layer in 0..config.num_layers {
            out.push((
                KvBuffer::K(layer),
                k_slots * config.k_block_bytes_for_layer(layer),
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
