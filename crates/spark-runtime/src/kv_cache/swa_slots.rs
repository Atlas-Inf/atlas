// SPDX-License-Identifier: AGPL-3.0-only

//! Slot-mapped pools for sliding-window attention layers.
//!
//! A sliding-window layer only ever reads its last `window` tokens, so its
//! pool need not hold one block per physical block: a small pool of slots is
//! lent to each freshly allocated block and reclaimed once the block falls
//! more than `lag_blocks` behind its sequence's newest block. Kernels
//! translate `physical block -> slot` through a device map (`u32[num_blocks]`).
//! The evicted history is gone, so a cached prefix is reusable only where the
//! blocks its end reads still hold slots: the prompt-end tails the prefix
//! cache pins (`swa_pins.rs`).

use anyhow::{Result, ensure};

use super::swa_pins::SwaPins;
use super::tail_slots::{TailSlotPlan, TailSlots};
use super::{KvCacheConfig, KvPlacement, PagedKvCache};
use crate::gpu::{DevicePtr, GpuBackend};

/// Which attention layers hold slots, and how many.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SwaSlotPlan {
    /// Attention-layer indices whose K/V pools are slotted.
    pub layers: Vec<usize>,
    /// Blocks behind the newest written block that keep their slot: must
    /// cover the window plus one step's widest row span (a prefill chunk or
    /// a verify block) plus speculative rollback.
    pub lag_blocks: usize,
    /// Sequences that can hold slots at once.
    pub sequences: usize,
    /// Blocks a slotted layer reads behind a position (its window).
    pub window_blocks: usize,
    /// Prompt-end tails the prefix cache may keep slotted; 0 = no reuse.
    pub prefix_tails: usize,
    /// Blocks per pinned tail: the window plus slack for a next turn whose
    /// match ends a little short of the previous prompt's end.
    pub tail_blocks: usize,
}

impl SwaSlotPlan {
    fn slot_plan(&self) -> TailSlotPlan {
        TailSlotPlan {
            lag_blocks: self.lag_blocks,
            sequences: self.sequences,
        }
    }

    /// Slots per slotted layer: every sequence's lent ring plus the pinned
    /// tails (a pinned block may outlive the sequence that lent it).
    pub fn capacity(&self) -> usize {
        self.slot_plan().capacity() + self.prefix_tails * self.tail_blocks
    }

    pub fn holds(&self, layer: usize) -> bool {
        self.layers.contains(&layer)
    }

    /// Bytes of the slotted pools (K + V of every slotted layer).
    pub fn pool_bytes(&self, config: &KvCacheConfig) -> usize {
        self.layers
            .iter()
            .map(|&l| config.k_block_bytes_for_layer(l) + config.v_block_bytes_for_layer(l))
            .sum::<usize>()
            * self.capacity()
    }

    /// Per-block bytes of the layers that stay block-indexed.
    pub fn unslotted_block_bytes(&self, config: &KvCacheConfig) -> usize {
        (0..config.num_layers)
            .filter(|&l| !self.holds(l))
            .map(|l| config.k_block_bytes_for_layer(l) + config.v_block_bytes_for_layer(l))
            .sum()
    }
}

pub(super) struct SwaSlots {
    layers: Vec<bool>,
    pub(super) slots: TailSlots,
    pub(super) pins: SwaPins,
    pub(super) window_blocks: usize,
}

impl SwaSlots {
    /// Return `block`'s slot, unless a pinned tail holds it: the last unpin
    /// returns it then.
    fn release(&mut self, block: u32) {
        if !self.pins.park(block) {
            self.slots.release(block);
        }
    }

    /// `block` went back on the free list: no pin holds it (a pin holds a
    /// reference), so its slot goes too.
    pub(super) fn release_freed(&mut self, block: u32) {
        self.pins.forget(block);
        self.slots.release(block);
    }
}

impl PagedKvCache {
    /// A paged cache whose `plan.layers` pools hold `plan.capacity()` slots
    /// instead of `num_blocks` blocks.
    pub fn new_with_swa_slots(
        config: KvCacheConfig,
        num_blocks: usize,
        gpu: &dyn GpuBackend,
        placement: KvPlacement,
        plan: &SwaSlotPlan,
    ) -> Result<Self> {
        ensure!(
            plan.layers.iter().all(|&l| l < config.num_layers),
            "SWA slot plan names a layer beyond the cache's {} layers",
            config.num_layers
        );
        let mut layers = vec![false; config.num_layers];
        plan.layers.iter().for_each(|&l| layers[l] = true);
        let mut cache = Self::new_with_k_slots(
            config,
            num_blocks,
            num_blocks,
            gpu,
            false,
            placement,
            Some(plan),
        )?;
        let mut slots = TailSlots::new(
            num_blocks,
            plan.slot_plan(),
            plan.capacity(),
            gpu,
            &cache.placement,
        )?;
        slots.label = "sliding-window KV";
        let pins = SwaPins::new(num_blocks, plan.prefix_tails, plan.tail_blocks);
        cache.swa = Some(SwaSlots {
            layers,
            slots,
            pins,
            window_blocks: plan.window_blocks,
        });
        tracing::info!(
            "KV cache: {} sliding-window layers slotted ({} slots each, lag {} blocks, \
             {} prefix tails of {} blocks)",
            plan.layers.len(),
            plan.capacity(),
            plan.lag_blocks,
            plan.prefix_tails,
            plan.tail_blocks
        );
        Ok(cache)
    }

    /// Device block -> slot map for a slotted `layer`, else NULL (the
    /// layer's pool is indexed by block id).
    pub fn swa_slot_map_ptr(&self, layer: usize) -> DevicePtr {
        match &self.swa {
            Some(swa) if swa.layers[layer] => swa.slots.map,
            _ => DevicePtr::NULL,
        }
    }

    pub(super) fn swa_slotted(&self, layer: usize) -> bool {
        self.swa.as_ref().is_some_and(|s| s.layers[layer])
    }

    /// Lend each freshly allocated block a slot and publish the map, with
    /// any pending releases, before any kernel on `stream` writes them.
    pub fn lend_swa_slots(
        &mut self,
        blocks: &[u32],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let Some(swa) = self.swa.as_mut() else {
            return Ok(());
        };
        for &block in blocks {
            swa.slots.assign(block)?;
        }
        swa.slots.flush(gpu, stream)
    }

    /// Reclaim slots of blocks lagging more than `lag_blocks` behind
    /// `newest_block` (a logical index into `block_table`). Only blocks from
    /// `owned_from` on are this sequence's to release: the ones before it came
    /// from the prefix cache, lent by another sequence. Published with the
    /// next [`Self::lend_swa_slots`].
    pub fn release_lagging_swa_slots(
        &mut self,
        block_table: &[u32],
        newest_block: usize,
        owned_from: usize,
    ) {
        if let Some(swa) = self.swa.as_mut() {
            let owned = &block_table[owned_from.min(block_table.len())..];
            for i in swa
                .slots
                .lagging(owned.len(), newest_block.saturating_sub(owned_from))
            {
                swa.release(owned[i]);
            }
        }
    }

    /// A departing sequence returns the slots of the blocks it lent: a block
    /// the prefix cache keeps is never read through them again unless a
    /// pinned tail holds it.
    pub fn release_swa_slots(&mut self, owned_blocks: &[u32]) {
        if let Some(swa) = self.swa.as_mut() {
            owned_blocks.iter().for_each(|&block| swa.release(block));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvCacheDtype;

    fn config() -> KvCacheConfig {
        KvCacheConfig {
            block_size: 16,
            num_kv_heads: 8,
            head_dim: 192,
            num_layers: 4,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: Vec::new(),
            layer_dims: vec![(4, 192), (8, 192), (8, 192), (4, 192)],
            cache_blocks_per_seq: None,
        }
    }

    #[test]
    fn budget_splits_slotted_and_block_indexed_layers() {
        let cfg = config();
        let plan = SwaSlotPlan {
            layers: vec![1, 2],
            lag_blocks: 10,
            sequences: 2,
            window_blocks: 2,
            prefix_tails: 0,
            tail_blocks: 3,
        };
        assert_eq!(plan.capacity(), 24);
        assert_eq!(
            SwaSlotPlan {
                prefix_tails: 4,
                ..plan.clone()
            }
            .capacity(),
            36
        );
        let wide = cfg.k_block_bytes_for_layer(1) + cfg.v_block_bytes_for_layer(1);
        let narrow = cfg.k_block_bytes_for_layer(0) + cfg.v_block_bytes_for_layer(0);
        assert_eq!(plan.pool_bytes(&cfg), 2 * wide * 24);
        assert_eq!(plan.unslotted_block_bytes(&cfg), 2 * narrow);
        assert_eq!(
            plan.unslotted_block_bytes(&cfg) + 2 * wide,
            cfg.block_bytes_kv_all_layers()
        );
    }
}
