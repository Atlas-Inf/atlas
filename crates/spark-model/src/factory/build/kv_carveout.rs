// SPDX-License-Identifier: AGPL-3.0-only

//! KV blocks in the GPU backend's carveout (`spark_runtime::gpu::carveout`,
//! the GB10 display carveout under `spark display-carveout`).
//!
//! The carveout is extra memory, so it adds blocks without taking any from
//! system memory: the pool grows to the most blocks whose buffers left in
//! system memory still fit the bytes the budgeted pool would have used,
//! with whole per-layer buffers moved into the carveout. The placement is
//! planned at that count and kept through the ranks' agreement
//! (`glm::agree_kv_blocks`), which can only lower it.
//!
//! Under a latent shard (`ATLAS_GLM_KV_SHARD=1`) each rank's K pools hold
//! `ceil(blocks / 2)` slots and the shard's one scratch allocation grows with
//! the pool (its identity table), so both enter the system-memory account the
//! same way. Both ranks plan from the same sizes; their latents differ, their
//! buffer sizes do not.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{
    KvBuffer, KvCacheConfig, KvPlacement, LatentShard, LatentShardSpec, PagedKvCache,
    SparseIndexCacheConfig, TailSlotPlan,
};

/// The cache geometry the factory allocates (`glm::new_kv_cache` and
/// `attach_sparse_index_with_tail_slots`).
#[derive(Clone, Copy)]
pub(super) struct KvShape<'a> {
    pub(super) config: &'a KvCacheConfig,
    pub(super) v_aliases_k: bool,
    pub(super) index: Option<SparseIndexCacheConfig>,
    pub(super) tail_slots: Option<TailSlotPlan>,
    /// The latent shard the pool is built with (`glm::new_kv_cache`).
    pub(super) latent_shard: Option<LatentShardSpec>,
}

impl KvShape<'_> {
    fn buffers(self, blocks: usize) -> Vec<(KvBuffer, usize)> {
        let k_slots = self
            .latent_shard
            .map_or(blocks, |s| LatentShard::local_blocks_for(blocks, s.world));
        PagedKvCache::buffer_sizes_with_k_slots(
            self.config,
            blocks,
            k_slots,
            self.v_aliases_k,
            self.index,
            self.tail_slots,
        )
    }

    /// System-memory bytes outside the per-layer buffers that depend on
    /// `blocks`: the latent shard's scratch with its identity table.
    fn side_bytes(self, blocks: usize) -> usize {
        self.latent_shard
            .map_or(0, |s| LatentShard::allocation_bytes(blocks, &s))
    }

    /// System-memory bytes of a `blocks` pool with `placement`'s buffers in
    /// the carveout.
    fn system_bytes(self, blocks: usize, placement: &KvPlacement) -> usize {
        let buffers = self.buffers(blocks);
        total(&buffers) - placement.carved_bytes(&buffers) + self.side_bytes(blocks)
    }
}

fn total(buffers: &[(KvBuffer, usize)]) -> usize {
    buffers.iter().map(|(_, bytes)| bytes).sum()
}

/// The block count and placement for a pool budgeted at `blocks`, given a
/// carveout of `capacity` bytes. Without one: `blocks`, nothing placed.
pub(super) fn plan(shape: KvShape, blocks: usize, capacity: usize) -> (usize, KvPlacement) {
    if capacity == 0 || blocks == 0 {
        return (blocks, KvPlacement::default());
    }
    let system = shape.system_bytes(blocks, &KvPlacement::default());
    // Per block over two blocks: a sharded K pool grows by a slot every
    // second block. Unsharded this is the one-block step.
    let per_block = (total(&shape.buffers(blocks + 2)) - total(&shape.buffers(blocks))) / 2;
    // A block's bytes all in the carveout bound the gain (to within the one
    // block the two-block step can hide); walk down from there to the first
    // count whose system-memory share fits. `blocks` always does.
    let ceiling = blocks + capacity / per_block.max(1) + 1;
    for n in (blocks..=ceiling).rev() {
        let placement = KvPlacement::plan(&shape.buffers(n), capacity);
        if shape.system_bytes(n, &placement) <= system {
            return (n, placement);
        }
    }
    unreachable!("the budgeted count fits its own system-memory bytes")
}

/// [`plan`] against `gpu`'s carveout, logged.
pub(super) fn extend(gpu: &dyn GpuBackend, shape: KvShape, blocks: usize) -> (usize, KvPlacement) {
    let capacity = gpu.carveout_capacity();
    let (n, placement) = plan(shape, blocks, capacity);
    if !placement.is_empty() {
        let carved = placement.carved_bytes(&shape.buffers(n));
        tracing::info!(
            "KV cache: display carveout {:.0} MiB holds {} buffers ({:.0} MiB) → {} blocks \
             (+{}, +{:.1}%) for the same system memory",
            capacity as f64 / (1 << 20) as f64,
            placement.len(),
            carved as f64 / (1 << 20) as f64,
            n,
            n - blocks,
            (n - blocks) as f64 * 100.0 / blocks as f64,
        );
    }
    (n, placement)
}

#[cfg(test)]
#[path = "kv_carveout_tests.rs"]
mod tests;
