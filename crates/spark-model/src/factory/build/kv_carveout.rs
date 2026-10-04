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

use spark_runtime::gpu::GpuBackend;
use spark_runtime::kv_cache::{
    KvBuffer, KvCacheConfig, KvPlacement, PagedKvCache, SparseIndexCacheConfig, TailSlotPlan,
};

/// The cache geometry the factory allocates (`glm::new_kv_cache` and
/// `attach_sparse_index_with_tail_slots`).
#[derive(Clone, Copy)]
pub(super) struct KvShape<'a> {
    pub(super) config: &'a KvCacheConfig,
    pub(super) v_aliases_k: bool,
    pub(super) index: Option<SparseIndexCacheConfig>,
    pub(super) tail_slots: Option<TailSlotPlan>,
}

impl KvShape<'_> {
    fn buffers(self, blocks: usize) -> Vec<(KvBuffer, usize)> {
        PagedKvCache::buffer_sizes(
            self.config,
            blocks,
            self.v_aliases_k,
            self.index,
            self.tail_slots,
        )
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
    let system = total(&shape.buffers(blocks));
    let per_block = total(&shape.buffers(blocks + 1)) - system;
    // A block's bytes all in the carveout bound the gain; walk down from there
    // to the first count whose system-memory share fits. `blocks` always does.
    let ceiling = blocks + capacity / per_block.max(1);
    for n in (blocks..=ceiling).rev() {
        let buffers = shape.buffers(n);
        let placement = KvPlacement::plan(&buffers, capacity);
        if total(&buffers) - placement.carved_bytes(&buffers) <= system {
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
