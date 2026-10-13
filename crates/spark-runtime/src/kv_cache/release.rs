// SPDX-License-Identifier: AGPL-3.0-only

//! `PagedKvCache` teardown (`ModelResource::release`). Split from
//! `kv_cache.rs` (500-LoC cap).

use super::PagedKvCache;

/// Release K/V and any attached sparse-index pools for every layer.
///
/// Each layer allocates its K and V pools separately, so freeing per layer is
/// correct. The block bookkeeping (`free_blocks`, `block_ref_counts`) is host
/// state indexing into those pools — cleared with them so a released cache
/// cannot hand out a block into freed memory.
impl atlas_core::scope::ModelResource<dyn crate::gpu::GpuBackend> for PagedKvCache {
    fn label(&self) -> &'static str {
        "kv cache"
    }

    fn release(&mut self, gpu: &dyn crate::gpu::GpuBackend) -> anyhow::Result<()> {
        let mut first_error = None;
        for layer in self.layers.drain(..) {
            for ptr in [
                layer.k_pool,
                layer.owned_v_pool(),
                layer.sparse_index_values,
                layer.sparse_index_scales,
                layer.sparse_index_tail,
            ] {
                if ptr.is_null() {
                    continue;
                }
                if let Err(e) = gpu.free(ptr)
                    && first_error.is_none()
                {
                    first_error = Some(e);
                }
            }
        }
        if let Some(tails) = self.tail_slots.take()
            && let Err(e) = gpu.free(tails.map)
            && first_error.is_none()
        {
            first_error = Some(e);
        }
        self.free_blocks.clear();
        self.block_ref_counts.clear();
        self.num_blocks = 0;
        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }
}
