// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl TransformerModel {
    pub(in crate::model::trait_impl) fn num_free_blocks_dispatch(&self) -> usize {
        self.kv_cache.lock().num_free_blocks()
    }

    pub(in crate::model::trait_impl) fn num_total_blocks_dispatch(&self) -> usize {
        // The constructor permanently owns one zeroed padding block. Admission
        // and occupancy report only capacity available to real sequences.
        self.kv_cache.lock().num_blocks().saturating_sub(1)
    }

    pub(in crate::model::trait_impl) fn reclaim_prefix_blocks_dispatch(
        &self,
        num_blocks: usize,
    ) -> usize {
        if num_blocks == 0 || !self.prefix_cache.is_active() {
            return 0;
        }
        // KV lock BEFORE `evict`, as every other eviction site does: with the
        // NVMe tier, `evict` hands out record slots that must be written before
        // anyone else can evict/restore against the same tree + pool.
        let mut kv = self.kv_cache.lock();
        let evicted = self.prefix_cache.evict(num_blocks);
        if evicted.is_empty() {
            return 0;
        }
        let before = kv.num_free_in_all();
        self.apply_evicted(evicted, &mut kv);
        kv.num_free_in_all().saturating_sub(before)
    }
}
