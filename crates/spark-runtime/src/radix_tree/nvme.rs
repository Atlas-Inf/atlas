// SPDX-License-Identifier: AGPL-3.0-only

//! [`NvmePrefixTier`] for [`RadixTree`]: thin locking wrappers over the
//! bookkeeping in `inner/nvme.rs`. Lock order as elsewhere: `inner`, released
//! before `snapshot_index`.

use super::RadixTree;
use super::inner::NvmeIndex;
use crate::prefix_cache::nvme::{NvmePrefixTier, NvmeStats, RestorePlan, SpillOrder};

impl NvmePrefixTier for RadixTree {
    fn enable_classes(&self, per_class: u32, classes: u32) -> bool {
        let mut inner = self.inner.lock();
        if inner.nvme.is_some() || per_class == 0 || classes == 0 {
            return false;
        }
        inner.nvme = Some(NvmeIndex::new(per_class, classes));
        true
    }

    fn is_enabled(&self) -> bool {
        self.inner.lock().nvme_on()
    }

    fn set_keep_restored(&self, keep: bool) {
        if let Some(idx) = self.inner.lock().nvme.as_mut() {
            idx.keep_restored = keep;
        }
    }

    fn forget_kept(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
        blocks: std::ops::Range<usize>,
    ) {
        self.inner
            .lock()
            .nvme_forget_kept(tokens, block_size, adapter_id, blocks);
    }

    fn plan_restore(&self, tokens: &[u32], block_size: usize, adapter_id: u64) -> RestorePlan {
        self.inner.lock().nvme_plan(tokens, block_size, adapter_id)
    }

    fn complete_restore(
        &self,
        tokens: &[u32],
        block_size: usize,
        adapter_id: u64,
        plan: &RestorePlan,
        restored: &[u32],
        failed: bool,
    ) -> Vec<u32> {
        self.inner
            .lock()
            .nvme_complete(tokens, block_size, adapter_id, plan, restored, failed)
    }

    fn spill_failed(&self, orders: &[SpillOrder]) -> Vec<u32> {
        self.inner.lock().nvme_spill_failed(orders)
    }

    fn snapshot_anchor_depth(
        &self,
        tokens: &[u32],
        limit: usize,
        session_hash: u64,
        adapter_id: u64,
    ) -> usize {
        self.snapshot_index
            .lock()
            .peek_deepest(tokens, limit, session_hash, adapter_id)
    }

    fn nvme_stats(&self) -> NvmeStats {
        self.inner.lock().nvme_stats()
    }
}
