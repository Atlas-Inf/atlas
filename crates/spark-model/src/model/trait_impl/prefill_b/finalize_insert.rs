// SPDX-License-Identifier: AGPL-3.0-only

//! The last chunk's prefix-cache inserts, under the prompt's cache key
//! (`vision_cache`).

use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// The radix key `tokens` (a prefix of `seq`'s stream) registers under
    /// (`vision_cache::pc_key`), or `None` to decline: a placeholder the
    /// content key does not cover (two images would collide on one key), or
    /// a slid HSS window (`block_table` no longer parallels the token
    /// stream, see `hss_window_slid`).
    pub(super) fn finalize_key<'a>(
        &self,
        seq: &SequenceState,
        tokens: &'a [u32],
    ) -> Option<std::borrow::Cow<'a, [u32]>> {
        self.pc_key(seq, tokens)
            .filter(|_| !self.hss_window_slid(seq))
    }

    /// Insert `tokens` over `(block_table, disk_block_ids)` under its key
    /// ([`Self::finalize_key`]), or nothing where it declines.
    pub(super) fn finalize_insert(
        &self,
        seq: &SequenceState,
        tokens: &[u32],
        (blocks, disk): (&[u32], &[u32]),
        matched: usize,
        kv_cache: &mut PagedKvCache,
    ) {
        let Some(key) = self.finalize_key(seq, tokens) else {
            return;
        };
        let bs = kv_cache.block_size();
        let acquired = self
            .prefix_cache
            .insert(&key, blocks, disk, bs, matched, seq.adapter_id);
        super::super::super::super::block_mgmt::cache_acquires_refs(&acquired, kv_cache);
    }
}
