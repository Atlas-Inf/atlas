// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix reuse under sliding-window KV eviction (`ATLAS_SWA_EVICT=1`).
//!
//! A slotted sliding-window layer keeps only the newest rows of each
//! sequence, so a cached prefix is resumable only where the window behind
//! its end still holds slots. A finished prompt pins its tail
//! (`PagedKvCache::pin_swa_tail`), and a lookup is capped to the deepest
//! block whose window is resident, before the ranks agree on a match.

use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::PrefixMatch;

use super::prefix_share::cap_prefix_match;
use super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Bring `prefix_match` down to the deepest block a sliding-window layer
    /// can resume from. A no-op unless sliding-window layers are slotted.
    pub(crate) fn swa_cap_prefix_match(
        &self,
        key_tokens: &[u32],
        seq: &SequenceState,
        prefix_match: PrefixMatch,
        kv_cache: &mut PagedKvCache,
    ) -> PrefixMatch {
        let Some(depth) = kv_cache.swa_adoptable_blocks(&prefix_match.matched_blocks) else {
            return prefix_match;
        };
        let bs = kv_cache.block_size();
        let keep = depth * bs;
        if keep >= prefix_match.matched_tokens {
            return prefix_match;
        }
        tracing::info!(
            "SWA prefix reuse: radix matched {} tokens, sliding-window rows resident \
             through {keep}",
            prefix_match.matched_tokens
        );
        cap_prefix_match(
            self.prefix_cache.as_ref(),
            key_tokens,
            bs,
            seq.session_hash,
            seq.adapter_id,
            prefix_match,
            keep,
        )
    }

    /// After a prompt's prefill: pin the sliding-window tail of its whole
    /// blocks, if the prefix cache published them.
    pub(crate) fn pin_swa_prompt_tail(&self, seq: &SequenceState) {
        if !self.prefix_cache.is_active() {
            return;
        }
        let mut kv_cache = self.kv_cache.lock();
        if !kv_cache.swa_prefix_tails_enabled() {
            return;
        }
        let bs = kv_cache.block_size();
        let whole = seq.prompt_len.min(seq.tokens.len()) / bs;
        if whole == 0 || whole > seq.block_table.len() {
            return;
        }
        let key = self.pc_key_tokens(seq, &seq.tokens[..whole * bs]);
        if self
            .prefix_cache
            .peek_matched_tokens(&key, bs, seq.adapter_id)
            < whole * bs
        {
            return;
        }
        kv_cache.pin_swa_tail(&seq.block_table[..whole]);
    }
}
