// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix reuse under sliding-window KV eviction (`ATLAS_SWA_EVICT=1`).
//!
//! A slotted sliding-window layer keeps only the newest rows of each
//! sequence, so a cached prefix is resumable only where the window behind
//! its end still holds slots. A finished prompt pins its tail
//! (`PagedKvCache::pin_swa_tail`), and the agreed prefix match is capped to
//! the deepest block whose window is resident (after the ranks agree).

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

    /// After the prefill of `prompt` (the whole prompt): pin the
    /// sliding-window tail of its whole blocks. Pins come from the block
    /// table alone, so both EP ranks pin the same prompt ends even when
    /// their prefix caches disagree.
    pub(crate) fn pin_swa_prompt_tail(&self, prompt: &[u32], seq: &SequenceState) {
        if !self.prefix_cache.is_active() {
            return;
        }
        let mut kv_cache = self.kv_cache.lock();
        if !kv_cache.swa_prefix_tails_enabled() {
            return;
        }
        let bs = kv_cache.block_size();
        let whole = prompt.len() / bs;
        if whole == 0 || whole > seq.block_table.len() {
            return;
        }
        // Pin from the block table alone, not the radix: the two EP ranks'
        // caches can disagree on what is published, and both must pin the
        // same prompt ends or their slot maps diverge.
        kv_cache.pin_swa_tail(&seq.block_table[..whole]);
    }
}
