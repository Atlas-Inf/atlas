// SPDX-License-Identifier: AGPL-3.0-only

//! R11 fix 2: cap prefix-cache KV reuse at the SSM anchor (split from
//! `prefix_lookup.rs` for the 500-LoC cap).

use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// R11 fix 2 (`ATLAS_PREFIX_KV_CAP=0` restores the old behaviour): when the
    /// SSM anchor `snap_tok` sits below the radix match, drop the matched KV
    /// blocks past the anchor so `[snap_tok, matched)` is recomputed into the
    /// sequence's own blocks by the same pass a cold run uses. The old path
    /// kept those blocks (written by another request's pass layout) and set a
    /// no-rewrite floor at `matched`. Returns false when the re-acquire of the
    /// capped prefix fails; the caller then takes the full-recompute path with
    /// no prefix (nothing restored yet).
    pub(in crate::model) fn prefill_b_cap_kv_to_snapshot(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        matched_block_count: usize,
        snap_tok: usize,
        matched: &mut usize,
        kv_cache: &mut PagedKvCache,
    ) -> bool {
        let bs = kv_cache.block_size();
        // Part of the exact path: off with the grid, so default cache-on
        // keeps main's KV reuse.
        if self.prefill_grid() == 0
            || snap_tok >= *matched
            || std::env::var("ATLAS_PREFIX_KV_CAP").as_deref() == Ok("0")
            || !snap_tok.is_multiple_of(bs)
            || !seq.disk_block_ids.is_empty()
            || self.multi_rank_protocol_active()
            || matched_block_count > seq.block_table.len()
        {
            return true;
        }
        let old = *matched;
        let base = seq.block_table.len() - matched_block_count;
        for b in seq.block_table.drain(base..).collect::<Vec<_>>() {
            kv_cache.dec_ref(b);
        }
        self.prefix_cache
            .release_matched(tokens, bs, old, seq.adapter_id);
        let m2 =
            self.prefix_cache
                .lookup(&tokens[..snap_tok], bs, seq.session_hash, seq.adapter_id);
        if m2.matched_tokens != snap_tok || m2.matched_blocks.len() != snap_tok / bs {
            if m2.matched_tokens > 0 {
                self.prefix_cache.release_matched(
                    &tokens[..snap_tok],
                    bs,
                    m2.matched_tokens,
                    seq.adapter_id,
                );
            }
            tracing::warn!(
                "prefix KV cap: re-acquire of {snap_tok} tokens matched {} — \
                 dropping the prefix, full recompute",
                m2.matched_tokens
            );
            *matched = 0;
            seq.cached_prefix_tokens = 0;
            seq.cached_prefix_blocks = 0;
            seq.prefix_ref_tokens.clear();
            return false;
        }
        for &b in &m2.matched_blocks {
            kv_cache.inc_ref(b);
            seq.block_table.push(b);
        }
        *matched = snap_tok;
        seq.cached_prefix_tokens = snap_tok;
        seq.cached_prefix_blocks = m2.matched_blocks.len();
        seq.prefix_ref_tokens = tokens[..snap_tok].to_vec();
        tracing::info!(
            "prefix KV cap: reuse capped at the SSM anchor {snap_tok} (radix matched {old}); \
             recomputing [{snap_tok}, {old}) into private blocks"
        );
        true
    }
}
