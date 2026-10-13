// SPDX-License-Identifier: AGPL-3.0-only

//! How a sequence takes cached KV blocks.
//!
//! Invariant: no sequence writes a row for its own tokens into a KV block
//! that anyone else holds. A block is shared, with the prefix cache or another
//! sequence, only while every position in it is a committed token of the
//! shared prefix. Two rules keep it:
//!
//! * A match covers whole blocks only (`PrefixCache::lookup_whole_blocks`,
//!   checked again at adoption below). A sequence appends from its matched
//!   length on, so the first block it appends to is one it allocated.
//! * The cache publishes whole blocks only (`RadixTree` insert), so the block
//!   a live sequence still appends to, or rewrites during a speculative
//!   verify, is referenced by that sequence alone.
//!
//! What the invariant does not cover: a sequence may still rewrite rows of a
//! shared block with values recomputed from the same token prefix the rows
//! already hold. Holders read an equivalent row, not a bit-identical one.
//! Three paths do it:
//!
//! * A hybrid-SSM prefill that matched a prefix but has no snapshot to resume
//!   from, or whose exact-hit snapshot shortcut is bypassed (the default),
//!   recomputes the whole prompt and rewrites every matched row. This is the
//!   usual outcome for a new conversation that shares a long prefix with
//!   another one, and for an identical block-aligned re-send.
//! * A GLM prefill that resumes from a snapshot below its matched length
//!   rewrites `[snapshot, matched)`: `prefill_attention_paged` applies the
//!   `cached_prefix_tokens` write floor on its non-MLA path only, and the
//!   GLM branch returns before it.
//! * A full-prompt hit that replays its last token through the one-token
//!   decode fork rewrites that row: the fork takes no write floor.

use anyhow::{Result, ensure};
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::prefix_cache::{PrefixCache, PrefixMatch};

use super::block_mgmt::reuse_prefix_match_disk_ids;
use crate::traits::SequenceState;

/// Take the sequence's references on a prefix match and start its block
/// table with the matched blocks. Every prefill path acquires its cached
/// prefix here. A match that ends inside a block is refused before any block
/// is taken: the sequence would append to that block under its other holders.
pub(crate) fn adopt_prefix_match(
    seq: &mut SequenceState,
    prefix_match: &PrefixMatch,
    kv_cache: &mut PagedKvCache,
) -> Result<()> {
    ensure!(
        prefix_match.matched_tokens == prefix_match.matched_blocks.len() * kv_cache.block_size(),
        "prefix match of {} tokens over {} blocks does not cover whole blocks",
        prefix_match.matched_tokens,
        prefix_match.matched_blocks.len()
    );
    seq.cached_prefix_tokens = prefix_match.matched_tokens;
    seq.cached_prefix_blocks = prefix_match.matched_blocks.len();
    for &block in &prefix_match.matched_blocks {
        kv_cache.inc_ref(block);
        seq.block_table.push(block);
    }
    reuse_prefix_match_disk_ids(
        &prefix_match.matched_disk_block_ids,
        &mut seq.disk_block_ids,
    );
    Ok(())
}

/// Bring this rank's match down to the length every rank agreed on (the
/// minimum of the ranks' whole-block matches). A rank that matched more
/// releases exactly what its lookup acquired and looks the agreed prefix up
/// again, so it holds the agreed blocks and nothing deeper.
pub(crate) fn cap_prefix_match(
    cache: &dyn PrefixCache,
    tokens: &[u32],
    block_size: usize,
    session_hash: u64,
    adapter_id: u64,
    local: PrefixMatch,
    agreed: usize,
) -> PrefixMatch {
    if agreed >= local.matched_tokens {
        return local;
    }
    // Releasing the whole prompt instead would also drop the cache's own
    // reference on deeper nodes that this lookup stopped short of.
    cache.release_matched(tokens, block_size, local.matched_tokens, adapter_id);
    if agreed == 0 {
        return PrefixMatch::empty();
    }
    cache.lookup_whole_blocks(&tokens[..agreed], block_size, session_hash, adapter_id)
}

#[cfg(test)]
#[path = "prefix_share_tests.rs"]
mod tests;
