// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-admission capacity for the swap-out gate: the blocks a new request
//! will actually ALLOCATE, so the gate never reclaims its own cached prefix.
//!
//! The gate frees room for each new request before any live sequence is
//! swapped to disk, reclaiming from the prefix cache first. It used to size
//! that room as the whole prompt (`prompt_len / block_size + 1`). On a warm
//! server free blocks sit near zero (the cache pins what it holds), so the
//! reclaim evicted least-recently-used radix leaves — the tail of the very
//! conversation the request was about to reuse — until a whole prompt's
//! worth was free. Past half the pool the match became `pool - prompt`: on
//! one DGX Spark (Qwen3.8-27B, 349K-token pool, 2026-11-09) a 178K-token
//! agent turn matched 170K instead of ~178K, re-prefilled ~8K tokens each
//! turn (20 s TTFT), and fell to a full 317 s re-prefill once the shrinking
//! match passed every SSM snapshot. Cached blocks are handed over by the
//! lookup, never allocated, so they are subtracted here.

use spark_model::traits::Model;

/// Blocks the prompt needs allocated beyond what the prefix cache will hand
/// it (`cached_blocks`, a read-only probe). The `+ 1` keeps room for the
/// first decode token's block, as before.
pub(super) fn prompt_blocks_to_allocate(
    prompt_len: usize,
    block_size: usize,
    cached_blocks: usize,
) -> usize {
    (prompt_len / block_size + 1).saturating_sub(cached_blocks)
}

/// [`prompt_blocks_to_allocate`] for `prompt` against `model`'s prefix cache.
pub(super) fn blocks_for_admission(
    model: &dyn Model,
    prompt: &[u32],
    adapter_slot: i32,
    block_size: usize,
) -> usize {
    let cached = model.cached_prefix_blocks(prompt, adapter_slot);
    prompt_blocks_to_allocate(prompt.len(), block_size, cached)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The production shape: a 178,608-token turn whose first 178,432 tokens
    /// are cached needs 11 fresh blocks, not the 11,164 that evicted the tail
    /// of its own conversation.
    #[test]
    fn a_cached_conversation_reclaims_only_its_new_tail() {
        let cached = 178_432 / 16;
        assert_eq!(prompt_blocks_to_allocate(178_608, 16, cached), 12);
        assert_eq!(prompt_blocks_to_allocate(178_608, 16, 0), 11_164);
    }

    /// A cold prompt keeps the old reservation exactly.
    #[test]
    fn a_cold_prompt_reserves_the_whole_prompt() {
        assert_eq!(prompt_blocks_to_allocate(4_000, 16, 0), 251);
    }

    /// A probe can never ask for less than nothing.
    #[test]
    fn an_over_reported_match_saturates() {
        assert_eq!(prompt_blocks_to_allocate(32, 16, 10), 0);
    }
}
