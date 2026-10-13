// SPDX-License-Identifier: AGPL-3.0-only

//! What a finished sequence leaves in the prefix cache, and the radix
//! references it holds when it is freed.

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// What `cache_sequence` does with a finished sequence's blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum FinishCache {
    /// Every rank caches its mirror (the flag).
    Mirrored,
    /// This rank caches it: a single-rank world.
    Local,
    /// No rank does.
    Skip,
}

/// `cache_sequence` runs on the head only. In a multi-rank world without the
/// flag its insert used to give the head alone a node for every block of the
/// output. The agreed match of the next turn cannot use them (the worker
/// holds none), so the blocks sat in the head's pool until evicted. And when
/// the next turn's history reproduced the output (a tool turn, or a client
/// that sends the reasoning back), that turn's prefill insert found those
/// nodes and kept their blocks, so from the turn after that the head
/// attended the rows decode had written while the worker attended its own
/// prefill of the same positions. A multi-rank world therefore caches a
/// finished sequence on every rank or on none.
///
/// This holds for every multi-rank model, and `Skip` leaves out all of the
/// head's finish-time caching: the radix insert and, where decode
/// checkpoints are on (`ATLAS_MARCONI_PREFILL_ONLY` off), the finish
/// snapshot the head alone used to register.
pub(super) fn finish_cache(multi_rank: bool, leaf: bool) -> FinishCache {
    match (leaf, multi_rank) {
        (true, _) => FinishCache::Mirrored,
        (false, true) => FinishCache::Skip,
        (false, false) => FinishCache::Local,
    }
}

/// How many of a sequence's `tokens` tokens it holds radix references over
/// when it is freed: all of them where it was cached when it finished (the
/// insert takes one on every whole block of the output), its prompt where it
/// was not. Releasing over the output there would take a reference the
/// sequence never held from any node another request's prompt had put on the
/// same tokens (a retried or duplicated earlier turn, once the conversation
/// has moved on), and a node without references ends every later match.
pub(super) fn held_tokens(finish: FinishCache, prompt_len: usize, tokens: usize) -> usize {
    match finish {
        FinishCache::Skip => prompt_len.min(tokens),
        FinishCache::Mirrored | FinishCache::Local => tokens,
    }
}

impl TransformerModel {
    /// `cache_sequence` on the head: carry out [`finish_cache`]. `false`
    /// leaves the insert to the caller (a single-rank world).
    pub(in super::super) fn finish_cache_multi_rank(&self, seq: &SequenceState, bs: usize) -> bool {
        match finish_cache(self.multi_rank_protocol_active(), self.leaf_on()) {
            FinishCache::Mirrored => self.finish_leaf_cache(seq, bs),
            FinishCache::Skip => {}
            FinishCache::Local => return false,
        }
        true
    }

    /// The tokens whose radix nodes `free_sequence` releases.
    ///
    /// Normally a prefix of `seq.tokens` ([`held_tokens`]) covers the matched
    /// prefix, so releasing over it undoes the lookup's radix inc_refs. But a
    /// prefill that matched a prefix then FAILED to allocate its suffix never
    /// populated `seq.tokens` (that happens in a later finalize phase), so
    /// releasing over it would be a no-op and the matched radix nodes would
    /// stay pinned forever → the pool wedges. When `seq.tokens` is too short
    /// to cover the matched prefix, release over the stashed prefix tokens
    /// instead. Exactly one of the two covers the matched nodes, so they are
    /// released once (never double-released).
    pub(in super::super) fn radix_held_tokens<'a>(&self, seq: &'a SequenceState) -> &'a [u32] {
        if seq.tokens.len() < seq.cached_prefix_tokens {
            return &seq.prefix_ref_tokens;
        }
        let finish = finish_cache(self.multi_rank_protocol_active(), self.leaf_on());
        &seq.tokens[..held_tokens(finish, seq.prompt_len, seq.tokens.len())]
    }
}

#[cfg(test)]
#[path = "../finish_cache_tests.rs"]
mod tests;
