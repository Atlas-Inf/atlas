// SPDX-License-Identifier: AGPL-3.0-only

//! What `cache_sequence` does with a finished sequence in a multi-rank
//! world: the decision, the rank asymmetry the head-only insert caused, and
//! the real entry point. Then the radix references a sequence gives back
//! when it is freed.

// The real model and recording layer of `prefill_stream_tests`.
#[allow(clippy::duplicate_mod)]
#[path = "prefill_stream_test_fixture.rs"]
mod fixture;

use super::{FinishCache, finish_cache, held_tokens};
use crate::traits::Model;
use fixture::*;
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::radix_tree::RadixTree;

const BS: usize = 16;

#[test]
fn a_multi_rank_world_caches_a_finished_sequence_on_every_rank_or_none() {
    assert_eq!(finish_cache(false, false), FinishCache::Local);
    assert_eq!(finish_cache(true, false), FinishCache::Skip);
    assert_eq!(finish_cache(true, true), FinishCache::Mirrored);
    // The flag needs no second rank to cache through its own path.
    assert_eq!(finish_cache(false, true), FinishCache::Mirrored);
}

/// Three turns of one conversation on a head and a worker, with physical
/// block ids that say which turn wrote them (turn `t` allocates from
/// `t * 1000`). `head_finish` is base: the head alone inserts the whole
/// sequence when a turn finishes. Returns the blocks each rank matches for
/// the third turn.
fn third_turn_blocks(head_finish: bool) -> [Vec<u32>; 2] {
    let prompts = [1_000usize, 1_560, 2_100];
    let outputs = [504usize, 500];
    let tokens: Vec<u32> = (0..prompts[2] as u32).collect();
    let caches = [RadixTree::new(), RadixTree::new()];
    for (rank, cache) in caches.iter().enumerate() {
        for turn in 0..2 {
            let prompt = &tokens[..prompts[turn]];
            let m = cache.lookup(prompt, BS, 0, 0);
            // The sequence's table: the matched blocks, then its own.
            let mut table = m.matched_blocks.clone();
            let end = prompts[turn] + outputs[turn];
            table.extend((table.len()..end / BS + 1).map(|i| (turn as u32 + 1) * 1_000 + i as u32));
            // Prefill publishes the prompt's whole blocks on both ranks.
            cache.insert(prompt, &table, &[], BS, m.matched_tokens, 0);
            if head_finish && rank == 0 {
                cache.insert(&tokens[..end], &table, &[], BS, prompts[turn], 0);
            }
            cache.release(&tokens[..end], BS, 0);
        }
    }
    caches.map(|c| c.lookup(&tokens, BS, 0, 0).matched_blocks)
}

/// Base gave the head alone a node for each block of a turn's output. When
/// the next turn's history reproduces that output (here it does), its
/// prefill insert finds those nodes and keeps their blocks, so one turn
/// later the two ranks match the same tokens to different rows: the head to
/// the ones decode wrote, the worker to its own prefill of them. Without the
/// head-only insert both ranks hold the prefill's.
#[test]
fn a_head_only_finish_insert_made_the_ranks_cache_different_rows() {
    let [head, worker] = third_turn_blocks(true);
    // Both ranks match the second turn's whole prompt.
    assert_eq!((head.len(), worker.len()), (1_560 / BS, 1_560 / BS));
    // The first turn's output lies in blocks 62..94: the worker holds the
    // second turn's prefill of it, the head the first turn's own blocks.
    assert!(worker[62..94].iter().all(|b| (2_000..3_000).contains(b)));
    assert!(head[62..94].iter().all(|b| (1_000..2_000).contains(b)));

    let [head, worker] = third_turn_blocks(false);
    assert_eq!(head, worker);
    assert!(head[62..94].iter().all(|b| (2_000..3_000).contains(b)));
}

/// A finished turn on the real `cache_sequence`, which only the head calls:
/// a single rank publishes the output's whole blocks; a multi-rank head
/// (without the finish leaf, which mirrors the insert on the worker)
/// publishes nothing, so both ranks keep holding the same nodes.
#[test]
fn actual_multi_rank_head_does_not_cache_a_finished_sequence_alone() {
    let tokens: Vec<u32> = (1..=28).collect();
    for (tp, ep, nodes) in [(1, 1, 7), (2, 2, 6), (2, 1, 6)] {
        let mut f = Fixture::with_tail_split(tp, ep, 0);
        f.disable_capture();
        for (start, len, last) in [(0, 8, false), (8, 16, true)] {
            f.model
                .prefill_chunk(&tokens[..24], &mut f.seq, start, len, last, CALLER)
                .unwrap();
        }
        assert_eq!(f.model.prefix_cache.stats().0, 6, "TP{tp}/EP{ep}");
        // Four decoded tokens fill the sequence's seventh block.
        let block = f.model.kv_cache.lock().alloc_block().unwrap();
        f.seq.block_table.push(block);
        f.seq.tokens.extend_from_slice(&tokens[24..]);
        f.seq.seq_len = 28;
        f.model.cache_sequence(&f.seq);
        assert_eq!(f.model.prefix_cache.stats().0, nodes, "TP{tp}/EP{ep}");
    }
}

#[test]
fn a_sequence_gives_back_the_references_it_took() {
    // Cached when it finished: every whole block of prompt and output.
    assert_eq!(held_tokens(FinishCache::Local, 24, 28), 28);
    assert_eq!(held_tokens(FinishCache::Mirrored, 24, 28), 28);
    // Not cached: its prompt, or what it got to of it.
    assert_eq!(held_tokens(FinishCache::Skip, 24, 28), 24);
    assert_eq!(held_tokens(FinishCache::Skip, 24, 16), 16);
    // Resumed from a swap file without a prefill: it holds none.
    assert_eq!(held_tokens(FinishCache::Skip, 0, 28), 0);
}

const PROMPT: usize = 1_000;
const OUTPUT: usize = 504;
/// The second turn's prompt: the first, its output, a new message.
const NEXT: usize = 1_560;

/// A conversation's second turn, cached; then the first turn again (a retry
/// or a duplicate: the same prompt, and at temperature 0 the same output).
/// It took references over its prompt only. Giving back `released` tokens'
/// worth, returns how much of the second turn's prompt still matches.
fn match_after_a_duplicate_turn(released: usize) -> usize {
    let tokens: Vec<u32> = (0..NEXT as u32).collect();
    let cache = RadixTree::new();
    let turn = |len: usize, first_block: u32| {
        let m = cache.lookup(&tokens[..len], BS, 0, 0);
        let mut table = m.matched_blocks.clone();
        table.extend((table.len()..len / BS + 1).map(|i| first_block + i as u32));
        cache.insert(&tokens[..len], &table, &[], BS, m.matched_tokens, 0);
    };
    turn(PROMPT, 1_000);
    cache.release(&tokens[..PROMPT], BS, 0);
    turn(NEXT, 2_000);
    cache.release(&tokens, BS, 0);
    assert_eq!(
        cache.lookup(&tokens, BS, 0, 0).matched_tokens,
        NEXT / BS * BS
    );
    cache.release(&tokens, BS, 0);
    // The duplicate: matches its whole prompt, decodes the same output.
    turn(PROMPT, 3_000);
    cache.release(&tokens[..released], BS, 0);
    cache.lookup(&tokens, BS, 0, 0).matched_tokens
}

/// Releasing over prompt and output, as a rank that did not cache the
/// output used to, takes the cache's own reference from the nodes the second
/// turn put on that output: the conversation's next turn then matches only
/// up to the duplicate's prompt. Releasing over the prompt keeps the match.
#[test]
fn a_duplicate_turn_released_over_its_output_cut_the_conversations_match() {
    assert_eq!(
        match_after_a_duplicate_turn(PROMPT + OUTPUT),
        PROMPT / BS * BS
    );
    assert_eq!(match_after_a_duplicate_turn(PROMPT), NEXT / BS * BS);
}

/// The real `free_sequence` on each rank of a TP2 pair and on a single rank:
/// a finished 24-token turn whose four decoded tokens another prompt had
/// already published (the seventh block) leaves that node matchable.
#[test]
fn actual_free_keeps_the_nodes_another_prompt_put_on_the_output() {
    let tokens: Vec<u32> = (1..=28).collect();
    for (tp, ep, rank) in [(2, 2, 0), (2, 2, 1), (2, 1, 0), (1, 1, 0)] {
        let ctx = format!("TP{tp}/EP{ep}/rank{rank}");
        let mut f = Fixture::with_tail_split(tp, ep, rank);
        f.disable_capture();
        let mut seq = std::mem::replace(&mut f.seq, crate::traits::SequenceState::host_only(0));
        for (start, len, last) in [(0, 8, false), (8, 16, true)] {
            f.model
                .prefill_chunk(&tokens[..24], &mut seq, start, len, last, CALLER)
                .unwrap();
        }
        // The conversation's next turn, whose prompt covers this turn's
        // output: it matches the 24 tokens, publishes a seventh block and
        // is gone.
        let m = f.model.prefix_cache.lookup(&tokens, 4, 0, 0);
        assert_eq!(m.matched_tokens, 24, "{ctx}");
        let table = [&m.matched_blocks[..], &[99]].concat();
        f.model.prefix_cache.insert(&tokens, &table, &[], 4, 24, 0);
        f.model.prefix_cache.release(&tokens, 4, 0);
        // This turn decodes the same four tokens and finishes.
        let block = f.model.kv_cache.lock().alloc_block().unwrap();
        seq.block_table.push(block);
        seq.tokens.extend_from_slice(&tokens[24..]);
        seq.seq_len = 28;
        if rank == 0 {
            f.model.cache_sequence(&seq);
        }
        f.model.free_sequence(&mut seq).unwrap();
        let after = f.model.prefix_cache.lookup(&tokens, 4, 0, 0);
        assert_eq!(after.matched_tokens, 28, "{ctx}");
    }
}
