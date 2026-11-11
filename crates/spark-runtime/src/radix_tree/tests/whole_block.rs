// SPDX-License-Identifier: AGPL-3.0-only

//! The cache shares whole blocks only.
//!
//! A sequence writes the block that holds its next position (decode appends
//! one row, a speculative verify writes several and may reject some). A
//! lookup that ended inside a cached block handed that block to a sequence
//! that then wrote the rest of it, under whoever else held it. So a match
//! never ends inside a block, and a block is published only when every
//! position in it is a committed token.

use crate::prefix_cache::PrefixCache;
use crate::radix_tree::RadixTree;

const BS: usize = 16;
/// A prompt that ends five tokens into its third block.
const PROMPT: usize = 2 * BS + 5;

fn toks(range: std::ops::Range<u32>) -> Vec<u32> {
    range.collect()
}

/// One block per started block of `len` tokens, numbered from `first`.
fn table(first: u32, len: usize) -> Vec<u32> {
    (first..first + len.div_ceil(BS) as u32).collect()
}

fn assert_whole_blocks(tree: &RadixTree, query: &[u32]) -> Vec<u32> {
    let m = tree.lookup(query, BS, 0, 0);
    assert_eq!(
        m.matched_tokens % BS,
        0,
        "a {}-token lookup ended inside a block ({} matched)",
        query.len(),
        m.matched_tokens
    );
    assert_eq!(m.matched_blocks.len(), m.matched_tokens / BS);
    assert_eq!(
        tree.peek_matched_tokens(query, BS, 0),
        m.matched_tokens,
        "the read-only probe must agree with the lookup"
    );
    tree.release_matched(query, BS, m.matched_tokens, 0);
    // The acquiring lookup offers only the query's whole blocks: same match.
    let whole = tree.lookup_whole_blocks(query, BS, 0, 0);
    assert_eq!(whole.matched_blocks, m.matched_blocks);
    tree.release_matched(query, BS, whole.matched_tokens, 0);
    m.matched_blocks
}

/// No cached length and no lookup length ever produces a match that ends
/// inside a block, and the block past the last whole one is never handed out.
/// `ATLAS_PREFIX_SUBBLOCK` is not read: this holds with it unset or set.
#[test]
fn a_match_never_ends_inside_a_block() {
    for cached in 1..=5 * BS {
        let tree = RadixTree::new();
        let key = toks(0..cached as u32);
        let blocks = table(100, cached);
        tree.insert(&key, &blocks, &[], BS, 0, 0);
        for query in 1..=6 * BS {
            let got = assert_whole_blocks(&tree, &toks(0..query as u32));
            let whole = cached.min(query) / BS;
            assert_eq!(got, blocks[..whole], "cached {cached}, lookup {query}");
        }
        // A lookup that diverges inside a cached block stops before it.
        for fork in 0..cached {
            let mut query = key.clone();
            query[fork] = u32::MAX;
            let got = assert_whole_blocks(&tree, &query);
            assert_eq!(got, blocks[..fork / BS], "cached {cached}, fork {fork}");
        }
    }
}

/// A prompt's partly filled last block is not published: the inserting
/// sequence still decodes into it, and any later holder would write it too.
#[test]
fn the_frontier_block_is_not_published() {
    let tree = RadixTree::new();
    let acquired = tree.insert(&toks(0..PROMPT as u32), &[10, 11, 12], &[], BS, 0, 0);
    assert_eq!(acquired.blocks, vec![10, 11]);
    assert_eq!(tree.stats(), (2, 2));
    // With HSS its disk slot is not taken either.
    let tree = RadixTree::new();
    let hss = tree.insert(&toks(0..PROMPT as u32), &[10, 11, 12], &[7, 8, 9], BS, 0, 0);
    assert_eq!(hss.blocks, vec![10, 11]);
    assert_eq!(hss.disk_block_ids, vec![7, 8]);
    // Eviction returns exactly the published blocks.
    tree.release(&toks(0..PROMPT as u32), BS, 0);
    let mut evicted = tree.evict(8).physical;
    evicted.sort_unstable();
    assert_eq!(evicted, vec![10, 11]);
}

/// Live donor: a second request with the same prompt arrives while the first
/// is still decoding. It shares the two whole prompt blocks and nothing else.
#[test]
fn live_donor_keeps_its_frontier_block() {
    let tree = RadixTree::new();
    let prompt = toks(0..PROMPT as u32);
    tree.insert(&prompt, &[10, 11, 12], &[], BS, 0, 0);
    assert_eq!(assert_whole_blocks(&tree, &prompt), vec![10, 11]);
}

/// Retired donor: choices of a blocking `n > 1` request run one after another
/// on one prompt. Choice 1 stopped six tokens past the prompt, inside the
/// prompt's last block, and retired; choice 2's prompt ends inside that block.
#[test]
fn retired_donor_block_is_not_reused_by_the_next_choice() {
    let tree = RadixTree::new();
    let prompt = toks(0..PROMPT as u32);
    tree.insert(&prompt, &[10, 11, 12], &[], BS, 0, 0);
    let finished = toks(0..(PROMPT + 6) as u32);
    let acquired = tree.insert(&finished, &[10, 11, 12], &[], BS, PROMPT, 0);
    tree.release(&finished, BS, 0);
    assert_eq!(assert_whole_blocks(&tree, &prompt), vec![10, 11]);
    // Block 12 was still partly filled at retire, so it never entered the
    // cache and went back to the pool with its sequence.
    assert!(acquired.blocks.is_empty(), "{acquired:?}");
    // A choice that ran past the block fills and publishes it; the next
    // choice's prompt still ends inside it and stops before it.
    let longer = toks(0..(3 * BS + 9) as u32);
    let acquired = tree.insert(&longer, &[10, 11, 22, 23], &[], BS, PROMPT, 0);
    assert_eq!(acquired.blocks, vec![22]);
    assert_eq!(assert_whole_blocks(&tree, &prompt), vec![10, 11]);
}

/// Strict-prefix retry: turn 1 is sent again after turn 2 (a longer prompt
/// with the same start) was cached. The retry ends inside a turn-2 block.
#[test]
fn strict_prefix_retry_stops_before_the_longer_prompts_block() {
    let tree = RadixTree::new();
    let turn2 = toks(0..(4 * BS) as u32);
    tree.insert(&turn2, &[10, 11, 12, 13], &[], BS, 0, 0);
    assert_eq!(
        assert_whole_blocks(&tree, &turn2[..PROMPT]),
        vec![10, 11],
        "the retry must not take block 12, which holds turn-2 rows past it"
    );
}

/// The chain: B sends A's prompt while A is still decoding, and leaves. A
/// retires, which caches its third block, then sends turn 2 and turn 3. B
/// never held A's third block, so the node that A's turn 2 finds already
/// cached, and that turn 3 then matches, holds a block only A wrote.
#[test]
fn chain_keeps_the_original_block_through_node_exists() {
    let tree = RadixTree::new();
    let prompt = toks(0..PROMPT as u32);
    // A turn 1: prefilled and decoding.
    tree.insert(&prompt, &[10, 11, 12], &[], BS, 0, 0);

    // B takes the two whole prompt blocks, not the block A is writing.
    assert_eq!(assert_whole_blocks(&tree, &prompt), vec![10, 11]);
    let b = tree.lookup(&prompt, BS, 0, 0);
    let b_table = [10, 11, 20];
    let acquired = tree.insert(&prompt, &b_table, &[], BS, b.matched_tokens, 0);
    assert!(acquired.blocks.is_empty(), "{acquired:?}");
    tree.release(&prompt, BS, 0);

    // A retires 25 tokens past its prompt: block 12 is whole and cached now.
    let a1 = toks(0..(3 * BS + 9) as u32);
    let acquired = tree.insert(&a1, &[10, 11, 12, 13], &[], BS, PROMPT, 0);
    assert_eq!(acquired.blocks, vec![12]);
    tree.release(&a1, BS, 0);

    // A turn 2 recomputes from the end of its first prompt's whole blocks
    // into fresh blocks. Its insert finds the third chunk's node and keeps
    // that node's block.
    let a2 = toks(0..(5 * BS + 3) as u32);
    let m = tree.lookup(&a2, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![10, 11]);
    let acquired = tree.insert(&a2, &[10, 11, 30, 31, 32, 33], &[], BS, 2 * BS, 0);
    assert_eq!(acquired.blocks, vec![31, 32]);
    tree.release(&a2, BS, 0);

    // A turn 3 matches past it.
    let a3 = toks(0..(7 * BS) as u32);
    assert_eq!(assert_whole_blocks(&tree, &a3), vec![10, 11, 12, 31, 32]);
}

/// Two ranks whose caches diverged agree on the smaller match. The rank that
/// matched more releases and looks the agreed prefix up again, landing on the
/// same whole blocks with balanced references.
#[test]
fn rank_min_relookup_lands_on_the_agreed_whole_blocks() {
    let query = toks(0..(3 * BS + 5) as u32);
    let deep = RadixTree::new();
    deep.insert(&toks(0..(4 * BS) as u32), &[10, 11, 12, 13], &[], BS, 0, 0);
    deep.release(&toks(0..(4 * BS) as u32), BS, 0);
    let shallow = RadixTree::new();
    shallow.insert(&toks(0..(2 * BS) as u32), &[10, 11], &[], BS, 0, 0);
    shallow.release(&toks(0..(2 * BS) as u32), BS, 0);

    let local = [
        deep.lookup_whole_blocks(&query, BS, 0, 0),
        shallow.lookup_whole_blocks(&query, BS, 0, 0),
    ];
    assert_eq!(local[0].matched_tokens, 3 * BS);
    assert_eq!(local[1].matched_tokens, 2 * BS);
    let agreed = local[0].matched_tokens.min(local[1].matched_tokens);
    deep.release_matched(&query, BS, local[0].matched_tokens, 0);
    let again = deep.lookup_whole_blocks(&query[..agreed], BS, 0, 0);
    assert_eq!(again.matched_tokens, agreed);
    assert_eq!(again.matched_blocks, local[1].matched_blocks);

    // The sequence holds exactly the agreed blocks: the rest of the deeper
    // cache is evictable now, the agreed blocks once the sequence releases.
    assert_eq!(deep.evict(8).physical, vec![13, 12]);
    deep.release(&query[..agreed], BS, 0);
    assert_eq!(deep.evict(8).physical, vec![11, 10]);
}
