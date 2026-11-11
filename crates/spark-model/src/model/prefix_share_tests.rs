// SPDX-License-Identifier: AGPL-3.0-only

//! Sequences sharing one prefix cache, replayed on the host through the
//! production lookup, adoption, block allocation, cache insert and release.
//! Every KV row write checks that the block holds an index tail, that no
//! other sequence wrote it and that no other sequence holds it. Every read
//! checks that the row was computed for the reader's own token prefix.

use std::collections::{HashMap, HashSet};

use super::{adopt_prefix_match, cap_prefix_match};
use crate::model::block_mgmt::{
    apply_evicted_blocks, cache_acquires_refs, ensure_blocks_through_decode,
    ensure_blocks_through_prefill,
};
use crate::traits::SequenceState;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{
    KvCacheConfig, KvCacheDtype, PagedKvCache, SparseIndexCacheConfig, TailSlotPlan,
};
use spark_runtime::prefix_cache::{InsertAcquired, PrefixCache, PrefixMatch};
use spark_runtime::radix_tree::RadixTree;

#[path = "prefix_share_guard_tests.rs"]
mod guard;
#[path = "prefix_share_pair_tests.rs"]
mod pair;

const BS: usize = 16;
const BLOCKS: usize = 96;
/// A prompt that ends five tokens into its third block.
const PROMPT: usize = 2 * BS + 5;

fn toks(range: std::ops::Range<u32>) -> Vec<u32> {
    range.collect()
}

fn join(parts: &[&[u32]]) -> Vec<u32> {
    parts.concat()
}

/// Stands in for the KV a row holds: a function of the whole token prefix.
fn prefix_hash(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325, |h, &t| {
        (h ^ u64::from(t)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// One rank: its KV pool, its prefix cache and what every row holds.
struct World {
    gpu: MockGpuBackend,
    kv: PagedKvCache,
    cache: RadixTree,
    /// `(block, row) -> prefix_hash` of the tokens the row was written for.
    rows: HashMap<(u32, usize), u64>,
    /// The sequence that has written each block since it was allocated.
    writers: HashMap<u32, usize>,
    /// Blocks the prefix cache holds its reference on.
    published: HashSet<u32>,
    sequences: usize,
}

impl World {
    /// A GLM-style cache with slotted index tails.
    fn new() -> Self {
        let gpu = MockGpuBackend::new();
        let config = KvCacheConfig {
            block_size: BS,
            num_kv_heads: 1,
            head_dim: 64,
            num_layers: 1,
            dtype: KvCacheDtype::Bf16,
            layer_dtypes: vec![],
            layer_dims: vec![],
            cache_blocks_per_seq: None,
        };
        let mut kv = PagedKvCache::new(config, BLOCKS, &gpu).unwrap();
        let plan = TailSlotPlan {
            lag_blocks: 2,
            sequences: 4,
        };
        kv.attach_sparse_index_with_tail_slots(
            SparseIndexCacheConfig::bf16(4, 128),
            Some(plan),
            &gpu,
        )
        .unwrap();
        Self {
            gpu,
            kv,
            cache: RadixTree::new(),
            rows: HashMap::new(),
            writers: HashMap::new(),
            published: HashSet::new(),
            sequences: 0,
        }
    }

    /// Write the row for `fed[pos]` (computed over `fed[..=pos]`).
    fn write_row(&mut self, seq: &SequenceState, fed: &[u32], pos: usize) {
        let (block, me) = (seq.block_table[pos / BS], seq.slot_idx);
        let at = format!("sequence {me} writes position {pos} into block {block}");
        assert!(
            !self.kv.tail_slot_missing(block),
            "{at}, which holds no index tail"
        );
        let first = *self.writers.entry(block).or_insert(me);
        assert_eq!(first, me, "{at}, which sequence {first} wrote");
        let cache_ref = u32::from(self.published.contains(&block));
        assert_eq!(
            self.kv.ref_count(block) - cache_ref,
            1,
            "{at}, which another sequence holds"
        );
        self.rows
            .insert((block, pos % BS), prefix_hash(&fed[..=pos]));
    }

    /// Attention reads every committed row of the sequence.
    fn read_rows(&self, seq: &SequenceState) {
        for pos in 0..seq.seq_len {
            let block = seq.block_table[pos / BS];
            assert_eq!(
                self.rows.get(&(block, pos % BS)),
                Some(&prefix_hash(&seq.tokens[..=pos])),
                "sequence {} reads row {pos} (block {block}), which was not computed for its \
                 tokens",
                seq.slot_idx
            );
        }
    }

    /// The cache takes its references on the blocks an insert published.
    fn publish(&mut self, acquired: InsertAcquired) {
        cache_acquires_refs(&acquired, &mut self.kv);
        self.published.extend(acquired.blocks);
    }

    /// Blocks `seq` allocated past its first `had` come from the pool: no
    /// writer yet, and no longer the cache's if they were evicted from it.
    fn allocated(&mut self, seq: &SequenceState, had: usize) {
        for block in &seq.block_table[had..] {
            self.writers.remove(block);
            self.published.remove(block);
        }
    }

    /// Chunk-0 lookup through the end-of-prefill cache insert.
    fn prefill(&mut self, prompt: &[u32]) -> SequenceState {
        let local = self.cache.lookup_whole_blocks(prompt, BS, 0, 0);
        self.prefill_matched(prompt, local)
    }

    /// Adoption of `matched` through the end-of-prefill cache insert.
    fn prefill_matched(&mut self, prompt: &[u32], matched: PrefixMatch) -> SequenceState {
        let mut seq = SequenceState::host_only(self.sequences);
        self.sequences += 1;
        adopt_prefix_match(&mut seq, &matched, &mut self.kv).unwrap();
        seq.prompt_len = prompt.len();
        let (adopted, last) = (seq.block_table.len(), (prompt.len() - 1) / BS);
        ensure_blocks_through_prefill(
            &mut seq,
            last,
            &mut self.kv,
            &self.cache,
            &self.gpu,
            0,
            false,
        )
        .unwrap();
        self.allocated(&seq, adopted);
        for pos in matched.matched_tokens..prompt.len() {
            self.write_row(&seq, prompt, pos);
        }
        seq.tokens = prompt.to_vec();
        seq.seq_len = prompt.len();
        self.read_rows(&seq);
        let acquired = self.cache.insert(
            prompt,
            &seq.block_table,
            &[],
            BS,
            seq.cached_prefix_tokens,
            0,
        );
        self.publish(acquired);
        seq
    }

    fn grow(&mut self, seq: &mut SequenceState, pos: usize) {
        let had = seq.block_table.len();
        ensure_blocks_through_decode(
            seq,
            pos / BS,
            &mut self.kv,
            &self.cache,
            &self.gpu,
            0,
            false,
        )
        .unwrap();
        self.allocated(seq, had);
    }

    fn decode(&mut self, seq: &mut SequenceState, tokens: &[u32]) {
        for &token in tokens {
            self.read_rows(seq);
            let pos = seq.seq_len;
            self.grow(seq, pos);
            seq.tokens.push(token);
            self.write_row(seq, &seq.tokens, pos);
            seq.seq_len += 1;
        }
    }

    /// A speculative verify: writes one row per draft, rejected ones included,
    /// then commits the first `accepted`.
    fn verify(&mut self, seq: &mut SequenceState, drafts: &[u32], accepted: usize) {
        self.read_rows(seq);
        let base = seq.seq_len;
        let fed = join(&[&seq.tokens, drafts]);
        for t in 0..drafts.len() {
            self.grow(seq, base + t);
            self.write_row(seq, &fed, base + t);
        }
        seq.tokens.extend_from_slice(&drafts[..accepted]);
        seq.seq_len = base + accepted;
    }

    /// `free_sequence`: what a worker rank does with a finished sequence.
    fn free(&mut self, seq: SequenceState) {
        self.read_rows(&seq);
        self.cache.release(&seq.tokens, BS, 0);
        self.kv.free_blocks(&seq.block_table);
    }

    /// `cache_sequence` then `free_sequence`: the head rank.
    fn retire(&mut self, seq: SequenceState) {
        if seq.tokens.len() >= BS {
            let acquired =
                self.cache
                    .insert(&seq.tokens, &seq.block_table, &[], BS, seq.prompt_len, 0);
            self.publish(acquired);
        }
        self.free(seq);
    }

    /// With every sequence gone the whole cache is evictable and every block
    /// returns to the pool: no reference was taken twice or left behind.
    fn drain(&mut self) {
        loop {
            let evicted = self.cache.evict(BLOCKS);
            if evicted.is_empty() {
                break;
            }
            apply_evicted_blocks(evicted, &mut self.kv, &self.cache, &self.gpu);
        }
        assert_eq!(self.cache.stats().0, 0, "cache nodes left unevictable");
        assert_eq!(self.kv.num_free_blocks(), BLOCKS, "KV blocks leaked");
    }
}

/// The block a sequence still appends to carries that sequence's reference
/// only, from the end of prefill until the block is full and the sequence
/// has retired.
#[test]
fn the_cache_takes_no_reference_on_a_block_its_sequence_still_writes() {
    let mut w = World::new();
    let mut seq = w.prefill(&toks(0..PROMPT as u32));
    assert_eq!(w.kv.ref_count(seq.block_table[2]), 1);
    assert_eq!(w.published.len(), 2);
    w.decode(&mut seq, &toks(500..530));
    assert_eq!(w.published.len(), 2);
    let table = seq.block_table.clone();
    w.retire(seq);
    // Four whole blocks of committed tokens are cached; the fifth, partly
    // filled, went back to the pool with its sequence.
    assert_eq!(
        [2, 3, 4].map(|b| w.kv.ref_count(table[b])),
        [1, 1, 0],
        "{table:?}"
    );
    w.drain();
}

/// Live donor: the same prompt arrives again while its first sequence is
/// still decoding. Both then decode, and either may finish first.
#[test]
fn live_donor_and_its_twin_write_their_own_blocks() {
    let prompt = toks(0..PROMPT as u32);
    for twin_leaves_first in [true, false] {
        let mut w = World::new();
        let mut donor = w.prefill(&prompt);
        let mut twin = w.prefill(&prompt);
        w.decode(&mut donor, &[500]);
        w.verify(&mut twin, &[600, 601, 602], 1);
        w.decode(&mut donor, &[501, 502]);
        w.verify(&mut twin, &toks(610..630), 14);
        // The twin reused the two whole prompt blocks and nothing else.
        assert_eq!(twin.cached_prefix_tokens, 2 * BS);
        assert_eq!(twin.block_table[..2], donor.block_table[..2]);
        assert_ne!(twin.block_table[2], donor.block_table[2]);

        let (leaver, mut stayer) = if twin_leaves_first {
            (twin, donor)
        } else {
            (donor, twin)
        };
        w.retire(leaver);
        // The stayer's frontier block kept its index tail and its rows.
        w.decode(&mut stayer, &toks(700..740));
        w.retire(stayer);
        w.drain();
    }
}

/// Retired donor: the choices of a blocking `n > 1` request run one after
/// another on one prompt, each after the previous one retired. A choice that
/// stops inside the prompt's last block leaves that block partly filled.
#[test]
fn serial_choices_of_one_prompt_do_not_touch_each_other() {
    let prompt = toks(0..PROMPT as u32);
    let mut w = World::new();
    let mut first = w.prefill(&prompt);
    let answer = toks(500..506);
    w.decode(&mut first, &answer);
    w.retire(first);

    // Choices of 5, 8, 17 and 20 tokens: inside the block and past it.
    for choice in 1..5u32 {
        let mut seq = w.prefill(&prompt);
        w.verify(&mut seq, &toks(choice * 1000..choice * 1000 + 8), 3);
        w.decode(
            &mut seq,
            &toks(choice * 2000..choice * 2000 + 2 + 3 * (choice % 2)),
        );
        if choice > 2 {
            w.decode(&mut seq, &toks(choice * 3000..choice * 3000 + 12));
        }
        assert_eq!(seq.cached_prefix_tokens, 2 * BS, "choice {choice}");
        w.retire(seq);
    }

    // The first choice's conversation continues and reads its own rows.
    let turn2 = join(&[&prompt, &answer, &toks(900..920)]);
    let mut seq = w.prefill(&turn2);
    assert_eq!(seq.cached_prefix_tokens, 2 * BS);
    w.decode(&mut seq, &toks(950..960));
    w.retire(seq);
    w.drain();
}

/// Strict-prefix retry: turn 1 is sent again after turn 2, a longer prompt
/// with the same start, was cached. The retry's prompt ends inside a block
/// that holds turn-2 rows; the conversation's next turn must still read them.
#[test]
fn strict_prefix_retry_leaves_the_longer_prompt_intact() {
    let turn1 = toks(0..PROMPT as u32);
    let answer1 = toks(500..520);
    let turn2 = join(&[&turn1, &answer1, &toks(900..930)]);
    let answer2 = toks(600..625);
    for turn2_is_live in [false, true] {
        let mut w = World::new();
        let mut seq = w.prefill(&turn1);
        w.decode(&mut seq, &answer1);
        w.retire(seq);
        let mut conversation = w.prefill(&turn2);

        let retry_now = |w: &mut World| {
            let mut retry = w.prefill(&turn1);
            w.verify(&mut retry, &toks(7000..7008), 2);
            w.decode(&mut retry, &toks(7100..7140));
            assert_eq!(retry.cached_prefix_tokens, 2 * BS);
            w.retire(retry);
        };
        if turn2_is_live {
            retry_now(&mut w);
        }
        w.decode(&mut conversation, &answer2);
        w.retire(conversation);
        if !turn2_is_live {
            retry_now(&mut w);
        }

        let turn3 = join(&[&turn2, &answer2, &toks(950..970)]);
        let mut seq = w.prefill(&turn3);
        assert_eq!(seq.cached_prefix_tokens, turn2.len() / BS * BS);
        w.decode(&mut seq, &toks(980..990));
        w.retire(seq);
        w.drain();
    }
}

/// The chain: while A decodes, B sends A's prompt and runs one verify whose
/// rejected drafts land on positions A already committed. B is cancelled,
/// before or after A finishes. A's retire caches the third block. A's turn 2
/// recomputes that span into a fresh block, but its insert finds the node and
/// the cache keeps the block from turn 1. A's turn 3 matches through the
/// node and reads it: every row must still be one A wrote.
#[test]
fn chain_through_a_cached_node_reads_only_rows_written_for_it() {
    let prompt = toks(0..PROMPT as u32);
    let answer = toks(500..520);
    for b_leaves_first in [true, false] {
        let mut w = World::new();
        let mut a1 = w.prefill(&prompt);
        w.decode(&mut a1, &answer[..4]);

        let mut b = w.prefill(&prompt);
        w.verify(&mut b, &[answer[0], 8001, 8002, 8003], 1);
        assert_ne!(b.block_table[2], a1.block_table[2]);
        let mut b = Some(b);
        if b_leaves_first {
            w.retire(b.take().unwrap());
        }
        w.decode(&mut a1, &answer[4..]);
        let a1_third_block = a1.block_table[2];
        w.retire(a1);
        b.into_iter().for_each(|b| w.retire(b));

        let turn2 = join(&[&prompt, &answer, &toks(900..925)]);
        let answer2 = toks(600..630);
        let mut a2 = w.prefill(&turn2);
        assert_eq!(a2.cached_prefix_tokens, 2 * BS);
        assert_ne!(a2.block_table[2], a1_third_block);
        w.decode(&mut a2, &answer2);
        w.retire(a2);

        let turn3 = join(&[&turn2, &answer2, &toks(950..975)]);
        let mut a3 = w.prefill(&turn3);
        assert_eq!(a3.cached_prefix_tokens, turn2.len() / BS * BS);
        assert_eq!(a3.block_table[2], a1_third_block);
        w.decode(&mut a3, &toks(980..990));
        w.retire(a3);
        w.drain();
    }
}

/// A prompt that ends on a block boundary is matched in full; the twins
/// still decode into separate blocks.
#[test]
fn block_aligned_prompts_share_every_prompt_block() {
    let prompt = toks(0..(2 * BS) as u32);
    let mut w = World::new();
    let mut donor = w.prefill(&prompt);
    let mut twin = w.prefill(&prompt);
    assert_eq!(twin.cached_prefix_tokens, 2 * BS);
    assert_eq!(twin.block_table, donor.block_table);
    w.decode(&mut donor, &[500, 501]);
    w.verify(&mut twin, &[600, 601, 602], 2);
    assert_ne!(twin.block_table[2], donor.block_table[2]);
    w.retire(donor);
    w.decode(&mut twin, &toks(700..720));
    w.retire(twin);
    w.drain();
}
