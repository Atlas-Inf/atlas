// SPDX-License-Identifier: AGPL-3.0-only

//! The rolling finish leaf: where it lands, what it copies, who holds its
//! slot, and that two ranks place and restore it alike.

use super::super::super::ssm_batched_copy::run_ssm_state_copies;
use super::super::super::ssm_pool::SsmStatePool;
use super::super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::prefill_b::pc_policy::{Agreed, agree_restore, tail_cut};
use super::flag::resolve;
use super::rolling::{LeafSave, boundary_row};
use super::*;
use crate::ssm_reserve::SsmRollbackMode;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::radix_tree::RadixTree;

const BS: usize = 16;

#[test]
fn the_flag_needs_every_precondition() {
    let needs = |a, b, c| [("SUBBLOCK=0", a), ("PREFILL_ONLY=1", b), ("PC_EVICT=1", c)];
    assert!(resolve(true, &needs(true, true, true)));
    for one_missing in [
        needs(false, true, true),
        needs(true, false, true),
        needs(true, true, false),
    ] {
        assert!(!resolve(true, &one_missing));
    }
    assert!(!resolve(false, &needs(true, true, true)));
    assert!(!resolve(false, &needs(false, false, false)));
}

#[test]
fn boundary_row_is_the_last_boundary_the_step_reaches() {
    // A 5-row step from 45 commits 46..=50: row 3 lands on 48.
    assert_eq!(boundary_row(45, 5, BS), Some(3));
    // Ending exactly on the boundary: the whole step.
    assert_eq!(boundary_row(45, 3, BS), Some(3));
    assert_eq!(boundary_row(47, 1, BS), Some(1));
    // Starting on a boundary does not count: that state is already behind.
    assert_eq!(boundary_row(48, 5, BS), None);
    assert_eq!(boundary_row(48, 16, BS), Some(16));
    assert_eq!(boundary_row(45, 2, BS), None);
    // A step wider than a block keeps the deepest boundary.
    assert_eq!(boundary_row(45, 40, BS), Some(35));
    // A coarser span skips the block boundaries in between.
    assert_eq!(boundary_row(45, 5, 4 * BS), None);
    assert_eq!(boundary_row(60, 9, 4 * BS), Some(4));
    assert_eq!(boundary_row(45, 5, 0), None);
}

/// Steps of a decode: `(rows, plain)`, a verify commit of 1..=9 rows or one
/// plain decode token, from a small LCG so the mix is fixed.
fn steps(seed: u64, n: usize) -> Vec<(usize, bool)> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let plain = (x >> 40).is_multiple_of(5);
            (if plain { 1 } else { 1 + (x >> 33) as usize % 9 }, plain)
        })
        .collect()
}

/// The leaf position a sequence of `len` tokens holds after `steps`, through
/// the plan both hooks execute ([`leaf_save`]): a commit of `rows` of the 9
/// verified rows with the sequence already rolled back, or a plain decode.
fn run(mut len: usize, steps: &[(usize, bool)], span: usize) -> (usize, Option<usize>) {
    let mut leaf = None;
    for &(rows, plain) in steps {
        let (pre, k) = (len, if plain { 1 } else { 9 });
        len += rows;
        if let Some(save) = leaf_save(len, rows, k, span) {
            assert!(save.at.is_multiple_of(span) && save.at > pre && save.at <= len);
            assert_eq!(save.at - pre, save.rows);
            // The boundary's conv state: the verify's snapshot after that
            // row (rows 0..k - 1 have one), or the live state after row k.
            let conv = (save.rows < k).then(|| save.rows - 1);
            assert_eq!(save.conv_row, conv, "rows={rows} k={k}");
            leaf = Some(save.at);
        }
    }
    (len, leaf)
}

/// At finish the leaf is the last save boundary at or below the final
/// committed token, whatever the step widths were.
#[test]
fn the_leaf_ends_on_the_last_boundary_at_or_below_the_end() {
    for seed in 0..200u64 {
        for span in [BS, 4 * BS] {
            let start = 300 + seed as usize % 37;
            let (end, leaf) = run(start, &steps(seed, 3 + seed as usize % 90), span);
            let want = end / span * span;
            assert_eq!(
                leaf,
                (want > start).then_some(want),
                "seed={seed} span={span}"
            );
        }
    }
}

/// `leaf_save` takes the length after the rejected rows were rolled back. A
/// 9-row verify from 60 that accepts 5 rows ends at 65 and crosses 64 at its
/// row 4. Planned before the rollback (69), the same commit would label the
/// state after row 4 as token 68.
#[test]
fn the_plan_needs_the_rolled_back_length() {
    let save = |end| leaf_save(end, 5, 9, 4);
    assert_eq!(
        save(65),
        Some(LeafSave {
            rows: 4,
            at: 64,
            conv_row: Some(3)
        })
    );
    assert_eq!(save(69).map(|s| (s.rows, s.at)), Some((4, 68)));
    // No step to plan for: fewer tokens than rows.
    assert_eq!(leaf_save(3, 5, 9, 4), None);
    // A full accept that ends on the boundary saves the live conv state.
    assert_eq!(leaf_save(64, 9, 9, BS).unwrap().conv_row, None);
    assert_eq!(leaf_save(64, 1, 1, BS).unwrap().conv_row, None);
    // A partial accept that ends there reads the snapshot the rewind reads.
    assert_eq!(leaf_save(64, 5, 9, BS).unwrap().conv_row, Some(4));
}

/// The rolling slot is registered from its first save and moved boundary by
/// boundary, so the index can hand it to a checkpoint that needs a slot at
/// any time. The sequence asks for it back before each save.
#[test]
fn the_rolling_leaf_is_the_indexes_to_evict() {
    let cache = RadixTree::new();
    let tokens: Vec<u32> = (0..2_000).collect();
    assert_eq!(cache.insert_leaf_snapshot(&tokens[..1_024], 5, 0, 0), None);
    // The next boundary: the sequence still holds the slot and moves it.
    assert!(cache.take_leaf_snapshot(&tokens[..1_024], 5, 0));
    assert_eq!(cache.snapshot_count(), 0);
    assert_eq!(cache.insert_leaf_snapshot(&tokens[..1_088], 5, 0, 0), None);
    // At finish the turn reports the leaf it still has there.
    assert_eq!(cache.snapshot_at(&tokens, 1_088, 0), Some(5));
    // A checkpoint save takes the slot; the sequence finds its leaf gone
    // and must not write the slot again.
    assert_eq!(cache.evict_snapshot_lru(), Some(5));
    assert!(!cache.take_leaf_snapshot(&tokens[..1_088], 5, 0));
    assert_eq!(cache.snapshot_at(&tokens, 1_088, 0), None);
    // A rewind that regenerated the text under the leaf: not this prefix.
    assert_eq!(cache.insert_leaf_snapshot(&tokens[..1_088], 6, 0, 0), None);
    let mut other = tokens.clone();
    other[1_000] = 9;
    assert!(!cache.take_leaf_snapshot(&other[..1_088], 6, 0));
    assert!(cache.take_leaf_snapshot(&tokens[..1_088], 6, 0));
}

fn tiny_config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.linear_num_key_heads = 2;
    c.linear_key_head_dim = 4;
    c.linear_num_value_heads = 2;
    c.linear_value_head_dim = 4;
    c.linear_conv_kernel_dim = 4;
    c
}

/// The save takes h from the live state and conv from the verify snapshot
/// after the boundary row (the live conv then holds all `k` rows), or the
/// live conv when the step ends on the boundary. A restore returns both.
#[test]
fn the_save_pairs_live_h_with_the_boundary_conv() {
    const SEQ: usize = 1;
    const SNAP: usize = 2;
    let gpu = MockGpuBackend::new();
    let pool = SsmStatePool::new(
        &tiny_config(),
        4,
        true,
        4,
        3,
        false,
        SsmRollbackMode::Snapshot,
        false,
        &gpu,
    )
    .unwrap();
    let layers = pool.num_ssm_layers;
    let snaps =
        SsmSnapshotPool::new(3, pool.h_bytes, pool.conv_bytes, layers, 0, 0, 8, &gpu).unwrap();
    let (h_live, conv_live, conv_row) = (
        |l: usize| 10 + l as u8,
        |l: usize| 80 + l as u8,
        |l, r| (150 + 8 * l + r) as u8,
    );
    for l in 0..layers {
        let put = |v: u8, n: usize, p| gpu.copy_h2d(&vec![v; n], p).unwrap();
        put(h_live(l), pool.h_stored_bytes, pool.h_state(l, SEQ));
        put(conv_live(l), pool.conv_bytes, pool.conv_state(l, SEQ));
        for r in 0..pool.num_intermediates {
            put(
                conv_row(l, r),
                pool.conv_bytes,
                pool.conv_intermediate(l, SEQ, r),
            );
        }
    }
    let saved = |row: Option<usize>| {
        let (h, conv) = leaf_copies(&pool, &snaps, SEQ, SNAP, row);
        run_ssm_state_copies(&gpu, &h, &conv, 0).unwrap();
        // Restore into another slot and read that slot's live state back.
        snaps.restore(SNAP, 3, &pool, &gpu, 0).unwrap();
        (0..layers)
            .map(|l| {
                let (mut h, mut c) = (vec![0u8; pool.h_stored_bytes], vec![0u8; pool.conv_bytes]);
                gpu.copy_d2h(pool.h_state(l, 3), &mut h).unwrap();
                gpu.copy_d2h(pool.conv_state(l, 3), &mut c).unwrap();
                assert!(h.iter().all(|&b| b == h[0]) && c.iter().all(|&b| b == c[0]));
                (h[0], c[0])
            })
            .collect::<Vec<_>>()
    };
    let want = |conv: &dyn Fn(usize) -> u8| {
        (0..layers)
            .map(|l| (h_live(l), conv(l)))
            .collect::<Vec<_>>()
    };
    assert_eq!(saved(Some(1)), want(&|l| conv_row(l, 1)));
    assert_eq!(saved(None), want(&conv_live));
}

/// One rank's prefix cache through a turn: prefill (prompt blocks and the
/// tail checkpoint), then finish (all whole blocks, and the leaf when the
/// rank kept one), as `cache_sequence` does on both ranks with the flag on.
struct Rank {
    cache: RadixTree,
    rank: usize,
}

impl Rank {
    fn turn(rank: usize, tokens: &[u32], prompt: usize, leaf: Option<usize>) -> Self {
        let cache = RadixTree::new();
        let blocks: Vec<u32> = (0..(tokens.len() / BS + 1) as u32).collect();
        let cut = tail_cut(prompt, BS);
        let acquired = cache.insert(&tokens[..prompt], &blocks[..prompt / BS], &[], BS, 0, 0);
        assert_eq!(acquired.blocks.len(), prompt / BS);
        cache.insert_intermediate_snapshot(
            &tokens[..cut],
            &blocks[..cut / BS],
            &[],
            BS,
            cut + rank,
            0,
            cut,
            0,
        );
        cache.insert(
            tokens,
            &blocks[..tokens.len() / BS],
            &[],
            BS,
            owned_from(prompt, BS),
            0,
        );
        if let Some(at) = leaf {
            let none = cache.insert_leaf_snapshot(&tokens[..at], at + rank, 0, 0);
            assert_eq!(none, None);
        }
        cache.release(tokens, BS, 0);
        Self { cache, rank }
    }

    /// `(matched, restorable depth)` for the next turn's prompt.
    fn lookup(&self, next: &[u32]) -> (usize, usize) {
        let m = self.cache.lookup(next, BS, 0, 0);
        assert_eq!(
            m.ssm_snapshot,
            (m.ssm_snapshot_tokens > 0).then(|| m.ssm_snapshot_tokens + self.rank)
        );
        (m.matched_tokens, m.ssm_snapshot_tokens)
    }
}

/// The block the prompt end falls inside is created at finish, so the
/// finishing sequence must own it: passed `prompt_len` itself (as base does),
/// the sequence's release takes that node to zero references and the next
/// turn's walk stops there, before every block of the generated output.
#[test]
fn the_block_under_the_prompt_end_is_owned_by_the_finishing_sequence() {
    let (prompt, end) = (1_000usize, 1_504usize);
    let tokens: Vec<u32> = (0..end as u32).collect();
    let blocks: Vec<u32> = (0..(end / BS) as u32).collect();
    let matched = |owned: usize| {
        let cache = RadixTree::new();
        cache.insert(&tokens[..prompt], &blocks[..prompt / BS], &[], BS, 0, 0);
        cache.insert(&tokens, &blocks, &[], BS, owned, 0);
        cache.release(&tokens, BS, 0);
        cache.peek_matched_tokens(&tokens, BS, 0)
    };
    assert_eq!(owned_from(prompt, BS), 992);
    assert_eq!(matched(owned_from(prompt, BS)), end);
    assert_eq!(
        matched(prompt),
        992,
        "base: the generated blocks are unreachable"
    );
    // A prompt that ends on a boundary has no such block either way.
    assert_eq!(owned_from(1_008, BS), 1_008);
}

/// Both ranks' [`agree_restore`] results for the next turn, over the
/// min-reductions their own lookups produce.
fn agreed(ranks: &[Rank; 2], next: &[u32]) -> [(usize, Agreed); 2] {
    let looked = [ranks[0].lookup(next), ranks[1].lookup(next)];
    assert_eq!(looked[0].0, looked[1].0, "ranks must match the same blocks");
    let depth = looked[0].1.min(looked[1].1);
    let holds =
        |i: usize| looked[i].1 == depth || ranks[i].cache.snapshot_at(next, depth, 0).is_some();
    let all = u32::from(depth > 0 && holds(0) && holds(1));
    [0, 1].map(|i| {
        let mut replies = [depth as u32, all].into_iter();
        agree_restore(
            looked[i].1,
            |_| Ok(replies.next().expect("at most two reductions")),
            |at| ranks[i].cache.snapshot_at(next, at, 0),
        )
        .unwrap()
    })
}

/// A turn decoded from the same verdicts leaves the same leaf on both ranks;
/// the next turn then matches the same blocks and restores at the leaf.
#[test]
fn both_ranks_place_and_restore_the_same_leaf() {
    for seed in 0..40u64 {
        let prompt = 1_000 + seed as usize * 13;
        let verdicts = steps(seed, 60 + seed as usize);
        // Head and worker apply the same verdict stream to mirrored sequences.
        let (end, leaf) = run(prompt, &verdicts, BS);
        assert_eq!((end, leaf), run(prompt, &verdicts, BS));
        let at = leaf.expect("a 60-step turn crosses a block boundary");
        assert_eq!(at, end / BS * BS);
        let tokens: Vec<u32> = (0..end as u32).map(|t| t * 3 + seed as u32).collect();
        let next: Vec<u32> = tokens.iter().copied().chain(900_000..900_040).collect();
        let ranks = [0, 1].map(|r| Rank::turn(r, &tokens, prompt, leaf));
        assert_eq!(ranks[0].lookup(&next), (at, at));
        assert_eq!(
            agreed(&ranks, &next),
            [(at, Agreed::Local); 2],
            "seed={seed}"
        );
    }
}

/// A rank that could not keep its leaf (snapshot pool exhausted) restores
/// from the prefill checkpoint; the agreement brings the other rank down to
/// it, so both resume at the same row.
#[test]
fn a_rank_without_the_leaf_pulls_both_to_the_tail_checkpoint() {
    let (prompt, end) = (1_000, 1_500);
    let (at, cut) = (end / BS * BS, tail_cut(prompt, BS));
    let tokens: Vec<u32> = (0..end as u32).collect();
    let next: Vec<u32> = tokens.iter().copied().chain(900_000..900_040).collect();
    let ranks = [
        Rank::turn(0, &tokens, prompt, Some(at)),
        Rank::turn(1, &tokens, prompt, None),
    ];
    assert_eq!(ranks[0].lookup(&next), (at, at));
    assert_eq!(ranks[1].lookup(&next), (at, cut));
    assert_eq!(
        agreed(&ranks, &next),
        [(cut, Agreed::At(cut)), (cut, Agreed::Local)]
    );
}

/// A next turn that diverges inside the previous output (a re-rendered
/// history) matches fewer blocks than the leaf covers and falls back to the
/// prefill checkpoint on both ranks: the leaf is never paired with a shorter
/// match.
#[test]
fn a_match_below_the_leaf_does_not_restore_it() {
    let (prompt, end) = (1_000, 1_500);
    let (at, cut) = (end / BS * BS, tail_cut(prompt, BS));
    let tokens: Vec<u32> = (0..end as u32).collect();
    let mut next: Vec<u32> = tokens.iter().copied().chain(900_000..900_040).collect();
    next[1_300] = 7;
    let ranks = [0, 1].map(|r| Rank::turn(r, &tokens, prompt, Some(at)));
    assert_eq!(ranks[0].lookup(&next), (1_296, cut));
    assert_eq!(agreed(&ranks, &next), [(cut, Agreed::Local); 2]);
}

/// The turn that restores a leaf settles it. A prompt that runs at most two
/// blocks past the leaf has its tail cut at or under it, so the prefill saves
/// no checkpoint and the leaf becomes the conversation's checkpoint (no
/// leaf's rolling slot may take it; the old checkpoint is history). A longer
/// prompt saves its own, and the leaf is the first slot to go.
#[test]
fn the_restoring_turn_keeps_or_retires_its_leaf() {
    let (prompt, end) = (1_000, 1_500);
    let (at, cut) = (end / BS * BS, tail_cut(prompt, BS));
    for extra in 1..=80 {
        assert_eq!(at >= tail_cut(at + extra, BS), extra <= 2 * BS, "{extra}");
    }
    let tokens: Vec<u32> = (0..end as u32).collect();
    for keep in [true, false] {
        let rank = Rank::turn(0, &tokens, prompt, Some(at));
        assert_eq!(rank.cache.evict_snapshot_for_leaf(), Some(at));
        let rank = Rank::turn(0, &tokens, prompt, Some(at));
        rank.cache.settle_leaf_snapshot(&tokens[..at], 0, keep);
        let first = if keep { cut } else { at };
        assert_eq!(rank.cache.evict_snapshot_for_leaf(), Some(first));
        // What is left is the conversation's frontier.
        assert_eq!(rank.cache.evict_snapshot_for_leaf(), None);
    }
}
