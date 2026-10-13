// SPDX-License-Identifier: AGPL-3.0-only

//! The qwen4_exp finish leaf: where it is saved, which of its two slots a
//! turn keeps, that both ranks place and restore it alike, that the
//! off-grid tail checkpoint is restored, and that a restore returns every
//! byte of the saved state.

use super::super::super::super::ssm_batched_copy::run_ssm_state_copies;
use super::super::super::super::ssm_pool::SsmStatePool;
use super::super::super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::super::prefill_b::pc_policy::{Agreed, agree_restore, tail_cut};
use super::super::rolling::{leaf_copies, owned_from};
use super::*;
use crate::ssm_reserve::SsmRollbackMode;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{GpuBackend, mock::MockGpuBackend};
use spark_runtime::prefix_cache::PrefixCache;
use spark_runtime::radix_tree::RadixTree;

const BS: usize = 16;
const SPAN: usize = 4 * BS;
const MIN: usize = 256;

#[test]
fn a_step_saves_at_its_end_when_it_crossed_a_boundary() {
    // An 8-row verify from 60 that keeps 7 rows ends at 67 and crossed 64.
    assert_eq!(step_end_save(67, 7, SPAN), Some(67));
    // Ending exactly on it, by a verify or a plain decode step.
    assert_eq!(step_end_save(64, 3, SPAN), Some(64));
    assert_eq!(step_end_save(64, 1, SPAN), Some(64));
    // Starting on it: that state is already behind.
    assert_eq!(step_end_save(68, 4, SPAN), None);
    assert_eq!(step_end_save(63, 8, SPAN), None);
    assert_eq!(step_end_save(3, 5, SPAN), None, "fewer tokens than rows");
}

/// One rank's leaves through a decode: `(rows)` per step, a verify commit of
/// 1..=8 rows (the exact lane's 7 drafts) or a plain decode step.
#[derive(Default, PartialEq, Debug)]
struct Leaves {
    cur: Option<FinishLeaf>,
    prev: Option<FinishLeaf>,
    saves: Vec<usize>,
    next_slot: usize,
}

impl Leaves {
    /// What `qwen4exp_leaf_plan` and `qwen4exp_leaf_save` do with a step.
    fn step(&mut self, len: usize, rows: usize) {
        let plan = plan_step(self.cur, len, rows, (SPAN, BS, MIN));
        if plan.retire_prev {
            self.prev = None;
        }
        if let Some(at) = plan.save_at {
            assert_eq!(at, len, "the leaf is the state the step left");
            let snap = self.prev.take().map_or_else(
                || {
                    self.next_slot += 1;
                    self.next_slot
                },
                |p| p.snap,
            );
            self.prev = self.cur.take();
            self.cur = Some(FinishLeaf { snap, tokens: at });
            self.saves.push(at);
        }
        // The one before is kept only while the current one is unreachable
        // (or was saved this step).
        if self.prev.is_some() {
            let cur = self.cur.unwrap();
            assert!(!reachable(cur.tokens, len, BS) || plan.save_at.is_some());
        }
    }
}

fn steps(seed: u64, n: usize) -> Vec<usize> {
    let mut x = seed;
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            1 + (x >> 33) as usize % 8
        })
        .collect()
}

/// At finish the turn keeps the deepest save its cached blocks reach, which
/// is at most a span and a block (plus a step) behind its end, and frees
/// the other slot.
#[test]
fn a_turn_keeps_its_deepest_reachable_leaf() {
    for seed in 0..300u64 {
        let start = 300 + seed as usize * 7 % 211;
        let mut len = start;
        let mut leaves = Leaves::default();
        for rows in steps(seed, 2 + seed as usize % 70) {
            len += rows;
            leaves.step(len, rows);
        }
        let (keep, free) = finish_pick(leaves.cur, leaves.prev, len, BS);
        let whole = len / BS * BS;
        let want = leaves.saves.iter().copied().filter(|&s| s <= whole).max();
        assert_eq!(keep.map(|l| l.tokens), want, "seed={seed} len={len}");
        if let Some(k) = keep {
            assert!(len - k.tokens < SPAN + BS + 8, "seed={seed}");
        }
        let held = [leaves.cur, leaves.prev].iter().flatten().count();
        assert_eq!(free.len() + usize::from(keep.is_some()), held);
    }
}

/// A turn that ends inside its leaf's block falls back to the leaf before,
/// not to the previous turn's tail checkpoint.
#[test]
fn a_turn_that_ends_in_its_leafs_block_keeps_the_one_before() {
    let mut leaves = Leaves::default();
    let mut len = 1_000;
    for rows in [8; 9] {
        len += rows;
        leaves.step(len, rows);
    }
    // 1,008 .. 1,072 by 8: the step to 1,024 ends on the boundary.
    assert_eq!((len, leaves.saves.clone()), (1_072, vec![1_024]));
    len += 7;
    leaves.step(len, 7); // 1,079
    len += 8;
    leaves.step(len, 8); // 1,087: still below 1,088
    len += 3;
    leaves.step(len, 3); // 1,090 crossed 1,088: save at 1,090
    assert_eq!(leaves.saves, vec![1_024, 1_090]);
    let (keep, free) = finish_pick(leaves.cur, leaves.prev, len, BS);
    assert_eq!(keep.map(|l| l.tokens), Some(1_024));
    assert_eq!(free.iter().map(|l| l.tokens).collect::<Vec<_>>(), [1_090]);
    // Once the block of 1,090 is complete the one before is retired.
    len += 8;
    leaves.step(len, 8);
    len += 8;
    leaves.step(len, 8); // 1,106 >= 1,104
    assert_eq!(leaves.prev, None);
    let (keep, _) = finish_pick(leaves.cur, leaves.prev, len, BS);
    assert_eq!(keep.map(|l| l.tokens), Some(1_090));
}

/// One rank's prefix cache through a turn: the prompt blocks, the prefill
/// checkpoint at `ckpt`, then finish (whole blocks of the turn, and the
/// leaf it kept), as both ranks run it with the switch.
fn turn(rank: usize, tokens: &[u32], prompt: usize, ckpt: usize, leaf: Option<usize>) -> RadixTree {
    let cache = RadixTree::new();
    let blocks: Vec<u32> = (0..(tokens.len() / BS + 1) as u32).collect();
    cache.insert(&tokens[..prompt], &blocks[..prompt / BS], &[], BS, 0, 0);
    let ckpt_blocks = &blocks[..ckpt / BS];
    cache.insert_intermediate_snapshot(
        &tokens[..ckpt],
        ckpt_blocks,
        &[],
        BS,
        100 + rank,
        0,
        ckpt,
        0,
    );
    let whole = &blocks[..tokens.len() / BS];
    cache.insert(tokens, whole, &[], BS, owned_from(prompt, BS), 0);
    if let Some(at) = leaf {
        assert_eq!(
            cache.insert_leaf_snapshot(&tokens[..at], 200 + rank, 0, 0),
            None
        );
    }
    cache.release(tokens, BS, 0);
    cache
}

/// Both ranks' `(matched, agreed depth, what each restores)` for `next`.
fn next_turn(ranks: &[RadixTree; 2], next: &[u32]) -> (usize, [(usize, Agreed); 2]) {
    let looked = ranks.each_ref().map(|c| {
        let m = c.lookup(next, BS, 0, 0);
        (m.matched_tokens, m.ssm_snapshot_tokens)
    });
    assert_eq!(looked[0].0, looked[1].0, "ranks must match the same blocks");
    let depth = looked[0].1.min(looked[1].1);
    let holds = |i: usize| looked[i].1 == depth || ranks[i].snapshot_at(next, depth, 0).is_some();
    let all = u32::from(depth > 0 && holds(0) && holds(1));
    let agreed = [0, 1].map(|i| {
        let mut replies = [depth as u32, all].into_iter();
        let min = |_| Ok(replies.next().expect("at most two reductions"));
        agree_restore(looked[i].1, min, |at| ranks[i].snapshot_at(next, at, 0)).unwrap()
    });
    (looked[0].0, agreed)
}

/// A leaf off the block grid (the end of an 8-row verify) is restored by
/// the next turn on both ranks when that turn reproduces the output past
/// its block, and the agreement keeps them at one depth.
#[test]
fn both_ranks_restore_a_leaf_off_the_block_grid() {
    let (prompt, end, at) = (2_000usize, 2_100usize, 2_052usize);
    let tokens: Vec<u32> = (0..end as u32).map(|t| t * 5 + 1).collect();
    let next: Vec<u32> = tokens.iter().copied().chain(900_000..900_030).collect();
    let cut = tail_cut(prompt, BS);
    let ranks = [0, 1].map(|r| turn(r, &tokens, prompt, cut, Some(at)));
    let (matched, agreed) = next_turn(&ranks, &next);
    assert_eq!(matched, end / BS * BS);
    assert_eq!(agreed, [(at, Agreed::Local); 2]);
    // A rank whose leaf was evicted pulls both to the prefill checkpoint.
    let ranks = [
        turn(0, &tokens, prompt, cut, Some(at)),
        turn(1, &tokens, prompt, cut, None),
    ];
    assert_eq!(
        next_turn(&ranks, &next).1,
        // Rank 0 restores its own checkpoint (snapshot id 100) at the agreed depth.
        [(cut, Agreed::At(100)), (cut, Agreed::Local)]
    );
}

/// The needle prompt: 77,463 tokens, its in-pass checkpoint off the block
/// grid at 77,380. The identical prompt and the follow-up (the same 77,463
/// tokens, then 38 more) match 77,456 tokens and restore it on both ranks.
#[test]
fn the_off_grid_tail_checkpoint_serves_the_next_turn() {
    let (prompt, cp) = (77_463usize, 77_380usize);
    let tokens: Vec<u32> = (0..prompt as u32).map(|t| t % 50_000 + 7).collect();
    let ranks = [0, 1].map(|r| turn(r, &tokens, prompt, cp, None));
    let follow: Vec<u32> = tokens.iter().copied().chain(800_000..800_038).collect();
    for next in [&tokens, &follow] {
        let (matched, agreed) = next_turn(&ranks, next);
        assert_eq!(matched, 77_456);
        assert_eq!(agreed, [(cp, Agreed::Local); 2]);
    }
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

/// The leaf's SSM half: every byte of every layer's live h and conv state
/// comes back on a restore into another sequence's slot, and the slot's aux
/// blobs (PLE, QSA) are handed to the restore as attached.
#[test]
fn a_restored_leaf_is_the_saved_state_byte_for_byte() {
    const SEQ: usize = 1;
    const SNAP: usize = 2;
    const OTHER: usize = 3;
    let gpu = MockGpuBackend::new();
    let cfg = tiny_config();
    let pool = SsmStatePool::new(
        &cfg,
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
    let pattern = |seed: usize, n: usize| -> Vec<u8> {
        (0..n)
            .map(|i| ((i * 131 + seed * 7919) % 251) as u8)
            .collect()
    };
    let read = |p, n| {
        let mut v = vec![0u8; n];
        gpu.copy_d2h(p, &mut v).unwrap();
        v
    };
    for l in 0..layers {
        gpu.copy_h2d(&pattern(l, pool.h_stored_bytes), pool.h_state(l, SEQ))
            .unwrap();
        gpu.copy_h2d(&pattern(100 + l, pool.conv_bytes), pool.conv_state(l, SEQ))
            .unwrap();
    }
    let (h, conv) = leaf_copies(&pool, &snaps, SEQ, SNAP, None);
    run_ssm_state_copies(&gpu, &h, &conv, 0).unwrap();
    let blobs = vec![(1u32, pattern(7, 977)), (5, pattern(8, 4_112))];
    snaps.set_aux(SNAP, blobs.clone());
    snaps.restore(SNAP, OTHER, &pool, &gpu, 0).unwrap();
    for l in 0..layers {
        let h_bytes = pool.h_stored_bytes;
        assert_eq!(read(pool.h_state(l, OTHER), h_bytes), pattern(l, h_bytes));
        let c_bytes = pool.conv_bytes;
        assert_eq!(
            read(pool.conv_state(l, OTHER), c_bytes),
            pattern(100 + l, c_bytes)
        );
    }
    let seen = snaps.with_aux(SNAP, |b| Ok(b.to_vec())).unwrap().unwrap();
    assert_eq!(seen, blobs);
}
