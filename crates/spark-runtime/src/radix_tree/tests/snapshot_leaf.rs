// SPDX-License-Identifier: AGPL-3.0-only

//! Finish leaves in the snapshot index: they never supersede or displace a
//! conversation's frontier checkpoint. Like the chain tests, these drive the
//! index directly (the flags are process-wide `OnceLock`s).

use super::*;
use crate::radix_tree::snapshot::{SnapLoc, SnapshotEntry};

const POOL: usize = 16;

/// One conversation's token path: a shared system prompt, then its own turns.
fn convo(session: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| if i < 2048 { i } else { session * 1_000_000 + i })
        .collect()
}

fn hash(tokens: &[u32]) -> u64 {
    hash_token_prefix(tokens, tokens.len(), 0)
}

/// A prefill checkpoint for `tokens`, as `RadixTree` registers one.
fn checkpoint(idx: &mut SsmSnapshotIndex, tokens: &[u32], id: usize) -> Option<usize> {
    let old = idx.insert(hash(tokens), id, 0, tokens.len());
    idx.link_chain(tokens, 0, hash(tokens));
    old
}

fn leaf(idx: &mut SsmSnapshotIndex, tokens: &[u32], id: usize) -> Option<usize> {
    idx.insert_leaf(hash(tokens), id, 0, tokens.len())
}

fn entry(idx: &SsmSnapshotIndex, id: usize) -> &SnapshotEntry {
    idx.entries.iter().find(|e| e.snapshot_id == id).unwrap()
}

fn resident(idx: &SsmSnapshotIndex, id: usize) -> bool {
    idx.entries.iter().any(|e| e.snapshot_id == id)
}

fn evict(idx: &mut SsmSnapshotIndex) -> usize {
    let v = idx.chain_victim(true).unwrap();
    idx.entries.swap_remove(v).snapshot_id
}

#[test]
fn a_leaf_does_not_supersede_the_checkpoint_under_it() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 31_000);
    checkpoint(&mut idx, &t[..30_000], 1);
    assert_eq!(leaf(&mut idx, &t[..30_400], 2), None);
    let (cut, l) = (entry(&idx, 1).chain, entry(&idx, 2).chain);
    assert!(!cut.superseded && !cut.branch, "still the frontier");
    assert!(l.leaf && l.id == 0 && !l.superseded);
    // The next turn's checkpoint links past the leaf, to the checkpoint.
    checkpoint(&mut idx, &t, 3);
    let (cut, l, next) = (
        entry(&idx, 1).chain,
        entry(&idx, 2).chain,
        entry(&idx, 3).chain,
    );
    assert!(cut.superseded && !cut.branch, "no fork at the checkpoint");
    assert_eq!(cut.id, next.id, "one conversation, one chain");
    assert!(l.leaf && l.superseded, "the leaf has served its turn");
}

#[test]
fn victims_are_dead_history_then_leaves_then_frontiers() {
    let mut idx = SsmSnapshotIndex::new();
    let (a, b) = (convo(1, 32_000), convo(2, 40_000));
    checkpoint(&mut idx, &b, 100); // B's frontier: the oldest entry of all
    checkpoint(&mut idx, &a[..30_000], 1);
    checkpoint(&mut idx, &a[..31_000], 2); // supersedes 1
    leaf(&mut idx, &a[..31_400], 3);
    let order: Vec<usize> = (0..4).map(|_| evict(&mut idx)).collect();
    assert_eq!(order, [1, 3, 100, 2]);
}

#[test]
fn a_leaf_never_takes_a_frontier_slot() {
    let mut idx = SsmSnapshotIndex::new();
    let a = convo(1, 32_000);
    checkpoint(&mut idx, &convo(2, 40_000), 100);
    checkpoint(&mut idx, &a[..30_000], 1);
    assert_eq!(idx.evict_for_leaf(), None, "two frontiers, nothing to take");
    checkpoint(&mut idx, &a[..31_000], 2);
    leaf(&mut idx, &a[..31_400], 3);
    assert_eq!(idx.evict_for_leaf(), Some(1), "dead history first");
    assert_eq!(idx.evict_for_leaf(), Some(3), "then another leaf");
    assert_eq!(idx.evict_for_leaf(), None);
    assert!(resident(&idx, 100) && resident(&idx, 2));
    // A spilled entry holds no slot to take.
    leaf(&mut idx, &a[..31_400], 4);
    idx.entries.iter_mut().for_each(|e| e.tiered = e.chain.leaf);
    assert_eq!(idx.evict_for_leaf(), None);
}

#[test]
fn a_checkpoint_and_a_leaf_at_one_prefix() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 30_000);
    // A checkpoint already there wins: the leaf's slot comes straight back.
    checkpoint(&mut idx, &t, 1);
    assert_eq!(leaf(&mut idx, &t, 2), Some(2));
    assert!(!entry(&idx, 1).chain.leaf && idx.entries.len() == 1);
    // A newer leaf replaces an older one and stays a leaf.
    let u = convo(3, 30_000);
    assert_eq!(leaf(&mut idx, &u, 5), None);
    assert_eq!(leaf(&mut idx, &u, 6), Some(5));
    assert!(entry(&idx, 6).chain.leaf);
    // A checkpoint saved over a leaf takes its place and joins a chain.
    assert_eq!(checkpoint(&mut idx, &u, 7), Some(6));
    let c = entry(&idx, 7).chain;
    assert!(!c.leaf && c.id != 0);
}

#[test]
fn the_restoring_turn_settles_its_leaf() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 31_000);
    checkpoint(&mut idx, &t[..30_000], 1);
    leaf(&mut idx, &t[..30_400], 2);
    // Not a leaf, or nothing there: no effect either way.
    for keep in [true, false] {
        idx.settle_leaf(&t[..30_000], 0, keep);
        idx.settle_leaf(&t[..30_416], 0, keep);
    }
    let (cut, l) = (entry(&idx, 1).chain, entry(&idx, 2).chain);
    assert!(!cut.superseded && l.leaf && !l.superseded);
    // The turn saves its own checkpoint: the leaf is the first victim, and
    // still restorable until it goes.
    idx.settle_leaf(&t[..30_400], 0, false);
    assert!(entry(&idx, 2).chain.superseded && !entry(&idx, 1).chain.superseded);
    assert_eq!(idx.resident_at(&t, 30_400, 0), Some(2));
    assert_eq!(idx.evict_for_leaf(), Some(2));
    // The turn saves none: the leaf becomes the frontier.
    leaf(&mut idx, &t[..30_400], 3);
    idx.settle_leaf(&t[..30_400], 0, true);
    let (cut, l) = (entry(&idx, 1).chain, entry(&idx, 3).chain);
    assert!(cut.superseded && !l.leaf && !l.superseded);
    assert_eq!(cut.id, l.id);
    assert_eq!(evict(&mut idx), 1, "the old checkpoint is dead history now");
}

#[test]
fn a_rolling_leaf_moves_only_while_the_sequence_still_holds_it() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 31_000);
    checkpoint(&mut idx, &t[..30_000], 1);
    leaf(&mut idx, &t[..30_080], 2);
    // Not this slot, not this prefix: nothing to take.
    assert!(!idx.take_leaf(hash(&t[..30_080]), 9));
    assert!(!idx.take_leaf(hash(&t[..30_144]), 2));
    // The sequence moves its leaf to the next boundary.
    assert!(idx.take_leaf(hash(&t[..30_080]), 2));
    assert!(!resident(&idx, 2));
    leaf(&mut idx, &t[..30_144], 2);
    // Another writer takes the slot: the sequence finds its leaf gone.
    assert_eq!(idx.evict_for_leaf(), Some(2));
    assert!(!idx.take_leaf(hash(&t[..30_144]), 2));
    // Promoted by the turn that restored it: a checkpoint is not the
    // sequence's to move. Settled as history: still its own.
    leaf(&mut idx, &t[..30_144], 3);
    idx.settle_leaf(&t[..30_144], 0, true);
    assert!(!idx.take_leaf(hash(&t[..30_144]), 3) && resident(&idx, 3));
    leaf(&mut idx, &t[..30_208], 4);
    idx.settle_leaf(&t[..30_208], 0, false);
    assert!(idx.take_leaf(hash(&t[..30_208]), 4));
    // Never a checkpoint, whatever slot it names.
    assert!(!idx.take_leaf(hash(&t[..30_000]), 1) && resident(&idx, 1));
}

/// A label for a checkpoint: `(conversation, turn)`.
type Label = (u32, usize);

/// `turns` rounds of `convs` conversations on one 16-slot pool, `wave` of
/// them decoding at a time. Each turn saves its prefill checkpoint and, with
/// `leaves`, rolls a finish leaf over four boundaries through the rules a
/// decoding sequence follows (take its own leaf back or find a slot that is
/// free, dead or another leaf). Returns the protected checkpoints (frontiers
/// and branch points) after every checkpoint save, and how many turns after
/// the first round found the conversation's leaf.
fn serve(convs: u32, turns: usize, leaves: bool, wave: usize) -> (Vec<Vec<Label>>, usize) {
    let mut idx = SsmSnapshotIndex::new();
    let (mut next_id, mut hits) = (0usize, 0usize);
    let mut labels = std::collections::HashMap::new();
    let mut protected = Vec::new();
    let mut last_leaf = vec![usize::MAX; convs as usize];
    let order: Vec<u32> = (0..convs).collect();
    for turn in 0..turns {
        for group in order.chunks(wave) {
            let prompt = |c: u32| convo(c + 1, 4_000 + turn * 800);
            for &c in group {
                let tokens = prompt(c);
                // The restore bumps the winner, as a lookup does, and settles
                // the leaf it took (this turn saves a checkpoint). Nothing to
                // restore: a cold turn (the first, or a lost conversation).
                if let Some(hit) = idx.lookup_tiered(&tokens, tokens.len(), 0, 0) {
                    hits += usize::from(hit.loc == SnapLoc::Hbm(last_leaf[c as usize]));
                    idx.settle_leaf(&tokens[..hit.token_count], 0, false);
                }
                // The prefill checkpoint: a free slot or the chain victim.
                if idx.entries.len() == POOL {
                    evict(&mut idx);
                }
                next_id += 1;
                labels.insert(next_id, (c, turn));
                checkpoint(&mut idx, &tokens[..tokens.len() - 32], next_id);
                let mut now: Vec<Label> = idx
                    .entries
                    .iter()
                    .filter(|e| e.chain.class() > DEAD + 1)
                    .map(|e| labels[&e.snapshot_id])
                    .collect();
                now.sort_unstable();
                protected.push(now);
            }
            // The group decodes together: each crosses four save boundaries.
            let mut held = vec![None; group.len()];
            for roll in 1..=4 {
                for (g, &c) in group.iter().enumerate().filter(|_| leaves) {
                    let end = prompt(c).len() + 96 * roll;
                    let out = convo(c + 1, end);
                    let mine = held[g].filter(|&(at, id)| idx.take_leaf(hash(&out[..at]), id));
                    let slot = match mine {
                        Some((_, id)) => Some(id),
                        None if idx.entries.len() < POOL => {
                            next_id += 1;
                            Some(next_id)
                        }
                        None => idx.evict_for_leaf(),
                    };
                    held[g] = slot.map(|id| (end, id));
                    if let Some(id) = slot {
                        assert_eq!(leaf(&mut idx, &out, id), None);
                        last_leaf[c as usize] = id;
                    }
                    assert!(idx.entries.len() <= POOL);
                }
            }
        }
    }
    (protected, hits)
}

/// The pool-pressure contract. With leaves on, the protected checkpoints are
/// exactly the ones the pool holds with leaves off, after every checkpoint
/// save, for any number of conversations and any number decoding at once:
/// a leaf only ever sits in a slot that is otherwise free or dead history.
///
/// Leaves live in that slack. A checkpoint and a leaf per conversation fit
/// the pool up to 8 conversations, and every leaf is then there for its
/// turn. Beyond that a strict rotation is the worst case for the leaves (the
/// least recently used one belongs to the conversation whose turn is next).
#[test]
fn leaves_never_cost_a_conversation_its_checkpoint() {
    for convs in [3, 8, 9, 12, 15, 20] {
        for wave in [1, 4] {
            let (on, hits) = serve(convs, 4, true, wave);
            let (off, _) = serve(convs, 4, false, wave);
            assert_eq!(on, off, "convs={convs} wave={wave}");
            let warm_turns = 3 * convs as usize;
            match convs {
                3 | 8 => assert_eq!(hits, warm_turns, "convs={convs} wave={wave}"),
                _ => assert!(hits < warm_turns, "convs={convs} wave={wave}: {hits}"),
            }
        }
    }
}
