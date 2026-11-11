// SPDX-License-Identifier: AGPL-3.0-only

//! Restore-ladder tests: a long agentic conversation in a small snapshot
//! pool, then a client rewrite a few thousand tokens behind the tail.

use super::super::hash_token_prefix;
use super::super::snapshot::SsmSnapshotIndex;
use super::super::snapshot_ladder::ssm_snap_ladder_base;

const SESSION: u64 = 0xc27f;
const POOL: usize = 16;

/// Deterministic token stream; `salt` changes every token from `from` on.
fn tokens(len: usize, from: usize, salt: u32) -> Vec<u32> {
    (0..len)
        .map(|i| {
            let base = (i as u32).wrapping_mul(2_654_435_761) >> 7;
            if i >= from { base ^ salt } else { base }
        })
        .collect()
}

/// Save a snapshot at `tokens.len()` the way a warm turn does, evicting
/// with the serving victim choice whenever the pool is full.
fn save(idx: &mut SsmSnapshotIndex, toks: &[u32], next_id: &mut usize) {
    while idx.entries.len() >= POOL {
        idx.evict_lru().expect("a full pool always yields a victim");
    }
    let hash = hash_token_prefix(toks, toks.len(), 0);
    idx.insert(hash, *next_id, SESSION, toks.len());
    idx.stamp_ladder(toks, SESSION, 0);
    *next_id += 1;
}

/// Grow one conversation from 8K to 178K tokens, ~600 tokens per turn —
/// the OMP shape that pushed every restore point into the last few turns.
fn grown_conversation() -> (SsmSnapshotIndex, Vec<u32>, usize) {
    let mut idx = SsmSnapshotIndex::new();
    let mut id = 0;
    let full = tokens(178_000, usize::MAX, 0);
    let (mut len, mut tail) = (8_000, 0);
    while len <= full.len() {
        save(&mut idx, &full[..len], &mut id);
        tail = len;
        len += 600;
    }
    (idx, full, tail)
}

/// The production failure: the client rewrote history ~8K tokens behind the
/// tail. The KV radix still matches up to the rewrite, so the deepest
/// resident snapshot at or below it must exist and be close — not a fall
/// back to recomputing all 170K tokens.
#[test]
fn rewrite_behind_tail_finds_a_nearby_anchor() {
    if ssm_snap_ladder_base().is_none() {
        return; // ladder disabled by env in this process
    }
    let (mut idx, _, tail) = grown_conversation();
    let base = ssm_snap_ladder_base().unwrap();
    for back in [1_000usize, 3_000, 8_300, 20_000, 60_000] {
        let diverge = tail - back;
        let rewritten = tokens(tail + 300, diverge, 0x5a5a);
        let hit = idx
            .lookup_tiered(&rewritten, diverge, SESSION, 0)
            .unwrap_or_else(|| panic!("no anchor at all {back} tokens behind the tail"));
        let replay = diverge - hit.token_count;
        // Bucket k holds a rung somewhere in [b·2^k, b·2^(k+1)) unless it
        // is refilling, so the replay is a small multiple of the rewrite.
        assert!(
            replay <= 4 * back.max(base) + 600,
            "rewrite {back} back: anchor {} replays {replay} tokens",
            hit.token_count
        );
    }
}

/// The ladder stays a handful of slots: most of the pool still turns over
/// by recency, so the newest restore points are always resident.
#[test]
fn ladder_leaves_recent_turns_resident() {
    if ssm_snap_ladder_base().is_none() {
        return;
    }
    let (idx, _, tail) = grown_conversation();
    let depths: Vec<usize> = idx.entries.iter().map(|e| e.token_count).collect();
    assert!(
        depths.contains(&tail),
        "the newest save must be resident: {depths:?}"
    );
    assert!(
        depths.contains(&(tail - 600)),
        "the previous turn's save must be resident: {depths:?}"
    );
    let protected = idx
        .entries
        .iter()
        .filter(|e| idx.ladder.holds(e.prefix_hash))
        .count();
    assert!(
        protected < POOL,
        "ladder must not pin the whole pool ({protected}/{POOL})"
    );
}

/// Rungs are chosen on the live path only: entries of a dead branch (an
/// earlier rewrite) are not protected, so they age out by recency.
#[test]
fn dead_branch_is_not_a_rung() {
    if ssm_snap_ladder_base().is_none() {
        return;
    }
    let mut idx = SsmSnapshotIndex::new();
    let mut id = 0;
    let live = tokens(40_000, usize::MAX, 0);
    let dead = tokens(40_000, 20_000, 0x77);
    save(&mut idx, &dead[..30_000], &mut id);
    save(&mut idx, &live[..30_000], &mut id);
    save(&mut idx, &live[..40_000], &mut id);
    let dead_hash = hash_token_prefix(&dead, 30_000, 0);
    let live_hash = hash_token_prefix(&live, 30_000, 0);
    assert!(
        !idx.ladder.holds(dead_hash),
        "dead branch entry must not be a rung"
    );
    assert!(
        idx.ladder.holds(live_hash),
        "live path entry 10K back must be a rung"
    );
}
