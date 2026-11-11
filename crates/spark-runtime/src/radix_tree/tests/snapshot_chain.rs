// SPDX-License-Identifier: AGPL-3.0-only

//! Chain-aware snapshot retention (`ATLAS_GLM_PC_EVICT`): linking, victim
//! order, the exact-depth probe, and rank determinism. The flag itself is a
//! process-wide `OnceLock`, so these drive `link_chain` / `chain_victim`
//! directly instead of mutating the environment.

use super::*;
use crate::radix_tree::snapshot::SnapshotEntry;

/// A shared system prompt followed by one conversation's turns: every session
/// starts with the same 1024+ tokens, so they all share one `session_hash`,
/// exactly the agentic-client case the default policy mishandles.
fn convo(session: u32, len: usize) -> Vec<u32> {
    (0..len as u32)
        .map(|i| if i < 2048 { i } else { session * 1_000_000 + i })
        .collect()
}

/// Register a snapshot for `tokens` the way `RadixTree` does with the flag on,
/// then evict down to `cap` resident slots with `victim`. Returns the ids
/// evicted.
fn save(
    idx: &mut SsmSnapshotIndex,
    tokens: &[u32],
    id: usize,
    session_hash: u64,
    cap: usize,
    victim: fn(&SsmSnapshotIndex) -> Option<usize>,
) -> Vec<usize> {
    let hash = hash_token_prefix(tokens, tokens.len(), 0);
    idx.insert(hash, id, session_hash, tokens.len());
    idx.link_chain(tokens, 0, hash);
    let mut evicted = Vec::new();
    while idx.entries.len() > cap {
        let v = victim(idx).expect("a victim");
        evicted.push(idx.entries.swap_remove(v).snapshot_id);
    }
    evicted
}

fn chain(idx: &SsmSnapshotIndex) -> Option<usize> {
    idx.chain_victim(true)
}

fn base(idx: &SsmSnapshotIndex) -> Option<usize> {
    idx.session_aware_victim_with_alpha(false, true, 0.0)
}

fn entry_of(idx: &SsmSnapshotIndex, id: usize) -> &SnapshotEntry {
    idx.entries.iter().find(|e| e.snapshot_id == id).unwrap()
}

fn resident(idx: &SsmSnapshotIndex, id: usize) -> bool {
    idx.entries.iter().any(|e| e.snapshot_id == id)
}

#[test]
fn growing_conversation_supersedes_its_previous_tail() {
    let mut idx = SsmSnapshotIndex::new();
    save(&mut idx, &convo(1, 30_000), 1, 7, 16, chain);
    save(&mut idx, &convo(1, 31_000), 2, 7, 16, chain);
    let (t1, t2) = (entry_of(&idx, 1).chain, entry_of(&idx, 2).chain);
    assert!(t1.superseded && !t1.branch);
    assert!(!t2.superseded);
    assert_eq!(t1.id, t2.id, "one conversation, one chain");
}

#[test]
fn diverging_continuations_make_a_branch_point() {
    let mut idx = SsmSnapshotIndex::new();
    let root = convo(1, 30_000);
    save(&mut idx, &root, 1, 7, 16, chain);
    let mut a = root.clone();
    a.extend(0..500u32);
    let mut b = root.clone();
    b.extend(9_000..9_700u32);
    save(&mut idx, &a, 2, 7, 16, chain);
    save(&mut idx, &b, 3, 7, 16, chain);
    let r = entry_of(&idx, 1).chain;
    assert!(r.branch && !r.superseded, "a fork point stays protected");
    assert_ne!(entry_of(&idx, 3).chain.id, entry_of(&idx, 2).chain.id);
}

#[test]
fn marked_branch_is_never_superseded() {
    let mut idx = SsmSnapshotIndex::new();
    let sys = convo(0, 2048);
    save(&mut idx, &sys, 1, 7, 16, chain);
    idx.mark_branch(hash_token_prefix(&sys, sys.len(), 0));
    save(&mut idx, &convo(4, 9_000), 2, 7, 16, chain);
    let s = entry_of(&idx, 1).chain;
    assert!(s.branch && !s.superseded);
}

#[test]
fn superseded_history_goes_before_any_frontier() {
    let mut idx = SsmSnapshotIndex::new();
    // B's only snapshot is the oldest entry; A then takes four turns.
    save(&mut idx, &convo(2, 40_000), 100, 7, 16, chain);
    for t in 0..4 {
        save(&mut idx, &convo(1, 30_000 + t * 1_000), t, 7, 16, chain);
    }
    let v = idx.chain_victim(true).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 0, "A's oldest dead tail");
    // The default policy (one session_hash for everyone) takes B's frontier.
    let v = base(&idx).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 100);
}

#[test]
fn frontiers_are_evicted_least_recently_used_first() {
    let mut idx = SsmSnapshotIndex::new();
    for s in 1..=3u32 {
        save(&mut idx, &convo(s, 30_000), s as usize, 7, 16, chain);
    }
    // Session 1 restores (lookup win bumps it); session 2 is now stalest.
    let t1 = convo(1, 30_016);
    assert!(idx.lookup_tiered(&t1, 30_016, 7, 0).is_some());
    let v = idx.chain_victim(true).unwrap();
    assert_eq!(idx.entries[v].snapshot_id, 2);
}

/// The serving scenario: 16 slots, one agent looping on fast tool calls and
/// three slower sessions. Chain retention keeps every session's frontier; the
/// default policy loses all three slow ones.
#[test]
fn a_fast_session_cannot_evict_slower_sessions() {
    for (policy, keeps_slow) in [(chain as fn(&_) -> _, true), (base, false)] {
        let mut idx = SsmSnapshotIndex::new();
        let mut id = 0usize;
        let mut frontier = [0usize; 4];
        for round in 0..3usize {
            for s in 1..4u32 {
                id += 1;
                save(&mut idx, &convo(s, 40_000 + round * 800), id, 7, 16, policy);
                frontier[s as usize] = id;
            }
            for t in 0..20usize {
                id += 1;
                let len = 30_000 + (round * 20 + t) * 600;
                save(&mut idx, &convo(9, len), id, 7, 16, policy);
                frontier[0] = id;
            }
            let slow_alive = (1..4).all(|s| resident(&idx, frontier[s]));
            assert_eq!(slow_alive, keeps_slow, "round {round}");
            assert!(resident(&idx, frontier[0]));
        }
    }
}

/// TP2: the worker's sequences carry `session_hash = 0`, the head's carry the
/// real hash. The same operation stream yields the same victims on both, so
/// `session_hash` is no longer a source of divergence.
#[test]
fn victims_do_not_depend_on_session_hash() {
    let run = |policy: fn(&SsmSnapshotIndex) -> Option<usize>, head: bool| {
        let mut idx = SsmSnapshotIndex::new();
        let mut evicted = Vec::new();
        let ops: [(u32, usize, u64); 6] = [
            (1, 30_000, 0xA),
            (2, 20_000, 0xB),
            (1, 31_000, 0xA),
            (3, 25_000, 0xC),
            (1, 32_000, 0xA),
            (4, 22_000, 0xD),
        ];
        for (id, (s, len, h)) in ops.into_iter().enumerate() {
            let sh = if head { h } else { 0 };
            evicted.extend(save(&mut idx, &convo(s, len), id, sh, 3, policy));
        }
        evicted
    };
    assert_eq!(run(chain, true), run(chain, false));
    assert_ne!(
        run(base, true),
        run(base, false),
        "the default grouping diverges across ranks on this stream"
    );
}

#[test]
fn resident_at_matches_exact_depth_only() {
    let mut idx = SsmSnapshotIndex::new();
    let t = convo(1, 30_000);
    save(&mut idx, &t[..20_000], 5, 7, 16, chain);
    let before = entry_of(&idx, 5).last_access;
    assert_eq!(idx.resident_at(&t, 20_000, 0), Some(5));
    assert!(entry_of(&idx, 5).last_access > before, "a hit is a use");
    assert_eq!(idx.resident_at(&t, 20_016, 0), None);
    assert_eq!(idx.resident_at(&t, 19_984, 0), None);
    assert_eq!(idx.resident_at(&convo(2, 30_000), 20_000, 0), None);
    assert_eq!(idx.resident_at(&t, 20_000, 3), None, "adapter-keyed");
    idx.entries[0].tiered = true;
    assert_eq!(
        idx.resident_at(&t, 20_000, 0),
        None,
        "spilled is not resident"
    );
    idx.entries[0].tiered = false;
    idx.entries[0].is_tail = true;
    assert_eq!(
        idx.resident_at(&t, 20_000, 0),
        None,
        "tails bleed past depth"
    );
}

/// Victim identity is NOT a cross-rank guarantee: a rank-local recency bump
/// (the F83 re-lookup on the rank whose match was capped, or `resident_at` on
/// the rank that restores shallower than it could) reorders that rank's LRU.
/// Safety comes from the restore-depth agreement (`pc_policy::agree_restore`),
/// which falls back to recompute when a rank lacks the agreed snapshot.
#[test]
fn a_rank_local_bump_can_change_the_victim() {
    let mut a = SsmSnapshotIndex::new();
    for s in 1..=3u32 {
        save(&mut a, &convo(s, 30_000), s as usize, 7, 16, chain);
    }
    let mut b = SsmSnapshotIndex::new();
    for s in 1..=3u32 {
        save(&mut b, &convo(s, 30_000), s as usize, 7, 16, chain);
    }
    assert_eq!(b.resident_at(&convo(1, 30_016), 30_000, 0), Some(1));
    let victim = |i: &SsmSnapshotIndex| i.entries[i.chain_victim(true).unwrap()].snapshot_id;
    assert_eq!((victim(&a), victim(&b)), (1, 2));
}

/// Branch placement reads the KV radix: a request forks from the cached path
/// at `depth` only when a different full block continues it there.
#[test]
fn forks_at_sees_only_a_diverging_full_block() {
    use crate::prefix_cache::PrefixCache;
    let tree = crate::radix_tree::RadixTree::new();
    let s0 = convo(1, 4_096 + 64);
    tree.insert(&s0, &(0..260u32).collect::<Vec<_>>(), &[], 16, 0, 0);
    let s1 = convo(2, 4_096);
    // Both share the first 2048 tokens (see `convo`); s0 continues differently.
    assert!(tree.forks_at(&s1, 2_048, 16, 0));
    // Along s0's own path nothing else branches off.
    assert!(!tree.forks_at(&s0, 2_048, 16, 0));
    assert!(!tree.forks_at(&s0, 4_096, 16, 0));
    // Past the cached path, unaligned, out of range, other adapter: no fork.
    assert!(!tree.forks_at(&s1, 2_064, 16, 0));
    assert!(!tree.forks_at(&s1, 2_050, 16, 0));
    assert!(!tree.forks_at(&s1, 4_096, 16, 0));
    assert!(!tree.forks_at(&s1, 2_048, 16, 5));
}
