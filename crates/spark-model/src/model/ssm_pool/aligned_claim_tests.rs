// SPDX-License-Identifier: AGPL-3.0-only
//! Actual CPU free-list/guard transitions; no GPU pointers are dereferenced.
use super::super::slot_guard_tests::bare_pool;

#[test]
fn specific_guard_selects_exact_index_for_either_free_list_order() {
    for order in [[0, 1], [1, 0]] {
        for requested in 0..2 {
            let pool = bare_pool(2);
            let mut guards = [
                Some(pool.claim_guarded().unwrap()),
                Some(pool.claim_guarded().unwrap()),
            ];
            for who in order {
                drop(guards[who].take());
            }
            assert_eq!(*pool.free_slots.lock(), order);
            let guard = pool.claim_specific_guarded(requested).unwrap();
            assert_eq!(guard.idx(), Some(requested));
            assert!(guard.belongs_to(&pool));
            assert_eq!(*pool.free_slots.lock(), [1 - requested]);
            assert!(pool.claim_specific_guarded(requested).is_err());
            assert_eq!(*pool.free_slots.lock(), [1 - requested]);
            let peer = pool.claim_guarded().unwrap();
            assert_eq!(peer.idx(), Some(1 - requested));
            assert!(pool.claim_guarded().is_err());
            drop(guard);
            assert_eq!(*pool.free_slots.lock(), [requested]);
            let replacement = pool.claim_specific_guarded(requested).unwrap();
            assert!(pool.claim_guarded().is_err());
            drop(replacement);
            drop(peer);
            let mut free = pool.free_slots.lock().clone();
            free.sort_unstable();
            assert_eq!(free, [0, 1], "each actual guard returns its index once");
        }
    }
}

#[test]
fn occupied_and_out_of_range_specific_claims_leave_all_owners_unchanged() {
    let pool = bare_pool(2);
    let held = pool.claim_guarded().unwrap();
    assert_eq!(held.idx(), Some(0));
    for requested in [0, 2, usize::MAX] {
        let before = pool.free_slots.lock().clone();
        assert!(pool.claim_specific_guarded(requested).is_err());
        assert_eq!(*pool.free_slots.lock(), before);
        assert_eq!(held.idx(), Some(0));
        assert!(held.belongs_to(&pool));
    }
    let peer = pool.claim_guarded().unwrap();
    assert_eq!(peer.idx(), Some(1), "no refused claim consumed the peer");
    assert!(pool.claim_guarded().is_err());
    drop(peer);
    drop(held);
    assert_eq!(*pool.free_slots.lock(), [1, 0]);
}

#[test]
fn taking_specific_guard_neutralizes_drop_without_releasing_slot() {
    let pool = bare_pool(2);
    let mut guard = pool.claim_specific_guarded(0).unwrap();
    assert_eq!(guard.take(), Some(0));
    assert_eq!(guard.take(), None);
    drop(guard);
    assert_eq!(*pool.free_slots.lock(), [1]);
    assert!(pool.claim_specific_guarded(0).is_err());
    let peer = pool.claim_guarded().unwrap();
    assert_eq!(peer.idx(), Some(1));
    assert!(pool.claim_guarded().is_err());
    // Only the caller that took the real index can now return it explicitly.
    pool.release_slot(0);
    assert_eq!(*pool.free_slots.lock(), [0]);
    drop(peer);
    assert_eq!(*pool.free_slots.lock(), [0, 1]);
}

#[test]
fn generic_guarded_claim_retains_legacy_lifo_for_either_release_order() {
    for order in [[0, 1], [1, 0]] {
        let pool = bare_pool(2);
        let mut guards = [
            Some(pool.claim_guarded().unwrap()),
            Some(pool.claim_guarded().unwrap()),
        ];
        for who in order {
            drop(guards[who].take());
        }
        let first = pool.claim_guarded().unwrap();
        let second = pool.claim_guarded().unwrap();
        assert_eq!(first.idx(), Some(order[1]));
        assert_eq!(second.idx(), Some(order[0]));
        assert!(pool.claim_guarded().is_err());
    }
}
