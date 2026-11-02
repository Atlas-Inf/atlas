// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill tier: what a restore plan does to the disk LRU, and when a kept
//! record (`set_keep_restored`) must be given up.

use super::nvme::{BS, cache, keeping_tree, spill_and_restore, spilled_tree, toks};
use crate::prefix_cache::{NvmePrefixTier, PrefixCache};

/// A plan the caller declines (no usable snapshot anchor) must leave the run
/// where it was in the disk LRU: refreshed on every lookup, a run nobody can
/// use would never be the budget's victim.
#[test]
fn a_declined_plan_does_not_refresh_its_records() {
    let tree = spilled_tree(2);
    let useless = toks(0, 1);
    let live = toks(5000, 1);
    cache(&tree, &useless, &[10]);
    cache(&tree, &live, &[20]);
    assert_eq!(tree.evict(2).spill.len(), 2);
    // Every arrival plans over `useless` and declines it.
    for _ in 0..3 {
        let plan = tree.plan_restore(&useless, BS, 0);
        assert_eq!(plan.disk.len(), 1);
        assert!(
            tree.complete_restore(&useless, BS, 0, &plan, &[], false)
                .is_empty()
        );
    }
    // The budget is full: the next spill drops the OLDER record.
    let next = toks(9000, 1);
    cache(&tree, &next, &[30]);
    assert_eq!(tree.evict(1).spill[0].block, 30);
    assert!(
        tree.plan_restore(&useless, BS, 0).disk.is_empty(),
        "dropped"
    );
    let plan = tree.plan_restore(&live, BS, 0);
    assert_eq!(plan.disk.len(), 1, "the newer record is still there");
    tree.complete_restore(&live, BS, 0, &plan, &[], false);
}

/// A restored block is as recently used as the restore, not as its record.
#[test]
fn a_restored_run_is_the_most_recently_used() {
    let tree = spilled_tree(8);
    let a = toks(0, 1);
    let b = toks(5000, 1);
    cache(&tree, &a, &[10]);
    cache(&tree, &b, &[20]);
    assert_eq!(tree.evict(1).spill[0].block, 10);
    let plan = tree.plan_restore(&a, BS, 0);
    assert!(
        tree.complete_restore(&a, BS, 0, &plan, &[11], false)
            .is_empty()
    );
    // `b` was cached after `a` but `a` was restored since: `b` goes first.
    assert_eq!(tree.evict(1).spill[0].block, 20);
}

/// `forget_kept`: a block about to be rewritten in place gives up the record
/// it was restored from, so its next eviction writes the block again; the
/// blocks outside the range keep theirs.
#[test]
fn a_rewritten_block_gives_up_its_kept_record() {
    let tree = keeping_tree(16);
    let t = toks(0, 3);
    spill_and_restore(&tree, &t, &[10, 20, 30], &[100, 101, 102]);
    assert_eq!(tree.nvme_stats().slots_used, 3);
    // A prefill resumes at block 1 under a 3-block match: blocks 1 and 2 are
    // recomputed in place.
    tree.forget_kept(&t, BS, 0, 1..3);
    assert_eq!(tree.nvme_stats().slots_used, 1);
    let ev = tree.evict(3);
    assert_eq!(ev.physical, vec![102, 101, 100]);
    let written: Vec<u32> = ev.spill.iter().map(|o| o.block).collect();
    assert_eq!(written, vec![102, 101], "block 0 still has its record");
    let s = tree.nvme_stats();
    assert_eq!((s.clean_evictions, s.slots_used), (1, 3));
    // An empty or out-of-path range, and a tree that keeps nothing: no-ops.
    tree.forget_kept(&t, BS, 0, 3..3);
    tree.forget_kept(&toks(7000, 2), BS, 0, 0..2);
    assert_eq!(tree.nvme_stats().slots_used, 3);
    let plain = spilled_tree(4);
    cache(&plain, &t, &[1, 2, 3]);
    plain.forget_kept(&t, BS, 0, 0..3);
    assert_eq!(plain.evict(3).spill.len(), 3);
}
