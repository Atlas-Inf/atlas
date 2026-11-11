// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill-tier bookkeeping: spill-on-evict, restore planning/promotion,
//! disk budget LRU, pins, failure handling and `insert` re-homing.

use crate::prefix_cache::{NvmePrefixTier, PrefixCache, SpillOrder};
use crate::radix_tree::RadixTree;

pub(super) const BS: usize = 16;

pub(super) fn toks(base: u32, blocks: usize) -> Vec<u32> {
    (base..base + (blocks * BS) as u32).collect()
}

/// Cache a finished request: insert as a cache miss, then the sequence exits.
pub(super) fn cache(tree: &RadixTree, tokens: &[u32], blocks: &[u32]) {
    tree.insert(tokens, blocks, &[], BS, 0, 0);
    tree.release(tokens, BS, 0);
}

pub(super) fn spilled_tree(max_slots: u32) -> RadixTree {
    let tree = RadixTree::new();
    assert!(tree.enable(max_slots));
    tree
}

#[test]
fn disabled_tier_deletes_and_plans_nothing() {
    let tree = RadixTree::new();
    let t = toks(0, 2);
    cache(&tree, &t, &[10, 20]);
    let ev = tree.evict(2);
    assert_eq!(ev.physical, vec![20, 10]);
    assert!(ev.spill.is_empty());
    assert!(!tree.is_enabled());
    assert!(tree.plan_restore(&t, BS, 0).disk.is_empty());
}

#[test]
fn enable_is_once_and_needs_budget() {
    let tree = RadixTree::new();
    assert!(!tree.enable(0));
    assert!(tree.enable(4));
    assert!(!tree.enable(8));
}

#[test]
fn evict_spills_leaf_first_and_keeps_nodes() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);

    let ev = tree.evict(1);
    assert_eq!(ev.physical, vec![30]);
    assert_eq!(ev.spill.len(), 1);
    assert_eq!(ev.spill[0].block, 30);

    // Resident-only lookup now stops above the spilled block.
    let m = tree.lookup(&t, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![10, 20]);
    tree.release(&t, BS, 0);

    let ev = tree.evict(2);
    assert_eq!(ev.physical, vec![20, 10]);
    assert_eq!(ev.spill.len(), 2);
    assert!(tree.lookup(&t, BS, 0, 0).is_empty());
    assert_eq!(tree.stats().0, 0, "no resident entries left");
    assert_eq!(tree.nvme_stats().slots_used, 3);
    assert_eq!(tree.nvme_stats().spills, 3);
}

#[test]
fn restore_round_trip_promotes_in_path_order() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    let ev = tree.evict(3);
    let slot_of = |b: u32| ev.spill.iter().find(|o| o.block == b).unwrap().slot;

    let plan = tree.plan_restore(&t, BS, 0);
    assert_eq!(plan.resident_tokens, 0);
    assert_eq!(plan.pinned_tokens, 3 * BS);
    let slots: Vec<u32> = plan.disk.iter().map(|d| d.slot).collect();
    assert_eq!(slots, vec![slot_of(10), slot_of(20), slot_of(30)]);
    for (d, b) in plan.disk.iter().zip([10, 20, 30]) {
        let o = ev.spill.iter().find(|o| o.block == b).unwrap();
        assert_eq!(d.tag, o.tag, "plan carries the spill's record tag");
    }

    let give_back = tree.complete_restore(&t, BS, 0, &plan, &[100, 101, 102], false);
    assert!(give_back.is_empty());
    let m = tree.lookup(&t, BS, 0, 0);
    assert_eq!(m.matched_blocks, vec![100, 101, 102]);
    tree.release(&t, BS, 0);
    let s = tree.nvme_stats();
    assert_eq!((s.restores, s.slots_used), (3, 0));
    // Pin released: the restored chain is evictable (spillable) again.
    assert_eq!(tree.evict(3).spill.len(), 3);
}

#[test]
fn restore_continues_below_resident_prefix() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    tree.evict(1); // only the leaf goes to disk
    let plan = tree.plan_restore(&t, BS, 0);
    assert_eq!(plan.resident_tokens, 2 * BS);
    assert_eq!(plan.disk.len(), 1);
    // A longer prompt plans the same run (only full, cached blocks).
    let mut longer = t.clone();
    longer.extend(1000..1010);
    let plan2 = tree.plan_restore(&longer, BS, 0);
    assert_eq!(plan2.disk, plan.disk);
    tree.complete_restore(&longer, BS, 0, &plan2, &[], false);
    tree.complete_restore(&t, BS, 0, &plan, &[77], false);
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![10, 20, 77]);
}

#[test]
fn pinned_path_survives_eviction_during_restore() {
    let tree = spilled_tree(8);
    let a = toks(0, 2);
    cache(&tree, &a, &[10, 20]);
    tree.evict(1); // 20 → disk; 10 stays resident
    let plan = tree.plan_restore(&a, BS, 0);
    assert_eq!(plan.disk.len(), 1);
    // The restore's own allocation evicts: the pinned resident parent must not
    // be picked (it has no resident child, so it WOULD be without the pin).
    assert!(tree.evict(1).is_empty());
    tree.complete_restore(&a, BS, 0, &plan, &[21], false);
    assert_eq!(tree.lookup(&a, BS, 0, 0).matched_blocks, vec![10, 21]);
}

#[test]
fn failed_read_drops_node_and_disk_subtree() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    tree.evict(3);
    let plan = tree.plan_restore(&t, BS, 0);
    // First record restored, second read failed.
    let give_back = tree.complete_restore(&t, BS, 0, &plan, &[100], true);
    assert!(give_back.is_empty());
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![100]);
    tree.release(&t, BS, 0);
    assert!(
        tree.plan_restore(&t, BS, 0).disk.is_empty(),
        "never restore past a bad record"
    );
    let s = tree.nvme_stats();
    assert_eq!((s.restore_failures, s.slots_used), (1, 0));
}

#[test]
fn unadopted_blocks_are_handed_back() {
    let tree = spilled_tree(8);
    let t = toks(0, 1);
    cache(&tree, &t, &[10]);
    tree.evict(1);
    let plan = tree.plan_restore(&t, BS, 0);
    // More blocks than the plan has nodes: the extra one goes back.
    let give_back = tree.complete_restore(&t, BS, 0, &plan, &[5, 6], false);
    assert_eq!(give_back, vec![6]);
}

#[test]
fn spill_failure_degrades_to_plain_eviction() {
    let tree = spilled_tree(8);
    let t = toks(0, 2);
    cache(&tree, &t, &[10, 20]);
    let ev = tree.evict(2);
    // The parent's write failed: it and the (on-disk) child are dropped.
    let parent: Vec<SpillOrder> = ev.spill.iter().copied().filter(|o| o.block == 10).collect();
    assert!(tree.spill_failed(&parent).is_empty());
    assert!(tree.plan_restore(&t, BS, 0).disk.is_empty());
    let s = tree.nvme_stats();
    assert_eq!((s.spill_failures, s.slots_used), (1, 0));
}

#[test]
fn stale_spill_failure_never_drops_the_slots_new_owner() {
    let tree = spilled_tree(1);
    let a = toks(0, 1);
    let b = toks(5000, 1);
    cache(&tree, &a, &[10]);
    cache(&tree, &b, &[20]);
    let first = tree.evict(1).spill[0]; // a -> slot 0
    let second = tree.evict(1).spill[0]; // b displaces a in slot 0
    assert_eq!((first.slot, second.slot), (0, 0));
    assert_ne!(first.tag, second.tag);
    // a's (late) failure report names slot 0 but a's tag: b must survive.
    assert!(tree.spill_failed(&[first]).is_empty());
    let plan = tree.plan_restore(&b, BS, 0);
    assert_eq!(plan.disk.len(), 1, "b's record still owns the slot");
    tree.complete_restore(&b, BS, 0, &plan, &[], false);
}

#[test]
fn over_released_node_is_deleted_not_spilled() {
    let tree = spilled_tree(8);
    let t = toks(0, 1);
    cache(&tree, &t, &[10]);
    tree.release(&t, BS, 0); // over-release: the cache's own ref is gone
    let ev = tree.evict(1);
    assert_eq!(ev.physical, vec![10]);
    assert!(ev.spill.is_empty(), "an unreachable node is never spilled");
    let s = tree.nvme_stats();
    assert_eq!((s.spills, s.slots_used, s.cold_drops), (0, 0, 0));
}

#[test]
fn budget_drops_coldest_disk_leaves_first() {
    let tree = spilled_tree(2);
    let a = toks(0, 2); // older conversation
    let b = toks(5000, 2); // newer conversation
    cache(&tree, &a, &[10, 20]);
    cache(&tree, &b, &[30, 40]);
    assert_eq!(tree.evict(2).spill.len(), 2); // a fills the budget
    // b spills by dropping a's coldest leaf (20), then a's now-leaf root (10).
    assert_eq!(tree.evict(2).spill.len(), 2);
    let s = tree.nvme_stats();
    assert_eq!((s.disk_drops, s.slots_used), (2, 2));
    assert!(tree.plan_restore(&a, BS, 0).disk.is_empty());
    let plan = tree.plan_restore(&b, BS, 0);
    assert_eq!(plan.disk.len(), 2);
    tree.complete_restore(&b, BS, 0, &plan, &[], false);

    // A newer block displaces b's deepest record; b stays restorable to depth 1.
    let c = toks(9000, 1);
    cache(&tree, &c, &[50]);
    let ev = tree.evict(1);
    assert_eq!(ev.spill.len(), 1);
    assert_eq!(ev.spill[0].block, 50);
    let plan = tree.plan_restore(&b, BS, 0);
    assert_eq!(plan.disk.len(), 1);
    tree.complete_restore(&b, BS, 0, &plan, &[], false);
}

#[test]
fn colder_candidate_is_deleted_not_spilled() {
    let tree = spilled_tree(1);
    let old = toks(0, 1);
    let held = toks(8000, 1);
    let new = toks(7000, 1);
    cache(&tree, &old, &[10]);
    cache(&tree, &held, &[30]);
    // A live sequence holds `held`: not evictable while the others go.
    assert_eq!(tree.lookup(&held, BS, 0, 0).matched_blocks, vec![30]);
    cache(&tree, &new, &[20]);
    // LRU picks `old` first: spilled into the only slot.
    assert_eq!(tree.evict(1).spill[0].block, 10);
    // `new` is hotter than `old`'s record: displaces it.
    assert_eq!(tree.evict(1).spill[0].block, 20);
    let s = tree.nvme_stats();
    assert_eq!((s.disk_drops, s.cold_drops), (1, 0));
    // Released, `held` is colder than the on-disk record: deleted outright.
    tree.release(&held, BS, 0);
    let ev = tree.evict(1);
    assert_eq!(ev.physical, vec![30]);
    assert!(ev.spill.is_empty());
    assert_eq!(tree.nvme_stats().cold_drops, 1);
}

#[test]
fn insert_rehomes_disk_nodes_onto_recomputed_blocks() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    cache(&tree, &t, &[10, 20, 30]);
    tree.evict(2); // 30, 20 → disk
    // A sequence that did not restore recomputes all 3 blocks and inserts.
    let acq = tree.insert(&t, &[11, 21, 31], &[], BS, 0, 0);
    assert_eq!(acq.blocks, vec![21, 31], "re-homed blocks take a cache ref");
    tree.release(&t, BS, 0);
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![10, 21, 31]);
    assert_eq!(tree.nvme_stats().slots_used, 0);
}

#[test]
fn snapshot_anchor_depth_is_read_only() {
    let tree = spilled_tree(8);
    let t = toks(0, 3);
    tree.insert_with_snapshot(&t[..2 * BS], &[10, 20], &[], BS, 7, 0, 0, 0);
    tree.release(&t[..2 * BS], BS, 0);
    assert_eq!(tree.snapshot_anchor_depth(&t, 3 * BS, 0, 0), 2 * BS);
    assert_eq!(tree.snapshot_anchor_depth(&t, BS, 0, 0), 0);
    assert_eq!(tree.snapshot_anchor_depth(&toks(900, 3), 3 * BS, 0, 0), 0);
    assert_eq!(tree.snapshot_count(), 1);
}

/// The sizing constant must cover what one on-disk block really costs.
#[test]
fn host_bytes_per_disk_block_covers_a_node() {
    use crate::prefix_cache::NVME_HOST_BYTES_PER_BLOCK;
    let node = std::mem::size_of::<crate::radix_tree::inner::RadixNode>();
    // The node at the arena Vec's average 1.5× slack; `parent_key` (16 × 4 B
    // + allocator header); the parent's one-entry `children` table (4 buckets
    // of 32 B + control bytes) and its key; owner + tag + disk-LRU entry.
    let estimate = node * 3 / 2 + 80 + 160 + 80 + 60;
    assert!(
        estimate <= NVME_HOST_BYTES_PER_BLOCK,
        "{estimate} B per on-disk block (node = {node} B): raise NVME_HOST_BYTES_PER_BLOCK"
    );
    assert!(
        estimate * 4 >= NVME_HOST_BYTES_PER_BLOCK * 3,
        "{estimate} B per on-disk block: the reserve is over 33% too generous"
    );
}

pub(super) fn keeping_tree(max_slots: u32) -> RadixTree {
    let tree = spilled_tree(max_slots);
    tree.set_keep_restored(true);
    tree
}

/// Spill `blocks` of `t`, then restore the whole chain into `into`.
pub(super) fn spill_and_restore(
    tree: &RadixTree,
    t: &[u32],
    blocks: &[u32],
    into: &[u32],
) -> Vec<SpillOrder> {
    cache(tree, t, blocks);
    let ev = tree.evict(blocks.len());
    assert_eq!(ev.spill.len(), blocks.len());
    let plan = tree.plan_restore(t, BS, 0);
    assert!(
        tree.complete_restore(t, BS, 0, &plan, into, false)
            .is_empty()
    );
    ev.spill
}

#[test]
fn a_kept_record_makes_the_next_eviction_free() {
    let tree = keeping_tree(16);
    let t = toks(0, 3);
    let spill = spill_and_restore(&tree, &t, &[10, 20, 30], &[100, 101, 102]);
    // Resident again, and the records stay.
    assert_eq!(
        tree.lookup(&t, BS, 0, 0).matched_blocks,
        vec![100, 101, 102]
    );
    tree.release(&t, BS, 0);
    assert_eq!(tree.nvme_stats().slots_used, 3);
    // Evicting the chain again writes nothing and frees the blocks at once.
    let again = tree.evict(3);
    assert_eq!(again.physical, vec![102, 101, 100]);
    assert!(again.spill.is_empty());
    let s = tree.nvme_stats();
    assert_eq!((s.spills, s.clean_evictions, s.slots_used), (3, 3, 3));
    assert!(tree.lookup(&t, BS, 0, 0).is_empty());
    // … and the SAME records (slot and tag) restore it.
    let plan = tree.plan_restore(&t, BS, 0);
    for (d, b) in plan.disk.iter().zip([10, 20, 30]) {
        let o = spill.iter().find(|o| o.block == b).unwrap();
        assert_eq!((d.slot, d.tag), (o.slot, o.tag));
    }
    assert!(
        tree.complete_restore(&t, BS, 0, &plan, &[7, 8, 9], false)
            .is_empty()
    );
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![7, 8, 9]);
}

#[test]
fn records_are_kept_only_while_half_the_budget_is_free() {
    // 3 of 4 slots in use: the first restored node gives its slot back, which
    // brings the tier to half full — the other two keep theirs.
    let tree = keeping_tree(4);
    let t = toks(0, 3);
    spill_and_restore(&tree, &t, &[10, 20, 30], &[100, 101, 102]);
    assert_eq!(tree.nvme_stats().slots_used, 2);
    let again = tree.evict(3);
    assert_eq!(again.physical, vec![102, 101, 100]);
    assert_eq!(
        again.spill.len(),
        1,
        "only the node without a record is written"
    );
    assert_eq!(again.spill[0].block, 100);
    assert_eq!(tree.nvme_stats().clean_evictions, 2);
}

#[test]
fn a_kept_record_survives_a_re_insert_and_goes_with_its_node() {
    let tree = keeping_tree(16);
    let t = toks(0, 2);
    spill_and_restore(&tree, &t, &[10, 20], &[100, 101]);
    // Another request over the same prefix: the nodes keep block and record.
    let acq = tree.insert(&t, &[55, 56], &[], BS, 0, 0);
    assert!(acq.blocks.is_empty());
    tree.release(&t, BS, 0);
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![100, 101]);
    tree.release(&t, BS, 0);
    assert_eq!(tree.nvme_stats().slots_used, 2);
    // An over-released (unreachable) node is deleted: its record goes too.
    tree.release(&t, BS, 0);
    let ev = tree.evict(2);
    assert_eq!(ev.physical, vec![101, 100]);
    assert!(ev.spill.is_empty());
    let s = tree.nvme_stats();
    assert_eq!((s.slots_used, s.clean_evictions), (0, 0));
    assert!(tree.plan_restore(&t, BS, 0).disk.is_empty());
}

#[test]
fn a_kept_record_below_the_resident_prefix_is_not_planned() {
    // Only the leaf is evicted (clean); the plan is the leaf alone, after two
    // resident — and still record-carrying — ancestors.
    let tree = keeping_tree(16);
    let t = toks(0, 3);
    spill_and_restore(&tree, &t, &[10, 20, 30], &[100, 101, 102]);
    assert!(tree.evict(1).spill.is_empty());
    let plan = tree.plan_restore(&t, BS, 0);
    assert_eq!((plan.resident_tokens, plan.disk.len()), (2 * BS, 1));
    tree.complete_restore(&t, BS, 0, &plan, &[77], false);
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![100, 101, 77]);
}

#[test]
fn a_late_failure_report_never_drops_a_node_restored_from_that_record() {
    // A multi-record run is reported failed as a whole although its leading
    // records may be intact. If one of those was read back (it verified) and
    // kept before the report arrives, the node is resident with the record.
    let tree = keeping_tree(16);
    let t = toks(0, 3);
    let spill = spill_and_restore(&tree, &t, &[10, 20, 30], &[100, 101, 102]);
    assert!(tree.spill_failed(&spill).is_empty(), "nothing handed back");
    assert_eq!(
        tree.lookup(&t, BS, 0, 0).matched_blocks,
        vec![100, 101, 102],
        "the restored chain is still served"
    );
    tree.release(&t, BS, 0);
    let s = tree.nvme_stats();
    assert_eq!((s.spill_failures, s.slots_used), (3, 3));
    // Once evicted again (no write: the record is kept), the node is on disk
    // ONLY — the same report then does drop it.
    assert!(tree.evict(1).spill.is_empty());
    let leaf = spill.iter().find(|o| o.block == 30).unwrap();
    assert!(tree.spill_failed(&[*leaf]).is_empty());
    assert_eq!(tree.nvme_stats().slots_used, 2);
    assert_eq!(tree.plan_restore(&t, BS, 0).disk.len(), 0);
    assert_eq!(tree.lookup(&t, BS, 0, 0).matched_blocks, vec![100, 101]);
}

/// Branch-point placement (`forks_at`, `ATLAS_GLM_PC_BRANCH`) reads the cached
/// path whether or not it is resident: a conversation whose continuation was
/// spilled still diverged there.
#[test]
fn a_spilled_continuation_still_counts_as_a_fork() {
    let tree = spilled_tree(8);
    let a = [toks(0, 2), toks(1000, 1)].concat();
    let b = [toks(0, 2), toks(2000, 1)].concat();
    cache(&tree, &a, &[10, 20, 30]);
    assert!(tree.forks_at(&b, 2 * BS, BS, 0));
    assert_eq!(tree.evict(1).spill[0].block, 30);
    assert!(tree.forks_at(&b, 2 * BS, BS, 0));
    assert!(!tree.forks_at(&a, 2 * BS, BS, 0));
}
