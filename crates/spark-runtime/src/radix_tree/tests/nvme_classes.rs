// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill tier with record-slot classes (`enable_classes`, used by a
//! latent-sharded KV cache): a block spills into a slot of its id residue,
//! each class has its own budget, and a full class drops only its own
//! on-disk leaves. One class is the unclassed tier, slot for slot.

use super::nvme::{BS, cache, spill_and_restore, toks};
use crate::prefix_cache::{NvmePrefixTier, PrefixCache};
use crate::radix_tree::RadixTree;

fn classed_tree(per_class: u32) -> RadixTree {
    let tree = RadixTree::new();
    assert!(tree.enable_classes(per_class, 2));
    tree
}

#[test]
fn enable_classes_needs_a_budget_and_a_class() {
    let tree = RadixTree::new();
    assert!(!tree.enable_classes(0, 2));
    assert!(!tree.enable_classes(4, 0));
    assert!(tree.enable_classes(4, 2));
    assert!(!tree.enable(4), "once");
    assert_eq!(tree.nvme_stats().max_slots, 8, "every class's slots");
}

/// A chain drawn as a latent shard draws it (id residue = logical index
/// residue) spills each block into a slot of its residue, and plans the
/// same slots back in path order.
#[test]
fn every_block_takes_a_slot_of_its_residue() {
    let tree = classed_tree(4);
    let t = toks(0, 4);
    // Logical 0..4 on residues 0, 1, 0, 1 (rank-local ids, as a shard draws).
    let blocks = [6, 3, 0, 9];
    cache(&tree, &t, &blocks);
    let ev = tree.evict(4);
    assert_eq!(ev.spill.len(), 4);
    for o in &ev.spill {
        assert_eq!(o.slot % 2, o.block % 2, "{o:?}");
    }
    let plan = tree.plan_restore(&t, BS, 0);
    let classes: Vec<u32> = plan.disk.iter().map(|d| d.slot % 2).collect();
    assert_eq!(classes, vec![0, 1, 0, 1], "path order keeps the residues");
    tree.complete_restore(&t, BS, 0, &plan, &[], false);
    let s = tree.nvme_stats();
    assert_eq!((s.slots_used, s.spills), (4, 4));
}

/// Two ranks of a shard hold different ids for the same chain, with the same
/// residues: their trees hand out the SAME slots (the classes, and so the
/// budget drops, agree across the pair).
#[test]
fn ranks_with_different_ids_get_the_same_slots() {
    let t = toks(0, 5);
    let u = toks(7000, 3);
    let run = |a: [u32; 5], b: [u32; 3]| {
        let tree = classed_tree(3);
        cache(&tree, &t, &a);
        cache(&tree, &u, &b);
        let mut slots: Vec<(u32, u64)> = Vec::new();
        for _ in 0..2 {
            let ev = tree.evict(8);
            slots.extend(ev.spill.iter().map(|o| (o.slot, o.tag)));
        }
        let plan = |x: &[u32]| {
            let p = tree.plan_restore(x, BS, 0);
            tree.complete_restore(x, BS, 0, &p, &[], false);
            p.disk
        };
        (slots, plan(&t), plan(&u), tree.nvme_stats())
    };
    let rank0 = run([0, 1, 2, 3, 4], [10, 11, 12]);
    let rank1 = run([8, 5, 12, 1, 6], [2, 9, 4]);
    assert_eq!(rank0, rank1);
}

/// A full class drops its own coldest leaf; a block of the OTHER class keeps
/// its record, and a class with room never drops anything.
#[test]
fn a_full_class_drops_only_its_own_leaves() {
    let tree = classed_tree(1);
    let a = toks(0, 1);
    let b = toks(5000, 1);
    let c = toks(9000, 1);
    cache(&tree, &b, &[3]); // class 1, the coldest record
    cache(&tree, &a, &[2]); // class 0
    assert_eq!(tree.evict(2).spill.len(), 2);
    assert_eq!(tree.nvme_stats().slots_used, 2);
    // Class 0 is full: `c` (class 0) drops `a`, not the colder `b`.
    cache(&tree, &c, &[4]);
    let ev = tree.evict(1);
    assert_eq!((ev.spill.len(), ev.spill[0].slot % 2), (1, 0));
    assert!(tree.plan_restore(&a, BS, 0).disk.is_empty(), "a dropped");
    let plan = tree.plan_restore(&b, BS, 0);
    assert_eq!(plan.disk.len(), 1, "b kept its class-1 record");
    tree.complete_restore(&b, BS, 0, &plan, &[], false);
    let s = tree.nvme_stats();
    assert_eq!((s.disk_drops, s.cold_drops, s.slots_used), (1, 0, 2));
}

/// A full class whose only on-disk leaf is hotter than the victim deletes
/// the victim (a cold drop), although the other class has room.
#[test]
fn a_full_class_cold_drops_a_colder_victim_beside_a_free_class() {
    let tree = classed_tree(1);
    let held = toks(0, 1);
    let newer = toks(5000, 1);
    cache(&tree, &held, &[30]); // class 0
    // A live sequence holds `held`: not evictable while `newer` goes.
    assert_eq!(tree.lookup(&held, BS, 0, 0).matched_blocks, vec![30]);
    cache(&tree, &newer, &[20]); // class 0, hotter
    assert_eq!(tree.evict(1).spill[0].block, 20);
    tree.release(&held, BS, 0);
    let ev = tree.evict(1);
    assert_eq!(ev.physical, vec![30]);
    assert!(ev.spill.is_empty(), "class 1's free slot is not class 0's");
    let s = tree.nvme_stats();
    assert_eq!((s.cold_drops, s.disk_drops, s.slots_used), (1, 0, 1));
    // Class 1 still takes its own blocks.
    let odd = toks(9000, 1);
    cache(&tree, &odd, &[7]);
    assert_eq!(tree.evict(1).spill[0].slot % 2, 1);
}

/// A full class whose last record in a chain sits above a leaf of the
/// other class trims the chain's tail: that record and the leaf below it.
#[test]
fn a_full_class_trims_a_chain_tail_across_classes() {
    let tree = classed_tree(2);
    let chain = toks(0, 4);
    cache(&tree, &chain, &[0, 1, 2, 3]); // classes 0, 1, 0, 1
    assert_eq!(tree.evict(4).spill.len(), 4);
    assert_eq!(tree.nvme_stats().slots_used, 4);
    // Class 0 is full ({0, 2}); its only leaf-adjacent record is block 2's,
    // under which block 3 (class 1) is the chain's leaf.
    let newer = toks(9000, 1);
    cache(&tree, &newer, &[6]);
    let ev = tree.evict(1);
    assert_eq!((ev.spill.len(), ev.spill[0].slot % 2), (1, 0));
    let s = tree.nvme_stats();
    assert_eq!((s.disk_drops, s.cold_drops, s.slots_used), (2, 0, 3));
    let plan = tree.plan_restore(&chain, BS, 0);
    assert_eq!(plan.disk.len(), 2, "the chain keeps its first two blocks");
    tree.complete_restore(&chain, BS, 0, &plan, &[], false);
}

/// Kept records (`set_keep_restored`) count against their own class: a
/// class past half full gives a restored record back while the other class,
/// at half, keeps its own.
#[test]
fn records_are_kept_while_their_own_class_is_half_free() {
    let tree = classed_tree(2);
    tree.set_keep_restored(true);
    // Class 0 holds blocks 0 and 2 (full), class 1 holds block 1 (half).
    let t = toks(0, 3);
    spill_and_restore(&tree, &t, &[0, 1, 2], &[10, 11, 12]);
    // Block 0's record goes back (class 0 was full: 2 of 2), which leaves
    // class 0 at half: block 2 keeps its record, and so does block 1.
    assert_eq!(tree.nvme_stats().slots_used, 2);
    let again = tree.evict(3);
    let written: Vec<u32> = again.spill.iter().map(|o| o.block).collect();
    assert_eq!(written, vec![10], "only block 0 is written again");
    assert_eq!(tree.nvme_stats().clean_evictions, 2);
}

/// One class is the unclassed tier: the same slots in the same order.
#[test]
fn one_class_is_the_plain_tier() {
    let t = toks(0, 3);
    let run = |tree: RadixTree| {
        cache(&tree, &t, &[11, 12, 13]);
        let ev = tree.evict(3);
        ev.spill
    };
    let plain = RadixTree::new();
    assert!(plain.enable(8));
    let one = RadixTree::new();
    assert!(one.enable_classes(8, 1));
    assert_eq!(run(plain), run(one));
}
