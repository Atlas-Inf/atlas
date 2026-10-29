// SPDX-License-Identifier: AGPL-3.0-only

//! The NVMe tier's record-slot index (split out of `nvme.rs`, 500-LoC cap):
//! slot allocation in classes under a fixed budget, slot owners and tags, the
//! on-disk LRU and the spill-candidate queue.

use std::cmp::Reverse;
use std::collections::{BTreeSet, VecDeque};

use super::NodeId;
use crate::prefix_cache::nvme::NvmeStats;

pub(in crate::radix_tree) const NO_OWNER: NodeId = usize::MAX;

pub(in crate::radix_tree) struct NvmeIndex {
    /// Slot classes (see [`slot_class`]): 1, or one per latent-shard rank.
    pub(in crate::radix_tree) classes: u32,
    /// Slots each class may hand out.
    per_class: u32,
    /// Per class: how many of its slots were ever handed out.
    next: Vec<u32>,
    /// Per class: released slots, reused last-in first-out.
    free: Vec<Vec<u32>>,
    /// slot → owning node (`NO_OWNER` when free).
    pub(in crate::radix_tree) owner: Vec<NodeId>,
    /// slot → tag stamped into the record at spill.
    pub(in crate::radix_tree) tags: Vec<u64>,
    /// Every on-disk node, keyed by its `last_access` (kept exact). Ties
    /// (a whole chain shares one access stamp) order the higher node id
    /// first — children are allocated after their parents, so the leaf a
    /// budget drop needs is found without walking the chain's interior.
    pub(in crate::radix_tree) lru: BTreeSet<(u64, Reverse<NodeId>)>,
    pub(in crate::radix_tree) epoch: u64,
    /// Cached spill candidates `(node, last_access)`, oldest first; each is
    /// re-validated when popped (the access stamp doubles as a generation).
    pub(in crate::radix_tree) victims: VecDeque<(NodeId, u64)>,
    /// A restored node keeps its record (it is then RESIDENT with a slot, and
    /// evicting it again needs no write). See `set_keep_restored`.
    pub(in crate::radix_tree) keep_restored: bool,
    pub(in crate::radix_tree) stats: NvmeStats,
}

/// The record class of a node spilled from physical block `block`, and of
/// slot `slot` (classes partition the slots: slot `s` is in class
/// `s % classes`). One class unless the cache is latent-sharded
/// (`ATLAS_GLM_KV_SHARD=1`), whose ranks store block `b`'s latents on rank
/// `b % world` and draw ids whose residue is the logical index's: a node's
/// class is then its depth's residue, the same on every rank, and each rank's
/// KV cache keeps one record file per class (full records for its own
/// residue, index-only records for the peer's).
pub(in crate::radix_tree) fn slot_class(id: u32, classes: u32) -> u32 {
    id % classes
}

impl NvmeIndex {
    /// `per_class` slots in each of `classes` (≥ 1) classes.
    pub(in crate::radix_tree) fn new(per_class: u32, classes: u32) -> Self {
        let classes = classes.max(1);
        let max_slots = per_class.saturating_mul(classes);
        Self {
            classes,
            per_class,
            next: vec![0; classes as usize],
            free: vec![Vec::new(); classes as usize],
            owner: Vec::new(),
            tags: Vec::new(),
            lru: BTreeSet::new(),
            epoch: 0,
            victims: VecDeque::new(),
            keep_restored: false,
            stats: NvmeStats {
                max_slots,
                ..NvmeStats::default()
            },
        }
    }

    pub(in crate::radix_tree) fn take_free_slot(&mut self, class: u32) -> Option<u32> {
        let c = class as usize;
        if let Some(s) = self.free[c].pop() {
            return Some(s);
        }
        if self.next[c] >= self.per_class {
            return None;
        }
        let s = self.next[c] * self.classes + class;
        self.next[c] += 1;
        if self.owner.len() <= s as usize {
            self.owner.resize(s as usize + 1, NO_OWNER);
            self.tags.resize(s as usize + 1, 0);
        }
        Some(s)
    }

    /// At most half of `class`'s slots hold a record (one class: half of
    /// the whole budget). Kept records are only kept while this holds.
    pub(in crate::radix_tree) fn half_free(&self, class: u32) -> bool {
        let c = class as usize;
        let used = u64::from(self.next[c]) - self.free[c].len() as u64;
        used * 2 <= u64::from(self.per_class)
    }

    pub(in crate::radix_tree) fn release_slot(&mut self, slot: u32) {
        self.owner[slot as usize] = NO_OWNER;
        self.free[slot_class(slot, self.classes) as usize].push(slot);
        self.stats.slots_used -= 1;
    }
}
