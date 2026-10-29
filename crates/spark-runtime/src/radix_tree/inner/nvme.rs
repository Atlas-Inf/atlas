// SPDX-License-Identifier: AGPL-3.0-only

//! NVMe spill-tier bookkeeping for [`RadixTreeInner`] (see
//! `crate::prefix_cache::nvme` for the contract). Pure CPU state: record-slot
//! allocation under a fixed budget, the on-disk LRU, spill-victim selection,
//! restore planning / promotion and subtree drops. No I/O happens here.

use std::cmp::Reverse;

use super::nvme_index::{NO_OWNER, slot_class};
use super::{NodeId, RadixTreeInner};
use crate::prefix_cache::nvme::{DiskRef, NvmeStats, RestorePlan, SpillOrder};

/// Spill candidates gathered per full scan of the node arena. The scan is
/// O(nodes) and nodes include every on-disk block, so it is amortised over
/// this many evictions instead of paid per block (the legacy delete path
/// rescans per block; with 10^5-10^6 on-disk nodes that would dominate).
const VICTIM_BATCH: usize = 256;

/// Record tag: binds a record to the node's causal-prefix hash and to the
/// spill that wrote it, so a stale or misdirected record never restores.
fn record_tag(context_hash: u64, epoch: u64) -> u64 {
    let mut h = context_hash ^ epoch.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^ (h >> 29)
}

impl RadixTreeInner {
    pub(in crate::radix_tree) fn nvme_on(&self) -> bool {
        self.nvme.is_some()
    }

    /// On disk ONLY. A resident node may also hold a slot (a kept record).
    fn is_on_disk(&self, id: NodeId) -> bool {
        self.nodes[id].nvme_slot != u32::MAX && self.nodes[id].block_idx == u32::MAX
    }

    fn has_resident_child(&self, id: NodeId) -> bool {
        self.nodes[id]
            .children
            .values()
            .any(|&c| self.nodes[c].block_idx != u32::MAX)
    }

    fn is_spill_candidate(&self, id: NodeId) -> bool {
        let n = &self.nodes[id];
        n.parent.is_some()
            && n.block_idx != u32::MAX
            && n.ref_count <= 1
            && !self.has_resident_child(id)
    }

    fn next_spill_victim(&mut self) -> Option<NodeId> {
        for _ in 0..2 {
            while let Some((id, access)) = self.nvme.as_mut()?.victims.pop_front() {
                if self.nodes[id].last_access == access && self.is_spill_candidate(id) {
                    return Some(id);
                }
            }
            let mut found: Vec<(u64, NodeId)> = (0..self.nodes.len())
                .filter(|&id| self.is_spill_candidate(id))
                .map(|id| (self.nodes[id].last_access, id))
                .collect();
            if found.is_empty() {
                return None;
            }
            if found.len() > VICTIM_BATCH {
                found.select_nth_unstable(VICTIM_BATCH);
                found.truncate(VICTIM_BATCH);
            }
            found.sort_unstable();
            let idx = self.nvme.as_mut()?;
            idx.victims.extend(found.into_iter().map(|(a, id)| (id, a)));
        }
        None
    }

    /// After `child` left the resident set, its parent may have become a spill
    /// candidate: queue it in LRU position so a chain drains oldest-first
    /// instead of waiting for the next full scan (which would spread eviction
    /// one leaf per conversation across the whole cached queue).
    fn requeue_parent(&mut self, parent: Option<NodeId>) {
        let Some(p) = parent.filter(|&p| self.is_spill_candidate(p)) else {
            return;
        };
        let access = self.nodes[p].last_access;
        let Some(idx) = self.nvme.as_mut() else {
            return;
        };
        let pos = idx.victims.partition_point(|&(_, a)| a <= access);
        if pos < idx.victims.len() || idx.victims.is_empty() {
            idx.victims.insert(pos, (p, access));
        }
    }

    /// A record slot of `class` for a node last accessed at `cold_access`,
    /// dropping the LRU on-disk leaf of that class when the class is full.
    /// `None` when every droppable on-disk block of the class is hotter than
    /// the candidate (or none is droppable).
    ///
    /// With several classes a chain's last block of `class` may sit just
    /// above a leaf of another class (a latent shard's chains alternate): it
    /// is droppable together with those leaves, so a full class can always
    /// trim a chain's tail.
    fn alloc_slot(&mut self, cold_access: u64, class: u32) -> Option<u32> {
        if let Some(s) = self.nvme.as_mut()?.take_free_slot(class) {
            return Some(s);
        }
        let idx = self.nvme.as_ref()?;
        let leaf = |id: NodeId| self.nodes[id].children.is_empty();
        let victim = idx.lru.iter().copied().find(|&(_, Reverse(id))| {
            let n = &self.nodes[id];
            let tail = leaf(id) || (idx.classes > 1 && n.children.values().all(|&c| leaf(c)));
            tail && n.ref_count <= 1 && slot_class(n.nvme_slot, idx.classes) == class
        });
        match victim {
            Some((access, Reverse(id))) if access <= cold_access => {
                let nodes = 1 + self.nodes[id].children.len() as u64;
                let freed = self.drop_subtree(id);
                debug_assert!(freed.is_empty(), "on-disk tail held resident blocks");
                let idx = self.nvme.as_mut()?;
                idx.stats.disk_drops += nodes;
                idx.take_free_slot(class)
            }
            _ => None,
        }
    }

    /// Spill-mode eviction: like [`Self::evict`] but the victim's block is
    /// written to disk (a returned [`SpillOrder`]) and its node kept.
    pub(super) fn evict_spill(&mut self, num_blocks: usize) -> (Vec<u32>, Vec<SpillOrder>) {
        let mut phys = Vec::new();
        let mut orders = Vec::new();
        while phys.len() < num_blocks {
            let Some(id) = self.next_spill_victim() else {
                break;
            };
            let access = self.nodes[id].last_access;
            let parent = self.nodes[id].parent;
            // A ref-0 node has already lost the cache's baseline ref (an
            // over-release): no lookup can reach it (`walk`/`nvme_path` need
            // ref > 0) and its block's KV ref may be gone, so spilling it
            // would only persist bytes nobody owns. Delete it as before.
            let orphan = self.nodes[id].ref_count == 0;
            if !orphan && self.nodes[id].nvme_slot != u32::MAX {
                // The record it was restored from is still there: no write.
                phys.push(std::mem::replace(&mut self.nodes[id].block_idx, u32::MAX));
                if let Some(idx) = self.nvme.as_mut() {
                    idx.lru.insert((access, Reverse(id)));
                    idx.stats.clean_evictions += 1;
                }
                self.requeue_parent(parent);
                continue;
            }
            let slot = if orphan {
                None
            } else {
                let classes = self.nvme.as_ref().map_or(1, |idx| idx.classes);
                self.alloc_slot(access, slot_class(self.nodes[id].block_idx, classes))
            };
            let Some(slot) = slot else {
                if !orphan && let Some(idx) = self.nvme.as_mut() {
                    idx.stats.cold_drops += 1;
                }
                phys.extend(self.drop_subtree(id));
                self.requeue_parent(parent);
                continue;
            };
            let node = &mut self.nodes[id];
            let block = std::mem::replace(&mut node.block_idx, u32::MAX);
            node.nvme_slot = slot;
            let ctx = node.context_hash;
            phys.push(block);
            let Some(idx) = self.nvme.as_mut() else {
                break;
            };
            idx.epoch += 1;
            let tag = record_tag(ctx, idx.epoch);
            idx.owner[slot as usize] = id;
            idx.tags[slot as usize] = tag;
            idx.lru.insert((access, Reverse(id)));
            idx.stats.slots_used += 1;
            idx.stats.spills += 1;
            orders.push(SpillOrder { block, slot, tag });
            self.requeue_parent(parent);
        }
        (phys, orders)
    }

    /// Remove `root` and everything below it; returns the resident blocks
    /// the cache held refs on (each to be returned once).
    fn drop_subtree(&mut self, root: NodeId) -> Vec<u32> {
        let mut blocks = Vec::new();
        if let Some(parent) = self.nodes[root].parent
            && let Some(key) = self.nodes[root].parent_key.clone()
        {
            self.nodes[parent].children.remove(&key);
        }
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            stack.extend(self.nodes[id].children.values().copied());
            let node = &mut self.nodes[id];
            if node.block_idx != u32::MAX {
                blocks.push(node.block_idx);
            }
            let slot = std::mem::replace(&mut node.nvme_slot, u32::MAX);
            let access = node.last_access;
            node.block_idx = u32::MAX;
            node.disk_block_id = u32::MAX;
            node.children.clear();
            node.parent = None;
            node.parent_key = None;
            if slot != u32::MAX
                && let Some(idx) = self.nvme.as_mut()
            {
                idx.lru.remove(&(access, Reverse(id)));
                idx.release_slot(slot);
            }
            self.free_nodes.push(id);
        }
        blocks
    }

    /// On-disk node → resident at `block`; `false` if the node was not on
    /// disk. The slot is released unless `block` holds the record's own bytes
    /// (`restored`) and the tier keeps restored records.
    pub(super) fn nvme_rehome(&mut self, id: NodeId, block: u32, restored: bool) -> bool {
        if !self.is_on_disk(id) {
            return false;
        }
        let node = &mut self.nodes[id];
        node.block_idx = block;
        let (access, slot) = (node.last_access, node.nvme_slot);
        let Some(idx) = self.nvme.as_mut() else {
            return true;
        };
        idx.lru.remove(&(access, Reverse(id)));
        let half_free = idx.half_free(slot_class(slot, idx.classes));
        if !(restored && idx.keep_restored && half_free) {
            self.nodes[id].nvme_slot = u32::MAX;
            idx.release_slot(slot);
        }
        true
    }

    fn touch(&mut self, id: NodeId, access: u64) {
        let old = std::mem::replace(&mut self.nodes[id].last_access, access);
        if self.is_on_disk(id)
            && let Some(idx) = self.nvme.as_mut()
        {
            idx.lru.remove(&(old, Reverse(id)));
            idx.lru.insert((access, Reverse(id)));
        }
    }

    /// Walk the verified full-block path of `tokens`; returns node ids of the
    /// resident prefix followed by the on-disk run (stops at the first node
    /// that is missing, stale, or resident after disk).
    fn nvme_path(&self, tokens: &[u32], bs: usize, adapter_id: u64) -> (Vec<NodeId>, usize) {
        let mut ids = Vec::new();
        let mut resident = 0;
        let Some(mut cur) = self.root_for_read(adapter_id) else {
            return (ids, 0);
        };
        let mut ctx = 0u64;
        for chunk in tokens.chunks_exact(bs) {
            let expected = super::context_hash_combine(ctx, chunk);
            let Some(&child) = self.nodes[cur].children.get(chunk) else {
                break;
            };
            let n = &self.nodes[child];
            if n.context_hash != expected || n.ref_count == 0 {
                break;
            }
            let on_disk = n.block_idx == u32::MAX;
            if !on_disk && ids.len() > resident {
                break; // invariant: nothing resident below disk
            }
            if !on_disk {
                resident += 1;
            }
            ids.push(child);
            ctx = expected;
            cur = child;
        }
        (ids, resident)
    }

    pub(in crate::radix_tree) fn nvme_plan(
        &mut self,
        tokens: &[u32],
        bs: usize,
        adapter_id: u64,
    ) -> RestorePlan {
        if !self.nvme_on() {
            return RestorePlan::default();
        }
        let (ids, resident) = self.nvme_path(tokens, bs, adapter_id);
        if ids.len() == resident {
            return RestorePlan::default();
        }
        // Pin the whole path; refresh the resident part only. The on-disk
        // run is touched when it is restored (`nvme_complete`): a run the
        // caller declines keeps its LRU place, so the budget can still drop it.
        let access = self.next_access();
        for (i, &id) in ids.iter().enumerate() {
            self.nodes[id].ref_count += 1;
            if i < resident {
                self.touch(id, access);
            }
        }
        let disk = ids[resident..]
            .iter()
            .map(|&id| {
                let slot = self.nodes[id].nvme_slot;
                let tag = self.nvme.as_ref().map_or(0, |i| i.tags[slot as usize]);
                DiskRef { slot, tag }
            })
            .collect();
        RestorePlan {
            resident_tokens: resident * bs,
            disk,
            pinned_tokens: ids.len() * bs,
        }
    }

    pub(in crate::radix_tree) fn nvme_complete(
        &mut self,
        tokens: &[u32],
        bs: usize,
        adapter_id: u64,
        plan: &RestorePlan,
        restored: &[u32],
        failed: bool,
    ) -> Vec<u32> {
        let mut give_back = Vec::new();
        if plan.pinned_tokens == 0 {
            give_back.extend_from_slice(restored);
            return give_back;
        }
        let (ids, resident) = self.nvme_path(&tokens[..plan.pinned_tokens], bs, adapter_id);
        let run = &ids[resident.min(ids.len())..];
        let mut adopted = 0;
        // Adopt only while the run still matches the plan slot-for-slot; the
        // pin makes a mismatch impossible, but a mismatch must never place
        // bytes under the wrong node.
        let consistent = resident * bs == plan.resident_tokens;
        let access = self.next_access();
        for (i, &block) in restored.iter().enumerate() {
            let ok = consistent
                && run.get(i).is_some_and(|&id| {
                    plan.disk
                        .get(i)
                        .is_some_and(|d| self.nodes[id].nvme_slot == d.slot)
                });
            if !ok || !self.nvme_rehome(run[i], block, true) {
                give_back.extend_from_slice(&restored[i..]);
                break;
            }
            self.touch(run[i], access);
            adopted += 1;
        }
        if let Some(idx) = self.nvme.as_mut() {
            idx.stats.restores += adopted as u64;
        }
        if failed
            && adopted == restored.len()
            && consistent
            && let Some(&bad) = run.get(adopted)
        {
            give_back.extend(self.drop_subtree(bad));
            if let Some(idx) = self.nvme.as_mut() {
                idx.stats.restore_failures += 1;
            }
        }
        self.dec_refs(tokens, bs, plan.pinned_tokens, adapter_id);
        give_back
    }

    /// See `NvmePrefixTier::forget_kept`. A kept record belongs to a RESIDENT
    /// node, which is in no disk LRU: only the slot goes back.
    pub(in crate::radix_tree) fn nvme_forget_kept(
        &mut self,
        tokens: &[u32],
        bs: usize,
        adapter_id: u64,
        blocks: std::ops::Range<usize>,
    ) {
        if !self.nvme.as_ref().is_some_and(|idx| idx.keep_restored) {
            return;
        }
        let (ids, resident) = self.nvme_path(tokens, bs, adapter_id);
        for &id in ids[..resident].iter().take(blocks.end).skip(blocks.start) {
            let slot = std::mem::replace(&mut self.nodes[id].nvme_slot, u32::MAX);
            if slot != u32::MAX
                && let Some(idx) = self.nvme.as_mut()
            {
                idx.release_slot(slot);
            }
        }
    }

    pub(in crate::radix_tree) fn nvme_spill_failed(&mut self, orders: &[SpillOrder]) -> Vec<u32> {
        let mut give_back = Vec::new();
        for o in orders {
            let Some(idx) = self.nvme.as_mut() else {
                break;
            };
            idx.stats.spill_failures += 1;
            // Keyed on (slot, tag): the slot may already have been released
            // and re-issued to another node's spill (a later epoch, new tag)
            // by the time this failure is reported — never drop that node.
            let owner = idx.owner.get(o.slot as usize).copied().unwrap_or(NO_OWNER);
            let same_spill = idx.tags.get(o.slot as usize) == Some(&o.tag);
            // On disk ONLY: a RESIDENT owner was restored from this very
            // record since (kept records), which proves it verified — a run
            // is reported failed as a whole, its leading records may be fine.
            if owner != NO_OWNER
                && same_spill
                && self.nodes[owner].nvme_slot == o.slot
                && self.is_on_disk(owner)
            {
                give_back.extend(self.drop_subtree(owner));
            }
        }
        give_back
    }

    pub(in crate::radix_tree) fn nvme_stats(&self) -> NvmeStats {
        self.nvme
            .as_ref()
            .map_or_else(NvmeStats::default, |i| i.stats)
    }
}
