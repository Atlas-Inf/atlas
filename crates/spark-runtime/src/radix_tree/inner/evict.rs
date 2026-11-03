// SPDX-License-Identifier: AGPL-3.0-only

//! LRU eviction + entry counting for [`RadixTreeInner`]. Split out of
//! `inner.rs` during the ≤500-line cap enforcement; a child module of
//! `inner` so it keeps field access to the private node arena.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use super::{NodeId, RadixTreeInner};
use crate::prefix_cache::SpillOrder;

impl RadixTreeInner {
    /// Evict up to `num_blocks` LRU zero-ref leaf nodes.
    /// Returns physical block indices that were freed plus parallel
    /// disk-block IDs (Phase 6.1.e). When HSS isn't in use, every disk_id
    /// in the result is `u32::MAX` and the caller should ignore them; the
    /// public-trait wrapper filters those out into the returned
    /// `EvictedBlocks::disk_block_ids`.
    ///
    /// With the NVMe spill tier on, victims are spilled instead of deleted:
    /// the third vec lists the writes the caller must perform first.
    ///
    /// Victim order is the exact old total order `(last_access, node_id)`.
    ///
    /// Single-block calls retain the allocation-free linear victim search.
    /// Decode cannot amortize building a heap to select just one leaf.
    ///
    /// Bulk calls (`num_blocks > 1`) scan the candidate set ONCE: eligible
    /// leaves go into a min-heap keyed by `(last_access, node_id)` and each
    /// eviction pops the minimum and pushes the single parent that the pop may
    /// have made eligible. The old loop re-scanned the whole arena for every
    /// block, so a full 10^3–10^5-node pool drained N blocks in O(N·K); this is
    /// O(N + K·log N) and wins once K is large enough to amortise the heapify.
    pub(in crate::radix_tree) fn evict(
        &mut self,
        num_blocks: usize,
    ) -> (Vec<u32>, Vec<u32>, Vec<SpillOrder>) {
        let mut freed_phys = Vec::new();
        let mut freed_disk = Vec::new();
        if num_blocks == 0 {
            return (freed_phys, freed_disk, Vec::new());
        }
        if self.nvme_on() {
            let (phys, orders) = self.evict_spill(num_blocks);
            return (phys, freed_disk, orders);
        }
        if num_blocks == 1 {
            if let Some(node_id) = self.least_evictable_leaf() {
                let (block, disk) = self.unlink_evictable_leaf(node_id);
                freed_phys.push(block);
                freed_disk.push(disk);
            }
            return (freed_phys, freed_disk, Vec::new());
        }

        // Heap key `(Reverse(access), Reverse(id))` so `peek`/`pop` yield the
        // smallest `last_access`, breaking ties on the smallest arena index —
        // identical to the old linear `min`. Nodes are only ever each pushed
        // once (seeded here; later only a just-emptied parent), and a stale
        // push is impossible because eligibility only ever turns ON for the
        // parent of a removal and is re-checked at pop time.
        // Build the initial candidate vector and heapify once. Repeated
        // `push` would make initialization O(N log N), obscuring the point
        // of the one-pass rewrite; `BinaryHeap::from` is linear heapify.
        let candidates: Vec<(Reverse<u64>, Reverse<usize>)> = self
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(id, node)| {
                Self::evictable_leaf(node).then_some((Reverse(node.last_access), Reverse(id)))
            })
            .collect();
        let mut heap = BinaryHeap::from(candidates);

        while freed_phys.len() < num_blocks {
            let Some(&(Reverse(access), Reverse(node_id))) = heap.peek() else {
                break;
            };
            // The parent of an evicted child is pushed only after its removal;
            // nothing else changes eligibility mid-call. Guard anyway so a
            // heap entry that no longer qualifies can never free a live or
            // internally-linked node.
            if self.nodes[node_id].last_access != access
                || !Self::evictable_leaf(&self.nodes[node_id])
            {
                heap.pop();
                continue;
            }
            heap.pop();

            let parent = self.nodes[node_id].parent;
            let (block, disk) = self.unlink_evictable_leaf(node_id);
            freed_phys.push(block);
            freed_disk.push(disk);
            // The removal may have emptied `parent`; if it just became an
            // eligible leaf it is now the only newly-eligible node, so one
            // push maintains the candidate set.
            if let Some(parent_id) = parent
                && Self::evictable_leaf(&self.nodes[parent_id])
            {
                heap.push((
                    Reverse(self.nodes[parent_id].last_access),
                    Reverse(parent_id),
                ));
            }
        }

        (freed_phys, freed_disk, Vec::new())
    }

    /// Smallest `(last_access, arena index)` eligible leaf, or `None` when the
    /// arena holds no evictable leaf. This is the pre-rewrite victim rule; the
    /// single-block path and the heap-based bulk path agree on it exactly.
    fn least_evictable_leaf(&self) -> Option<NodeId> {
        let mut best: Option<(NodeId, u64)> = None;
        for (id, node) in self.nodes.iter().enumerate() {
            if !Self::evictable_leaf(node) {
                continue;
            }
            match best {
                None => best = Some((id, node.last_access)),
                Some((_, access)) if node.last_access < access => {
                    best = Some((id, node.last_access))
                }
                _ => {}
            }
        }
        best.map(|(id, _)| id)
    }

    /// Unlink `node_id` (which the caller just proved an eligible leaf) from
    /// the tree and return its `(block_idx, disk_block_id)` so both eviction
    /// paths share one copy of the removal logic.
    fn unlink_evictable_leaf(&mut self, node_id: NodeId) -> (u32, u32) {
        let block = self.nodes[node_id].block_idx;
        let disk = self.nodes[node_id].disk_block_id;
        if let Some(parent_id) = self.nodes[node_id].parent
            && let Some(key) = self.nodes[node_id].parent_key.clone()
        {
            self.nodes[parent_id].children.remove(&key);
        }
        self.nodes[node_id].block_idx = u32::MAX;
        self.nodes[node_id].disk_block_id = u32::MAX;
        self.nodes[node_id].children.clear();
        self.nodes[node_id].parent = None;
        self.nodes[node_id].parent_key = None;
        self.free_nodes.push(node_id);
        (block, disk)
    }

    /// A node the delete-path eviction may remove: a non-root node (`parent`
    /// set), holding a resident block, with only the cache's own reference
    /// (`ref_count <= 1`) and no children. Mirrors the old inline scan test.
    #[inline]
    fn evictable_leaf(node: &super::RadixNode) -> bool {
        node.parent.is_some()
            && node.block_idx != u32::MAX
            && node.ref_count <= 1
            && node.children.is_empty()
    }

    pub(in crate::radix_tree) fn num_entries(&self) -> usize {
        // Count non-root, non-deleted nodes. Roots (per adapter_id) and freed
        // nodes both carry `block_idx == u32::MAX`, so this filter excludes them.
        self.nodes
            .iter()
            .filter(|n| n.block_idx != u32::MAX)
            .count()
    }
}
