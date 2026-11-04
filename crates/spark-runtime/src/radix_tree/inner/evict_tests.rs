// SPDX-License-Identifier: AGPL-3.0-only

//! Private differential oracle for the radix delete-path rewrite.

use super::RadixTreeInner;

impl RadixTreeInner {
    /// Literal pre-rewrite loop, retained only for differential tests.
    pub(in crate::radix_tree) fn evict_linear_scan(
        &mut self,
        num_blocks: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        let mut physical = Vec::new();
        let mut disk = Vec::new();
        if num_blocks == 0 {
            return (physical, disk);
        }
        if self.nvme_on() {
            let (p, _) = self.evict_spill(num_blocks);
            return (p, disk);
        }
        while physical.len() < num_blocks {
            let mut best: Option<(usize, u64)> = None;
            for (id, node) in self.nodes.iter().enumerate() {
                if node.parent.is_none()
                    || node.ref_count > 1
                    || !node.children.is_empty()
                    || node.block_idx == u32::MAX
                {
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
            let Some((id, _)) = best else { break };
            physical.push(self.nodes[id].block_idx);
            disk.push(self.nodes[id].disk_block_id);
            let parent = self.nodes[id].parent;
            if let Some(parent_id) = parent
                && let Some(key) = self.nodes[id].parent_key.clone()
            {
                self.nodes[parent_id].children.remove(&key);
            }
            self.nodes[id].block_idx = u32::MAX;
            self.nodes[id].disk_block_id = u32::MAX;
            self.nodes[id].children.clear();
            self.nodes[id].parent = None;
            self.nodes[id].parent_key = None;
            self.free_nodes.push(id);
        }
        (physical, disk)
    }

    pub(in crate::radix_tree) fn structure_snapshot(&self) -> String {
        use std::fmt::Write;
        let mut out = String::new();
        for (id, n) in self.nodes.iter().enumerate() {
            let mut children: Vec<_> = n.children.iter().map(|(k, v)| (k.clone(), *v)).collect();
            children.sort_by(|a, b| a.0.cmp(&b.0));
            let _ = writeln!(
                out,
                "node {id}: block={} disk={} nvme={} refs={} access={} parent={:?} key={:?} children={children:?}",
                n.block_idx,
                n.disk_block_id,
                n.nvme_slot,
                n.ref_count,
                n.last_access,
                n.parent,
                n.parent_key,
            );
        }
        let mut free = self.free_nodes.clone();
        free.sort_unstable();
        let _ = writeln!(out, "free={free:?} counter={}", self.access_counter);
        let mut roots: Vec<_> = self.roots.iter().map(|(k, v)| (*k, *v)).collect();
        roots.sort_unstable();
        let _ = writeln!(out, "roots={roots:?}");
        out
    }

    pub(in crate::radix_tree) fn force_test_access(&mut self, access: u64) {
        for node in &mut self.nodes {
            if node.block_idx != u32::MAX {
                node.last_access = access;
            }
        }
    }
}

/// Repeated single-block eviction must free the same leaves and newly exposed
/// parent as a bulk call, leaving the same tree available for prefix reuse.
#[test]
fn single_block_fast_path_matches_batch_evict() {
    use crate::prefix_cache::PrefixCache;
    use crate::radix_tree::RadixTree;

    let build = |tree: &RadixTree| {
        let root: Vec<u32> = (0..16u32).collect();
        tree.insert(&root, &[10], &[], 16, 0, 0);
        tree.release(&root, 16, 0);
        for (first, block) in [(100u32, 20u32), (200, 30), (300, 40), (400, 50)] {
            let mut tokens = root.clone();
            tokens.extend(first..first + 16);
            tree.insert(&tokens, &[10, block], &[], 16, 0, 0);
            tree.release(&tokens, 16, 0);
        }
    };
    let step_tree = RadixTree::new();
    let batch_tree = RadixTree::new();
    build(&step_tree);
    build(&batch_tree);
    let mut stepped = Vec::new();
    for _ in 0..5 {
        stepped.extend(step_tree.evict(1).physical);
    }
    let batched = batch_tree.evict(5).physical;
    assert_eq!(stepped, vec![20, 30, 40, 50, 10]);
    assert_eq!(stepped, batched);
    assert_eq!(
        step_tree.inner.lock().structure_snapshot(),
        batch_tree.inner.lock().structure_snapshot(),
    );
}
