// SPDX-License-Identifier: AGPL-3.0-only
//
// CompactFsm — the immutable CSR form of `Fsm`.
// Port of `class CompactFSM` (+ `CompactFSM::Impl`) from `cpp/fsm.{h,cc}`.
//
// A `CompactFsm` stores its edges in a `Compact2DArray<FsmEdge>` so every
// state's edge list is contiguous in memory; outgoing edges are sorted by
// `(min, max, target)`. It is immutable — mutate by converting to `Fsm`,
// editing, and converting back.

use ahash::AHashSet;

use super::compact_array::Compact2DArray;
use super::edge::{FsmEdge, RepeatEdgeRef};
use super::fsm::{Fsm, NO_NEXT_STATE};
use super::traversal;

/// An immutable, CSR-packed finite-state machine.
///
/// The storage is shared: a clone is a reference, as `std::shared_ptr`
/// makes it upstream. Every rule's view of a grammar's `complete_fsm`
/// holds one, and a deep copy per rule made a grammar O(rules x states)
/// in memory — gigabytes for a strict object of ~100 required properties.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct CompactFsm {
    edges: std::sync::Arc<Compact2DArray<FsmEdge>>,
    edge_aux_data: std::sync::Arc<Vec<i32>>,
}

impl CompactFsm {
    /// Build directly from a CSR edge array and aux data.
    pub fn new(edges: Compact2DArray<FsmEdge>, edge_aux_data: Vec<i32>) -> Self {
        Self {
            edges: std::sync::Arc::new(edges),
            edge_aux_data: std::sync::Arc::new(edge_aux_data),
        }
    }

    /* ----------------------- visitors ----------------------- */

    /// Number of states.
    pub fn num_states(&self) -> usize {
        self.edges.len()
    }

    /// The CSR edge array.
    pub fn all_edges(&self) -> &Compact2DArray<FsmEdge> {
        &self.edges
    }

    /// Outgoing edges of `state` as a contiguous slice.
    pub fn edges(&self, state: usize) -> &[FsmEdge] {
        self.edges.row(state)
    }

    /// The repeat-edge auxiliary data buffer.
    pub fn edge_aux_data(&self) -> &[i32] {
        &self.edge_aux_data
    }

    /// Replace the repeat-edge auxiliary data buffer.
    pub fn set_edge_aux_data(&mut self, data: Vec<i32>) {
        self.edge_aux_data = std::sync::Arc::new(data);
    }

    /// Decode the repeat-edge info stored at aux index `idx`.
    pub fn repeat_edge_info(&self, idx: i16) -> RepeatEdgeRef {
        RepeatEdgeRef::from_aux(&self.edge_aux_data, idx as usize)
    }

    /// Total edges across all states.
    pub fn num_edges(&self) -> usize {
        self.edges.total_elems()
    }

    /// True if `self` and `other` are the same storage (clones of one FSM).
    pub fn shares_storage_with(&self, other: &CompactFsm) -> bool {
        std::sync::Arc::ptr_eq(&self.edges, &other.edges)
    }

    /// Approximate heap footprint in bytes.
    pub fn memory_size(&self) -> usize {
        self.edges.memory_size() + self.edge_aux_data.len() * std::mem::size_of::<i32>()
    }

    /* ----------------------- traversal ----------------------- */

    /// Materialize the adjacency rows as `Vec<Vec<FsmEdge>>` so the
    /// shared traversal helpers can run. Edges are already sorted.
    fn adjacency(&self) -> Vec<Vec<FsmEdge>> {
        self.edges.iter_rows().map(|r| r.to_vec()).collect()
    }

    /// Collect every transition target from `from` for `value` along
    /// `edge_type` into `targets` (cleared first). Because the compact
    /// rows are sorted, this could early-break, but for simplicity it
    /// scans the (small) row fully — behavior is identical.
    pub fn next_states(&self, from: usize, value: i32, edge_type: i16, targets: &mut Vec<i32>) {
        targets.clear();
        for edge in self.edges(from) {
            if traversal::edge_matches(edge, value, edge_type) {
                targets.push(edge.target);
            }
        }
    }

    /// First transition target, or [`NO_NEXT_STATE`].
    pub fn next_state(&self, from: usize, value: i32, edge_type: i16) -> i32 {
        let mut t = Vec::new();
        self.next_states(from, value, edge_type, &mut t);
        t.first().copied().unwrap_or(NO_NEXT_STATE)
    }

    /// Epsilon closure (in place); `state_set` is *not* cleared.
    pub fn epsilon_closure(&self, state_set: &mut AHashSet<i32>) {
        traversal::epsilon_closure(&self.adjacency(), state_set);
    }

    /// Advance a set of states — see [`Fsm::advance`].
    pub fn advance(
        &self,
        from: &AHashSet<i32>,
        value: i32,
        result: &mut AHashSet<i32>,
        edge_type: i16,
        from_is_closure: bool,
    ) {
        traversal::advance(
            &self.adjacency(),
            from,
            value,
            result,
            edge_type,
            from_is_closure,
        );
    }

    /// All rule ids referenced by `state`'s outgoing edges.
    pub fn possible_rules(&self, state: usize, rules: &mut AHashSet<i32>) {
        rules.clear();
        for edge in self.edges(state) {
            if edge.is_rule_ref() {
                rules.insert(edge.ref_rule_id());
            }
        }
    }

    /// All states reachable from `from`. `result` is cleared first.
    ///
    /// Walks the packed rows in place, as `Fsm::reachable_states` walks
    /// its own. Expanding the whole FSM first cost O(states) per call, and
    /// eager mask warm-up calls this once per rule of a grammar whose
    /// rules share one FSM: O(rules x states).
    pub fn reachable_states(&self, from: &[i32], result: &mut AHashSet<i32>) {
        result.clear();
        let mut queue: std::collections::VecDeque<i32> = std::collections::VecDeque::new();
        for &s in from {
            if result.insert(s) {
                queue.push_back(s);
            }
        }
        while let Some(cur) = queue.pop_front() {
            for edge in self.edges(cur as usize) {
                if result.insert(edge.target) {
                    queue.push_back(edge.target);
                }
            }
        }
    }

    /* --------------------- construction ---------------------- */

    /// Expand back into a mutable [`Fsm`].
    pub fn to_fsm(&self) -> Fsm {
        let edges: Vec<Vec<FsmEdge>> = self.edges.iter_rows().map(|r| r.to_vec()).collect();
        Fsm::from_edges(edges, self.edge_aux_data.to_vec())
    }
}

impl Fsm {
    /// Sort edges and pack into the CSR [`CompactFsm`] form.
    pub fn to_compact(&self) -> CompactFsm {
        let mut sorted = self.clone();
        sorted.sort_edges();
        let mut arr: Compact2DArray<FsmEdge> = Compact2DArray::new();
        for row in &sorted.edges {
            arr.push_row(row);
        }
        CompactFsm::new(arr, sorted.edge_aux_data)
    }
}

#[cfg(test)]
#[path = "compact_tests.rs"]
mod tests;
