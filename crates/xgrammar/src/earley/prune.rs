// SPDX-License-Identifier: AGPL-3.0-only
//
// Dynamic Earley dead-state pruning — Tier 3a.
//
// ZapFormat-style pruning (arXiv:2506.01151, "Formatron / Earley-Driven
// Dynamic Pruning"). After every `advance` the parser's live scanable
// set accumulates states that can no longer lead to acceptance. Each one
// is re-scanned, re-predicted and walked for mask generation on every
// subsequent token — wasted work. This module identifies the provably
// dead ones and drops them before they enter the scanable history.
//
// The criterion (precise, conservative — see `is_node_productive`):
//
//   A live scanable Earley state sits at FSM node `n` of rule `r`. For
//   `r` to ever complete from `n` — and hence for any parent waiting on
//   `r` to advance — there must exist a path of FSM transitions from `n`
//   to an *end node* of `r`. If no such path exists, node `n` is
//   "non-productive": the rule can never finish from there, so the state
//   contributes to no accepting parse and is safe to drop.
//
// Productivity is a backward reachability over each per-rule FSM, the
// classic automaton "co-accessibility" analysis. It depends only on the
// grammar's FSM topology, so it is computed once per grammar
// (`GrammarData::productivity`) and shared by every parser over it — the
// matcher reuses one parser across tokens, but mask generation builds one
// per adaptive mask — and consulted with O(1) bitset lookups thereafter.
//
// Correctness: every FSM edge — char-range, epsilon, rule-ref,
// repeat-ref — is treated uniformly as "node `n` may transition to
// `edge.target`". Treating a rule-ref/repeat-ref edge as freely
// traversable is *conservative*: the referenced rule might not actually
// complete, so a node marked productive purely through such an edge may
// in truth be dead. That only ever causes a *missed* prune, never a
// wrong one. A node is marked non-productive only when NO edge of any
// kind can reach an end node — which is unconditionally dead. Pruning is
// therefore language-preserving: the accepted set, `is_completed`, and
// the acceptable-byte mask are byte-identical with and without it.
//
// A pruned state is one that is in the scanable history yet can never be
// scanned-then-completed. Dropping it removes it from the next
// `advance`'s scan loop and from every `acceptable_*` walk — exactly the
// wasted work ZapFormat targets — without changing any observable
// result, because a non-productive scanable state can contribute neither
// a completion nor an acceptable byte that leads anywhere.

use bitvec::vec::BitVec;

use super::state::ParserState;
use crate::grammar::GrammarData;

/// Master on/off switch for dead-state pruning. Pruning is a pure
/// optimization — flipping this to `false` must not change any
/// accept/reject decision or mask bit, only performance. Kept as a
/// `const` safety hatch (and so the no-prune path stays compiled).
pub(crate) const PRUNE_DEAD_STATES: bool = true;

/// Per-rule co-accessibility (productivity) bitsets.
///
/// `productive[r] = (base, bits)`: one bit per node of rule `r`'s FSM,
/// which occupies backing nodes `base ..` (a per-rule view of the
/// grammar's shared FSM); the bit is set when that node can reach an end
/// node of `r`. A rule with no FSM (`per_rule_fsms[r] == None`) gets an
/// empty bitset and is never consulted (its states are FSM-less and not
/// pruned).
#[derive(Debug, Clone, Default)]
pub(crate) struct ProductivityTable {
    productive: Vec<(usize, BitVec)>,
}

impl ProductivityTable {
    /// Build the table for `grammar` by running backward reachability on
    /// every per-rule FSM, over that rule's own nodes only. Runs once per
    /// grammar (`GrammarData::productivity`).
    pub(crate) fn build(grammar: &GrammarData) -> Self {
        let productive = grammar
            .per_rule_fsms
            .iter()
            .map(|fsm| match fsm {
                Some(f) => (f.base(), co_accessible_nodes(f)),
                None => (0, BitVec::new()),
            })
            .collect();
        Self { productive }
    }

    /// Approximate heap footprint in bytes.
    pub(crate) fn memory_size(&self) -> usize {
        self.productive
            .iter()
            .map(|(_, bits)| std::mem::size_of::<(usize, BitVec)>() + bits.len().div_ceil(8))
            .sum()
    }

    /// True if FSM node `node` of rule `rule_id` can reach an end node —
    /// i.e. the rule can still complete from there. Out-of-range rule or
    /// node ids (including nodes outside the rule's FSM) return `true`
    /// (conservative: never prune what we cannot positively classify as
    /// dead).
    pub(crate) fn is_node_productive(&self, rule_id: i32, node: i32) -> bool {
        if rule_id < 0 || node < 0 {
            return true;
        }
        let Some((base, bits)) = self.productive.get(rule_id as usize) else {
            return true;
        };
        (node as usize)
            .checked_sub(*base)
            .and_then(|i| bits.get(i))
            .is_none_or(|b| *b)
    }

    /// True if a live scanable `state` is provably dead — its FSM node
    /// can never reach its rule's end node, so no continuation completes
    /// the rule. FSM-less states (`rule_id == -1`) are never dead here:
    /// their productivity is not tracked, so they are kept conservatively.
    pub(crate) fn is_state_dead(&self, state: &ParserState) -> bool {
        debug_assert!(
            state.rule_id == -1
                || self
                    .productive
                    .get(state.rule_id as usize)
                    .is_none_or(|(base, bits)| {
                        bits.is_empty()
                            || (state.element_id as usize)
                                .checked_sub(*base)
                                .is_some_and(|i| i < bits.len())
                    }),
            "state {state:?} outside its rule's FSM nodes"
        );
        state.rule_id != -1 && !self.is_node_productive(state.rule_id, state.element_id)
    }

    /// Remove every provably-dead state from `states` in place. A pure
    /// optimization: the retained set parses exactly the same language.
    /// No-ops when pruning is disabled or nothing is dead (the common
    /// case — the scan avoids the `retain` rewrite when it would be a
    /// copy of the same vector).
    pub(crate) fn prune(&self, states: &mut Vec<ParserState>) {
        if !PRUNE_DEAD_STATES || states.is_empty() {
            return;
        }
        if states.iter().any(|s| self.is_state_dead(s)) {
            states.retain(|s| !self.is_state_dead(s));
        }
    }
}

/// Backward reachability on one per-rule FSM: the set of its nodes (bit i
/// is node `base + i`) that can reach one of its end nodes. A reverse
/// breadth-first walk from the end nodes over the rule's own edges, linear
/// in the rule's size: the shared FSM's other rules are never visited (a
/// rule's edges stay within its nodes).
fn co_accessible_nodes(fsm: &crate::fsm::CompactFsmWithStartEnd) -> BitVec {
    let (base, ends) = (fsm.base(), fsm.ends());
    let n = ends.len();
    let mut predecessors: Vec<Vec<u32>> = vec![Vec::new(); n];
    for i in 0..n {
        for edge in fsm.fsm().edges(base + i) {
            if edge.target < 0 {
                continue;
            }
            if let Some(j) = (edge.target as usize).checked_sub(base).filter(|&j| j < n) {
                predecessors[j].push(i as u32);
            }
        }
    }
    let mut productive: BitVec = BitVec::repeat(false, n);
    let mut stack: Vec<usize> = (0..n).filter(|&i| ends[i]).collect();
    for &i in &stack {
        productive.set(i, true);
    }
    while let Some(j) = stack.pop() {
        for &i in &predecessors[j] {
            if !productive[i as usize] {
                productive.set(i as usize, true);
                stack.push(i as usize);
            }
        }
    }
    productive
}
