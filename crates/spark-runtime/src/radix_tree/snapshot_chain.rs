// SPDX-License-Identifier: AGPL-3.0-only

//! Chain-aware SSM snapshot retention (`ATLAS_GLM_PC_EVICT=1`, default off).
//!
//! # The problem with the default victim
//!
//! The default victim (`session_aware_victim`) groups entries by
//! `session_hash`, the hash of the first 1024 prompt tokens. Agentic clients
//! (omp, Claude-Code-like tools) open every session with the same system
//! prompt and tools, so every session lands in ONE group and the policy is
//! plain LRU over entries. Each turn saves one snapshot (its prompt tail), so
//! a session looping on fast tool calls fills all 16 slots with its own
//! history — every snapshot but its newest is useless to it — and evicts the
//! only restore point of each slower session, whose next turn then
//! re-prefills its whole 30-50K transcript.
//!
//! Also, on TP2 the worker rank never learns `session_hash` (it stays 0 on
//! every sequence the worker builds), so the two ranks can group differently
//! and pick different victims.
//!
//! # The policy
//!
//! Every snapshot joins a *chain*, one conversation's growing path, when it is
//! inserted. Its deepest ancestor, the deepest resident snapshot whose token
//! prefix is a prefix of the new one, decides which chain it joins:
//!
//! * no ancestor: the snapshot starts a new chain;
//! * the ancestor is its chain's deepest member: the new snapshot extends
//!   that chain, and the ancestor becomes *superseded*, because the next
//!   turn of that conversation restores from the deeper one;
//! * the chain already goes deeper along another path: two continuations
//!   diverge here, so the ancestor becomes a *branch* point and the new
//!   snapshot starts a new chain.
//!
//! Victims are then taken from superseded, non-branch entries first, least
//! recently used first. Chain frontiers and branch points go only after that,
//! also least recently used first. So a session owns one protected slot, not
//! sixteen, and an idle session loses its frontier only after every
//! conversation's dead history is gone and it is the stalest frontier left.
//!
//! `session_hash` is deliberately unused, so it no longer makes the ranks
//! choose differently. Victim identity is still NOT guaranteed identical
//! across ranks: a rank-local recency bump (the F83 re-lookup on a rank whose
//! match was capped, a lookup win the other rank does not have, or
//! [`SsmSnapshotIndex::resident_at`] on a rank restoring shallower than it
//! could) reorders that rank's LRU. The TP2 safety mechanism is the
//! restore-depth agreement in `spark-model` (`pc_policy`): every rank
//! restores at one agreed depth or all recompute, whatever each pool holds.
//!
//! With the flag on this victim replaces both the session-aware one and
//! `ATLAS_SNAP_EVICT_LEGACY`, and the tail lease is not consulted: tail
//! entries are never linked, so they rank as protected frontiers anyway.
//!
//! Finish leaves (`snapshot_leaf`) are a third class between the two: never
//! linked, evicted after dead history and before any frontier or branch, and
//! dead history themselves once a deeper checkpoint joins their path.
//!
//! Credit: placing the deepest reusable state where a lookup can reach it,
//! and a depth-first, reuse-protected victim order, follow Reederey87's
//! prefix-cache overlays (Artem Matskevych,
//! <https://github.com/Reederey87/glm53-flash-exl3-2x-dgx-spark>,
//! `overlay/patch_apc_tail_boundary.py`, `overlay/cache_tail_evict.py`,
//! `overlay/patch_cache_hot_protect.py`; Apache-2.0, ideas only, no code),
//! and branch points follow Marconi (MLSys'25, arXiv:2411.19379). The
//! chain / superseded / one-protected-slot-per-session policy is ours.

use std::sync::OnceLock;

use super::snapshot::SsmSnapshotIndex;
use super::{hash_token_prefix, prefix_hash_push, prefix_hash_seed};

/// `ATLAS_GLM_PC_EVICT=1`: chain-aware snapshot eviction. Read once per
/// process; off leaves every victim choice byte-identical to base.
pub fn glm_pc_evict_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PC_EVICT").as_deref() == Ok("1"))
}

/// Per-entry chain state. `id == 0` means "never linked" (flag off, or a
/// tail/sibling entry, which are not linked); such an entry is ranked as a
/// frontier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct ChainMeta {
    /// Chain id: the prefix hash of the chain's first snapshot.
    pub id: u64,
    /// A deeper snapshot on this same path continues the chain.
    pub superseded: bool,
    /// Continuations diverge at or after this snapshot.
    pub branch: bool,
    /// A finish leaf (`snapshot_leaf`): never linked, never a parent.
    pub leaf: bool,
}

impl ChainMeta {
    /// Eviction class, lowest first: [`DEAD`] history (superseded, not a
    /// branch point), finish leaves, then chain frontiers and branch points.
    pub(super) fn class(&self) -> u8 {
        match (self.superseded && !self.branch, self.leaf) {
            (true, _) => DEAD,
            (false, true) => DEAD + 1,
            (false, false) => DEAD + 2,
        }
    }
}

/// The eviction class of superseded history.
pub(super) const DEAD: u8 = 0;

impl SsmSnapshotIndex {
    /// [`Self::link_chain`] when `ATLAS_GLM_PC_EVICT=1`, else nothing.
    pub(super) fn link_chain_if_enabled(&mut self, tokens: &[u32], adapter_id: u64, hash: u64) {
        if glm_pc_evict_enabled() {
            self.link_chain(tokens, adapter_id, hash);
        }
    }

    /// Link the entry just registered for `tokens` (prefix hash `hash`) into
    /// its chain. A re-save of an already-linked prefix keeps its links.
    pub(super) fn link_chain(&mut self, tokens: &[u32], adapter_id: u64, hash: u64) {
        let len = tokens.len();
        let Some(me) = self.entries.iter().position(|e| e.prefix_hash == hash) else {
            return;
        };
        if self.entries[me].chain.id != 0 {
            return;
        }
        // Hash every candidate depth in one pass over `tokens`.
        let mut shallower: Vec<usize> = (0..self.entries.len())
            .filter(|&i| i != me && self.entries[i].token_count < len)
            .collect();
        shallower.sort_by_key(|&i| self.entries[i].token_count);
        let (mut h, mut at) = (prefix_hash_seed(adapter_id), 0usize);
        let mut parent: Option<usize> = None;
        for i in shallower {
            let depth = self.entries[i].token_count;
            h = tokens[at..depth]
                .iter()
                .fold(h, |h, &t| prefix_hash_push(h, t));
            at = depth;
            if h != self.entries[i].prefix_hash {
                continue;
            }
            if self.entries[i].chain.leaf {
                // A finish leaf is no parent (it would shadow the checkpoint
                // below it); under a deeper checkpoint it has served its turn.
                self.entries[i].chain.superseded = true;
            } else {
                parent = Some(i); // sorted ascending: the last hit is the deepest
            }
        }
        let Some(p) = parent else {
            self.entries[me].chain.id = hash;
            return;
        };
        if self.entries[p].chain.id == 0 {
            self.entries[p].chain.id = self.entries[p].prefix_hash;
        }
        let (pid, pdepth) = (self.entries[p].chain.id, self.entries[p].token_count);
        // A deeper member of the parent's chain cannot be on this path (the
        // parent is the deepest ancestor), so the chain forks here.
        let forks = self
            .entries
            .iter()
            .enumerate()
            .any(|(i, e)| i != me && e.chain.id == pid && e.token_count > pdepth);
        let parent = &mut self.entries[p].chain;
        if forks {
            parent.branch = true;
            parent.superseded = false;
            self.entries[me].chain.id = hash;
        } else {
            parent.superseded = !parent.branch;
            self.entries[me].chain.id = pid;
        }
    }

    /// Mark the entry at `hash` as a branch point (never superseded).
    pub(super) fn mark_branch(&mut self, hash: u64) {
        if let Some(e) = self.entries.iter_mut().find(|e| e.prefix_hash == hash) {
            e.chain.branch = true;
            e.chain.superseded = false;
        }
    }

    /// Chain-aware victim: superseded non-branch entries first, then finish
    /// leaves, then frontiers and branch points ([`ChainMeta::class`]); least
    /// recently used within each class. `last_access` values are unique, so
    /// the choice is total.
    pub(super) fn chain_victim(&self, skip_tiered: bool) -> Option<usize> {
        self.entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !(skip_tiered && e.tiered))
            .min_by_key(|(_, e)| (e.chain.class(), e.last_access))
            .map(|(i, _)| i)
    }

    /// The resident, exact-prefix (non-tail) snapshot at exactly `depth`
    /// tokens of `tokens`. A hit counts as a use (recency bump), as a lookup
    /// win does: the rank that restores here through the agreement then bumps
    /// the same entry the shallower rank's lookup bumped.
    pub(super) fn resident_at(
        &mut self,
        tokens: &[u32],
        depth: usize,
        adapter_id: u64,
    ) -> Option<usize> {
        if depth == 0 || depth > tokens.len() {
            return None;
        }
        let hash = hash_token_prefix(tokens, depth, adapter_id);
        let i = self.entries.iter().position(|e| {
            !e.tiered && !e.is_tail && e.token_count == depth && e.prefix_hash == hash
        })?;
        self.access_counter += 1;
        self.entries[i].last_access = self.access_counter;
        Some(self.entries[i].snapshot_id)
    }
}

#[cfg(test)]
#[path = "tests/snapshot_chain.rs"]
mod tests;
