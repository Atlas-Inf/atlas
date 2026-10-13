// SPDX-License-Identifier: AGPL-3.0-only

//! Finish leaves in the snapshot index (`ATLAS_GLM_PC_FINISH_LEAF`, which
//! needs `ATLAS_GLM_PC_EVICT`; nothing here runs with it off).
//!
//! A finish leaf is the state a turn had at the last block boundary its
//! decode crossed. It serves only the next turn of the same conversation, and
//! only when that turn's prompt reproduces the output up to the boundary, so
//! it must never cost a conversation a restore point it would otherwise have:
//!
//! * it is evicted before any chain frontier or branch point, and a leaf is
//!   given a slot only from the free list, dead history or another leaf
//!   ([`SsmSnapshotIndex::evict_for_leaf`]), never from a frontier;
//! * it is not linked into a chain and is no parent, so the prefill
//!   checkpoint under it stays its conversation's protected frontier until
//!   the next prefill checkpoint supersedes it. That checkpoint also makes
//!   the leaf dead history (`link_chain`): the turn it was saved for has run;
//! * a checkpoint already registered at its prefix wins.
//!
//! A decoding sequence registers its leaf from the first save and moves it
//! forward boundary by boundary ([`SsmSnapshotIndex::take_leaf`], then
//! [`SsmSnapshotIndex::insert_leaf`] at the new one), so the slot is in the
//! index the whole time and the rules above hold for it too: no slot is held
//! outside the index. A leaf is unreachable until its turn finishes and
//! caches the blocks under it, and is a valid state for its prefix whenever
//! it is reached.
//!
//! The turn a leaf was saved for settles it when it restores it
//! ([`SsmSnapshotIndex::settle_leaf`]). If that turn saves a checkpoint of
//! its own, the leaf is dead history from then on, so the turn's own slot
//! needs come out of its own leaf and not out of another conversation's. If
//! it saves none (its new message is shorter than the tail gap), the leaf is
//! promoted to an ordinary checkpoint and the conversation's frontier keeps
//! moving.

use super::hash_token_prefix;
use super::snapshot::SsmSnapshotIndex;
use super::snapshot_chain::{ChainMeta, DEAD};

impl SsmSnapshotIndex {
    /// Register `snapshot_id` as the finish leaf at `token_count` tokens.
    /// Returns a slot for the caller to free: the leaf this one replaces, or
    /// `snapshot_id` itself when a checkpoint already serves the prefix.
    pub(super) fn insert_leaf(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        let at = self
            .entries
            .iter()
            .position(|e| e.prefix_hash == prefix_hash);
        if at.is_some_and(|i| !self.entries[i].chain.leaf) {
            return Some(snapshot_id);
        }
        let displaced = self.insert(prefix_hash, snapshot_id, session_hash, token_count);
        // `insert` re-homes the entry in place or pushes a new one last.
        let i = at.unwrap_or(self.entries.len() - 1);
        self.entries[i].chain = ChainMeta {
            leaf: true,
            ..Default::default()
        };
        displaced
    }

    /// Unregister the leaf at `prefix_hash` if it still holds `snapshot_id`,
    /// handing the slot back to the caller. `false`: the leaf was evicted,
    /// promoted or replaced since, and the slot is no longer the caller's.
    pub(super) fn take_leaf(&mut self, prefix_hash: u64, snapshot_id: usize) -> bool {
        let mine = |e: &super::snapshot::SnapshotEntry| {
            e.prefix_hash == prefix_hash
                && e.chain.leaf
                && !e.tiered
                && e.snapshot_id == snapshot_id
        };
        let at = self.entries.iter().position(mine);
        at.map(|i| self.entries.swap_remove(i)).is_some()
    }

    /// The finish leaf registered for exactly `tokens`, if any, was restored
    /// by the turn it was saved for. `keep`: that turn saves no checkpoint of
    /// its own, so the leaf becomes an ordinary checkpoint, linked into its
    /// chain. Otherwise it is dead history.
    pub(super) fn settle_leaf(&mut self, tokens: &[u32], adapter_id: u64, keep: bool) {
        let hash = hash_token_prefix(tokens, tokens.len(), adapter_id);
        let leaf = self
            .entries
            .iter_mut()
            .find(|e| e.prefix_hash == hash && e.chain.leaf);
        match leaf {
            Some(e) if keep => {
                e.chain = ChainMeta::default();
                self.link_chain(tokens, adapter_id, hash);
            }
            Some(e) => e.chain.superseded = true,
            None => {}
        }
    }

    /// Evict one resident snapshot to make room for a finish leaf: dead
    /// history first, then another leaf, least recently used first. `None`
    /// when only frontiers and branch points remain.
    pub(super) fn evict_for_leaf(&mut self) -> Option<usize> {
        let i = self.chain_victim(true)?;
        if self.entries[i].chain.class() > DEAD + 1 {
            return None;
        }
        self.stats.evictions += 1;
        Some(self.entries.swap_remove(i).snapshot_id)
    }
}

#[cfg(test)]
#[path = "tests/snapshot_leaf.rs"]
mod tests;
