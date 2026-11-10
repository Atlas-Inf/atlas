// SPDX-License-Identifier: AGPL-3.0-only

//! Restore ladder: keep a few SSM snapshots at log-spaced depths behind each
//! live conversation's newest save (`ATLAS_SSM_SNAP_LADDER`, default on).
//!
//! # The problem
//!
//! An SSM prefix hit needs a resident snapshot at or below the KV match. A
//! warm agentic turn saves ~3 snapshots near its tail, and the default victim
//! is least-recently-used, so once a conversation outgrows the pool every
//! resident snapshot sits within the last few turns. A client that rewrites
//! history a few thousand tokens back (pruned tool output, a re-rendered
//! message) then matches the KV radix up to the rewrite but finds no snapshot
//! below it, and the prefill recomputes the whole prompt: 317 s TTFT at 178K
//! tokens on a DGX Spark (2026-11-09, OMP on Qwen3.8-27B), where the
//! rewrite itself cost ~8K tokens.
//!
//! # The policy
//!
//! After every save on a path (and keyed by its session, or by its first
//! 1024 tokens when the session is unknown), the resident snapshots on that
//! exact token path are found in one hashing pass, and the shallowest one in
//! each distance bucket `[base·2^k, base·2^(k+1))` behind the save becomes a
//! rung. Rungs are victims only after every non-rung, non-leased entry is
//! gone. A rewrite `d` tokens back then replays at most ~4·`d` tokens, not
//! the whole prompt. Rungs of the [`LADDER_PATHS`] most recent paths are
//! kept; each new save on a path replaces that path's rungs, so the ladder
//! costs ~log2(prompt/base) slots per live conversation and dead branches
//! age out by recency.
//!
//! Rung choice depends only on token content, save order and session, so
//! TP ranks that see the same saves agree; the TP2 worker (session 0) keys
//! by tokens instead, and `pc_policy`'s restore-depth agreement covers any
//! remaining divergence as it does for rank-local recency.

use std::sync::OnceLock;

use super::snapshot::SsmSnapshotIndex;
use super::{hash_token_prefix, prefix_hash_push, prefix_hash_seed};

/// Conversations whose ladders are kept at once.
pub(super) const LADDER_PATHS: usize = 4;

/// Tokens hashed to key a path whose session is unknown.
const PATH_KEY_TOKENS: usize = 1024;

/// Closest rung distance in tokens; `None` when the ladder is off
/// (`ATLAS_SSM_SNAP_LADDER=0`/`off`). Default 1024. Read once.
pub fn ssm_snap_ladder_base() -> Option<usize> {
    static BASE: OnceLock<Option<usize>> = OnceLock::new();
    *BASE.get_or_init(|| match std::env::var("ATLAS_SSM_SNAP_LADDER").as_deref() {
        Ok("0") | Ok("off") | Ok("false") => None,
        Ok(v) => Some(v.parse().ok().filter(|&b: &usize| b > 0).unwrap_or(1024)),
        Err(_) => Some(1024),
    })
}

#[derive(Default)]
pub(crate) struct Ladder {
    /// Save counter, the recency of a path's ladder.
    generation: u64,
    /// (path key, generation of its last save, rung prefix hashes).
    paths: Vec<(u64, u64, Vec<u64>)>,
}

impl Ladder {
    /// Whether the snapshot with this prefix hash is a rung of a kept path.
    pub(crate) fn holds(&self, prefix_hash: u64) -> bool {
        self.paths
            .iter()
            .any(|(_, _, rungs)| rungs.contains(&prefix_hash))
    }

    fn set(&mut self, key: u64, rungs: Vec<u64>) {
        self.generation += 1;
        let generation = self.generation;
        if let Some(path) = self.paths.iter_mut().find(|p| p.0 == key) {
            *path = (key, generation, rungs);
            return;
        }
        if self.paths.len() >= LADDER_PATHS
            && let Some(oldest) = (0..self.paths.len()).min_by_key(|&i| self.paths[i].1)
        {
            self.paths.swap_remove(oldest);
        }
        self.paths.push((key, generation, rungs));
    }
}

impl SsmSnapshotIndex {
    /// Recompute the rungs of the path ending at `tokens` (a save just made
    /// at `tokens.len()`). No-op when the ladder is off.
    pub(super) fn stamp_ladder(&mut self, tokens: &[u32], session_hash: u64, adapter_id: u64) {
        let Some(base) = ssm_snap_ladder_base() else {
            return;
        };
        let frontier = tokens.len();
        if frontier <= base {
            return;
        }
        // Resident entries on this path, found in one pass over `tokens`.
        let mut by_depth: Vec<usize> = (0..self.entries.len())
            .filter(|&i| !self.entries[i].tiered && self.entries[i].token_count <= frontier)
            .collect();
        by_depth.sort_unstable_by_key(|&i| self.entries[i].token_count);
        let mut on_path: Vec<(usize, u64)> = Vec::with_capacity(by_depth.len());
        let (mut hash, mut pos) = (prefix_hash_seed(adapter_id), 0);
        for i in by_depth {
            let entry = &self.entries[i];
            while pos < entry.token_count {
                hash = prefix_hash_push(hash, tokens[pos]);
                pos += 1;
            }
            if hash == entry.prefix_hash {
                on_path.push((entry.token_count, entry.prefix_hash));
            }
        }
        // One rung per distance bucket [base·2^k, base·2^(k+1)): its
        // SHALLOWEST on-path entry. A rung ages through its bucket and, when
        // it crosses into the next one, refills that bucket if it is empty;
        // the newer entry it leaves behind loses protection only once the
        // bucket holds an older one. Keeping the deepest instead starves
        // every bucket past the first: the near rung moves on each turn and
        // never reaches the far ones.
        let mut rungs = Vec::new();
        let mut last_bucket = u32::MAX;
        for &(depth, hash) in &on_path {
            let distance = frontier - depth;
            if distance < base {
                break; // recent turns: resident by recency
            }
            let bucket = (distance / base).ilog2();
            if bucket != last_bucket {
                rungs.push(hash);
                last_bucket = bucket;
            }
        }
        let key = if session_hash != 0 {
            session_hash
        } else {
            hash_token_prefix(tokens, frontier.min(PATH_KEY_TOKENS), adapter_id)
        };
        self.ladder.set(key, rungs);
    }
}

#[cfg(test)]
#[path = "tests/snapshot_ladder.rs"]
mod tests;
