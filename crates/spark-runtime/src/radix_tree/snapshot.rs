// SPDX-License-Identifier: AGPL-3.0-only

//! SSM snapshot LRU index — independent of the token-radix structure.
//!
//! Snapshots are keyed by (session_hash, token_count, prefix_hash) so the
//! same prompt across requests can hit a cached SSM state without going
//! through the radix tree.

use super::hash_token_prefix;

pub(super) struct SnapshotEntry {
    snapshot_id: usize,
    session_hash: u64,
    token_count: usize,
    prefix_hash: u64,
    last_access: u64,
    /// Cumulative hits over the entry's lifetime — combined with
    /// `last_access` in eviction to approximate the forecast-based
    /// policy from the Marconi paper §4 (B.4, 2026-04-25). Hot
    /// prefixes (high hit count) survive longer than cold ones at
    /// the same age.
    hit_count: u32,
    /// True for the per-session TAIL snapshot (the restore point the next turn's
    /// block-floored `matched_tokens` looks up). Exactly one is kept per session.
    is_tail: bool,
}

pub(super) struct SsmSnapshotIndex {
    pub(super) entries: Vec<SnapshotEntry>,
    pub(super) access_counter: u64,
    /// Session whose `lookup` ran most recently — i.e. the conversation
    /// currently being served this turn. Used by `evict_lru` to protect the
    /// ACTIVE session's restore point while freely reclaiming slots from other
    /// (typically completed) sessions. Set on every session-tagged lookup.
    last_lookup_session: u64,
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
            last_lookup_session: 0,
        }
    }

    pub(super) fn insert(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Option<usize> {
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                let old = entry.snapshot_id;
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return Some(old);
            }
        }
        self.access_counter += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            hit_count: 0,
            is_tail: false,
        });
        None
    }

    /// Insert the per-session TAIL snapshot, superseding this session's previous one.
    ///
    /// Keeping every turn's tail grows the index by one entry per turn, which evicts
    /// the cold 512-grid checkpoints laid down during the trajectory's COLD prefill.
    /// Measured cost of not doing this: fallback replays up to 3712 tokens and a 57 s
    /// TTFT tail. Returns every displaced snapshot_id for the caller to free.
    pub(super) fn insert_tail(
        &mut self,
        prefix_hash: u64,
        snapshot_id: usize,
        session_hash: u64,
        token_count: usize,
    ) -> Vec<usize> {
        let mut displaced = Vec::new();
        if session_hash != 0 {
            let mut i = 0;
            while i < self.entries.len() {
                if self.entries[i].is_tail && self.entries[i].session_hash == session_hash {
                    displaced.push(self.entries.swap_remove(i).snapshot_id);
                } else {
                    i += 1;
                }
            }
        }
        for entry in &mut self.entries {
            if entry.prefix_hash == prefix_hash {
                displaced.push(entry.snapshot_id);
                entry.snapshot_id = snapshot_id;
                entry.session_hash = session_hash;
                entry.token_count = token_count;
                entry.is_tail = true;
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                return displaced;
            }
        }
        self.access_counter += 1;
        self.entries.push(SnapshotEntry {
            snapshot_id,
            session_hash,
            token_count,
            prefix_hash,
            last_access: self.access_counter,
            hit_count: 0,
            is_tail: true,
        });
        displaced
    }

    /// Find deepest snapshot matching session within matched_tokens range.
    pub(super) fn lookup(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
    ) -> Option<(usize, usize)> {
        // Remember the active conversation so a later `evict_lru` this turn
        // protects ITS restore point (not a stale/completed session's).
        if session_hash != 0 {
            self.last_lookup_session = session_hash;
        }
        let mut best: Option<(usize, usize)> = None; // (snapshot_id, token_count)
        for entry in &mut self.entries {
            if entry.token_count > matched_tokens {
                continue;
            }
            if session_hash != 0 && entry.session_hash != 0 && entry.session_hash != session_hash {
                continue;
            }
            let h = hash_token_prefix(tokens, entry.token_count);
            if h != entry.prefix_hash {
                continue;
            }
            if best.is_none() || entry.token_count > best.unwrap().1 {
                self.access_counter += 1;
                entry.last_access = self.access_counter;
                entry.hit_count = entry.hit_count.saturating_add(1);
                best = Some((entry.snapshot_id, entry.token_count));
            }
        }
        if std::env::var("ATLAS_SNAP_LOOKUP_DBG").is_ok() {
            let mut cands: Vec<usize> = self.entries.iter().map(|e| e.token_count).collect();
            cands.sort_unstable();
            tracing::info!(
                "snap-lookup: matched={matched_tokens} selected={:?} n_entries={} token_counts={:?}",
                best.map(|b| b.1),
                self.entries.len(),
                cands,
            );
        }
        best
    }

    pub(super) fn evict_lru(&mut self) -> Option<usize> {
        if self.entries.is_empty() {
            return None;
        }
        // Forecast-based policy (B.4, 2026-04-25, Marconi paper §4): evict the
        // lowest last_access * (1 + hit_count) — old AND cold first — weighting
        // by hit_count so recurrent prefixes survive (#155: the original
        // formula DIVIDED, inverting the intent so frequently-hit snapshots
        // were evicted first, forcing full-conversation recompute).
        //
        // Cross-session fairness (2026-07-03): with SEQUENTIAL agentic
        // trajectories, a COMPLETED trajectory's snapshots (old but with a high
        // hit_count from having been its own frozen restore point) outscore the
        // ACTIVE trajectory's fresh deep checkpoints and refuse to evict —
        // starving the active trajectory until ITS restore point re-freezes at
        // a shallow depth and the recompute tail grows unbounded (measured:
        // session B frozen at 13825 while it grew to 17808+, recompute climbing
        // 79 -> 4447). Fix: protect ONLY the active session's DEEPEST snapshot
        // (its next-turn restore point) and reclaim OTHER sessions' snapshots
        // first (coldest-first), dipping into the active session's own non-tip
        // churn only when no other-session slot remains. This hands the live
        // conversation effectively the whole pool — matching the single-session
        // behavior that tracks the tip (recompute <= one interval). The active
        // session is whichever ran the most recent session-tagged lookup this
        // turn. Retention-only: never perturbs the SSM forward path, so
        // restores stay byte-exact. active==0 (session tracking disabled) falls
        // through to the plain global forecast-LRU above.
        let active = self.last_lookup_session;
        let active_deepest = self
            .entries
            .iter()
            .filter(|e| e.session_hash == active)
            .map(|e| e.token_count)
            .max();
        let score = |e: &SnapshotEntry| e.last_access.saturating_mul(1 + e.hit_count as u64);
        let mut best_other: Option<(usize, u64)> = None; // any non-active-session entry
        let mut best_self: Option<(usize, u64)> = None; // active-session non-tip entry
        for (i, entry) in self.entries.iter().enumerate() {
            let s = score(entry);
            if active == 0 || entry.session_hash != active {
                if best_other.map_or(true, |(_, bs)| s < bs) {
                    best_other = Some((i, s));
                }
            } else if Some(entry.token_count) != active_deepest {
                if best_self.map_or(true, |(_, bs)| s < bs) {
                    best_self = Some((i, s));
                }
            }
        }
        // Reclaim from other sessions first, then the active session's own
        // non-tip churn. If the only entry left is the active session's
        // protected tip, decline rather than evict the live restore point.
        let victim_idx = best_other.or(best_self).map(|(i, _)| i)?;
        let entry = self.entries.swap_remove(victim_idx);
        Some(entry.snapshot_id)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}
