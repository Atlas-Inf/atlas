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
}

pub(super) struct SsmSnapshotIndex {
    pub(super) entries: Vec<SnapshotEntry>,
    pub(super) access_counter: u64,
}

impl SsmSnapshotIndex {
    pub(super) fn new() -> Self {
        Self {
            entries: Vec::new(),
            access_counter: 0,
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
        });
        None
    }

    /// Find deepest snapshot matching session within matched_tokens range.
    pub(super) fn lookup(
        &mut self,
        tokens: &[u32],
        matched_tokens: usize,
        session_hash: u64,
    ) -> Option<(usize, usize)> {
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
        // Forecast-based policy (B.4, 2026-04-25, Marconi paper §4):
        // evict the entry with the lowest last_access * (1 + hit_count)
        // — old AND cold first. Pure LRU (`last_access` only) discarded
        // hot prefixes that just happened to be re-accessed less
        // recently than a one-shot entry; weighting by hit_count keeps
        // recurrent prefixes (system prompts, tool descriptions in
        // agentic sessions) resident longer.
        //
        // #155: the original formula DIVIDED by (1 + hit_count), which
        // inverts the intent — frequently-hit snapshots scored LOWEST
        // and were evicted first at pool saturation (measured: a
        // just-selected snapshot evicted 7s later while ~50
        // never-accessed entries survived → selected=None mid-session
        // → full-conversation SSM recompute on the next warm hit).
        // Tail-pin (2026-07-02): never evict each session's DEEPEST (max
        // token_count) snapshot — that frontier tip is the near-match
        // checkpoint the next warm turn restores from. Without it, the
        // forecast score above entrenches hot SHALLOW prefixes (high
        // hit_count) and starves freshly-saved DEEP checkpoints (hit_count=0)
        // into a 3-of-48-slot rotation, so every deep checkpoint is evicted
        // before the next turn and the restore point ratchets stuck at a
        // fixed shallow token (measured on strix: frozen at 20481 while
        // matched grew to 25376 -> 4895-token SSM recompute tail). Pinning the
        // per-session tip breaks the ratchet; the pin migrates forward as each
        // turn saves a deeper leaf. Retention-only: never touches the SSM
        // forward path, so restores stay byte-exact (tok_agree=1.0).
        let mut deepest: std::collections::HashMap<u64, usize> =
            std::collections::HashMap::new();
        for e in &self.entries {
            let d = deepest.entry(e.session_hash).or_insert(0);
            if e.token_count > *d {
                *d = e.token_count;
            }
        }
        let mut victim: Option<usize> = None;
        let mut victim_score = u64::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            // Skip each session's frontier tip.
            if deepest.get(&entry.session_hash) == Some(&entry.token_count) {
                continue;
            }
            let score = entry.last_access.saturating_mul(1 + entry.hit_count as u64);
            if score < victim_score {
                victim_score = score;
                victim = Some(i);
            }
        }
        // Fallback: every entry is a per-session tip (many single-snapshot
        // sessions saturating the pool). Pinning all would deadlock the pool
        // (save -> reclaim -> None forever, dropping every new checkpoint), so
        // evict the global forecast-LRU among the tips.
        let victim_idx = victim.unwrap_or_else(|| {
            self.entries
                .iter()
                .enumerate()
                .min_by_key(|(_, e)| e.last_access.saturating_mul(1 + e.hit_count as u64))
                .map(|(i, _)| i)
                .unwrap()
        });
        let entry = self.entries.swap_remove(victim_idx);
        Some(entry.snapshot_id)
    }

    pub(super) fn len(&self) -> usize {
        self.entries.len()
    }
}
