// SPDX-License-Identifier: AGPL-3.0-only

//! The Marconi snapshot pool's size and host-aux budget for a large pool
//! (`ATLAS_QWEN4EXP_SNAPSHOT_SLOTS`, `ATLAS_QWEN4EXP_SNAPSHOT_AUX_MB`; both
//! default off, which leaves the pool and its eviction byte-identical).
//!
//! # Slots
//!
//! `ATLAS_QWEN4EXP_SNAPSHOT_SLOTS=N` sizes the pool at exactly `N` slots,
//! instead of `--ssm-cache-slots` raised to cover `--max-seq-len`
//! (`serve_phases::build::resolve_ssm_cache_slots`, where the ranks also
//! compare the resolved count at startup). A slot is every GDN layer's FP32
//! recurrent state plus its conv window: 56.8 MB a rank on Qwen3.8-Flash-Next
//! TP2 (36 layers x 24 heads x 128 x 128 x 4 B, plus the conv rows). The pool
//! is allocated before the KV budget is measured, so its bytes come out of
//! the KV pool (`factory::build_model`): 256 slots are 14.5 GB a rank, of a
//! KV pool that served a 3.2M-token budget against well under 1M in use.
//!
//! # Host aux
//!
//! Each qwen4_exp snapshot also carries host-side aux (`set_aux`): the QSA
//! indexer keys of every pooled block below its depth (768 B a token on
//! Qwen3.8-Flash-Next, so 25 MB for a 33K-token checkpoint, 113 MB at 147K)
//! and the PLE history and carry. Base never bounds it: a freed slot even
//! keeps its buffers' capacity for the next save. A 16-24-slot pool held at
//! most a few GB; a 256-slot pool of deep checkpoints could hold tens, on a
//! GB10 whose host and GPU share the same memory. So a large pool carries a
//! budget: `ATLAS_QWEN4EXP_SNAPSHOT_AUX_MB`, by default (with the slots
//! switch) `N` x 32K tokens of aux (24 MiB a slot on Qwen3.8-Flash-Next). It
//! is reserved out of the KV budget like the NVMe tiers' host memory, and
//! enforced once a prefill chunk, after its restore: freed slots drop their
//! buffers first, then the pool's own eviction (LRU, or the chain classes of
//! `ATLAS_GLM_PC_EVICT`, which keep frontiers and branch points longest)
//! reclaims cached snapshots until the aux fits.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use spark_runtime::kv_cache::PagedKvCache;

use super::ssm_snapshot::SsmSnapshotPool;
use super::types::TransformerModel;

/// Default aux depth a slot is budgeted for (tokens).
pub const DEFAULT_AUX_TOKENS: usize = 32 * 1024;

/// The aux budget in force (bytes; 0 = unbounded, base).
static BUDGET: AtomicUsize = AtomicUsize::new(0);

/// `ATLAS_QWEN4EXP_SNAPSHOT_SLOTS=N`: the snapshot pool's exact size.
pub fn snapshot_slots_requested() -> Option<usize> {
    static N: std::sync::OnceLock<Option<usize>> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_SNAPSHOT_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n| n > 0)
    })
}

/// `ATLAS_QWEN4EXP_SNAPSHOT_AUX_MB=M` (0 = derived, [`aux_budget_bytes`]).
pub fn snapshot_aux_mb() -> usize {
    static M: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_SNAPSHOT_AUX_MB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// The aux budget for a pool of `slots` and aux of `per_token` bytes a
/// token: `mb` MiB when set, else `slots` x [`DEFAULT_AUX_TOKENS`] tokens
/// when the slots switch is on, else 0 (unbounded).
pub fn aux_budget_bytes(mb: usize, slots_switch: bool, slots: usize, per_token: usize) -> usize {
    if mb > 0 {
        mb << 20
    } else if slots_switch {
        slots * per_token * DEFAULT_AUX_TOKENS
    } else {
        0
    }
}

/// Set the budget (factory, once the pool and the layers exist), and log
/// what the pool takes. Returns the bytes to reserve out of the KV budget.
pub(crate) fn install(pool: &SsmSnapshotPool, per_token: usize) -> usize {
    let slots = pool.num_slots;
    let budget = aux_budget_bytes(
        snapshot_aux_mb(),
        snapshot_slots_requested().is_some(),
        slots,
        per_token,
    );
    BUDGET.store(budget, Ordering::Relaxed);
    if slots > 0 && (budget > 0 || snapshot_slots_requested().is_some()) {
        let slot = pool.num_ssm_layers * (pool.h_bytes + pool.conv_bytes) + pool.hidden_bytes;
        let mib = |b: usize| b as f64 / (1u64 << 20) as f64;
        tracing::info!(
            "SSM snapshot pool (ATLAS_QWEN4EXP_SNAPSHOT_SLOTS): {slots} slots x {:.1} MiB \
             GPU state = {:.2} GiB, out of the KV budget; host aux budget {:.0} MiB \
             ({per_token} B/token), reserved out of the KV budget too",
            mib(slot),
            mib(slots * slot) / 1024.0,
            mib(budget),
        );
    }
    budget
}

/// The budget in force (bytes; 0 = unbounded).
pub(crate) fn budget() -> usize {
    BUDGET.load(Ordering::Relaxed)
}

/// Host bytes the aux map holds (buffer capacity, freed slots' included).
pub(crate) fn held(map: &HashMap<usize, Vec<(u32, Vec<u8>)>>) -> usize {
    map.values()
        .flat_map(|blobs| blobs.iter().map(|(_, b)| b.capacity()))
        .sum()
}

/// `free`'s hook: over budget, a freed slot drops its buffers instead of
/// keeping their capacity for the next save.
pub(crate) fn trim_freed(map: &mut HashMap<usize, Vec<(u32, Vec<u8>)>>, slot: usize) {
    let b = budget();
    if b > 0 && held(map) > b {
        map.remove(&slot);
    }
}

impl TransformerModel {
    /// Bring the snapshot aux under the budget: drop freed slots' buffers,
    /// then reclaim cached snapshots (the pool's eviction order) until it
    /// fits. A no-op without a budget.
    pub(in crate::model) fn enforce_snapshot_aux_budget(&self, kv_cache: &mut PagedKvCache) {
        let b = budget();
        if b == 0 || !self.ssm_snapshots.is_enabled() {
            return;
        }
        let over = || held(&self.ssm_snapshots.aux_blobs.lock()) > b;
        if !over() {
            return;
        }
        self.ssm_snapshots
            .aux_blobs
            .lock()
            .retain(|_, blobs| blobs.iter().any(|(_, x)| !x.is_empty()));
        let mut evicted = 0usize;
        while over() && evicted < self.ssm_snapshots.num_slots {
            let freed = self.ssm_snapshots.reclaim_from_cache(
                self.prefix_cache.as_ref(),
                kv_cache,
                self.ssm_tier_store.as_deref(),
                self.gpu.as_ref(),
            );
            if !freed {
                break;
            }
            evicted += 1;
        }
        if evicted > 0 {
            tracing::info!(
                "snapshot aux over its {} MiB budget: reclaimed {evicted} cached snapshots",
                b >> 20
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_budget_is_explicit_derived_or_off() {
        assert_eq!(aux_budget_bytes(512, true, 256, 768), 512 << 20);
        assert_eq!(aux_budget_bytes(0, true, 256, 768), 256 * 768 * 32 * 1024);
        assert_eq!(aux_budget_bytes(0, false, 24, 768), 0, "base: unbounded");
    }

    #[test]
    fn held_counts_capacity_of_live_and_freed_slots() {
        let mut map = HashMap::new();
        let mut freed = Vec::with_capacity(100);
        freed.extend_from_slice(&[1u8; 10]);
        freed.clear();
        map.insert(1, vec![(3u32, vec![0u8; 50])]);
        map.insert(2, vec![(3u32, freed)]);
        assert!(held(&map) >= 150);
        // No budget: a freed slot keeps its buffers.
        trim_freed(&mut map, 2);
        assert!(map.contains_key(&2));
    }
}
