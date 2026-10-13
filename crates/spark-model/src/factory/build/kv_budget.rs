// SPDX-License-Identifier: AGPL-3.0-only

//! How many bytes `build_model` may give the paged KV cache.
//!
//! The user-facing contract (vLLM / sparkrun convention) is that
//! `total × --gpu-memory-utilization` is the ceiling on everything THIS
//! process holds. The KV pool gets what is left of that ceiling after the
//! process's own pre-KV footprint and the reserves. Co-tenants (GB10 is
//! shared: ComfyUI, voice, a docker build) are excluded, so a low utilization
//! does not starve the pool for memory Atlas never took.
//!
//! **Own footprint** is the larger of two measures:
//!
//! 1. Free memory at context init minus free memory now
//!    (`gpu::baseline_free_bytes`), or, when
//!    `ATLAS_KV_EXTERNAL_RESERVE_GB` is set, raw used minus that figure.
//!    Exact while co-tenants hold still. When one RELEASES memory during the
//!    load, that memory is credited to Atlas as footprint it never had and
//!    the pool is sized into it: on 2026-09-30 two ranks read 100.1 and
//!    97.9 GiB for a process that holds 102.3, the pool came out 2.1 GiB
//!    large, and the host ran out of unified memory two requests later.
//! 2. What the process itself allocated (`GpuBackend::own_footprint`):
//!    nothing another process does can lower it. This is what fixes the
//!    defect, and only when its device figure is the driver's accounting. The
//!    allocation ledger it falls back to is a lower bound that can miss as
//!    much as a co-tenant releases; that case is a WARN and a larger headroom.
//!
//! A co-tenant that ALLOCATES during the load inflates (1), and a counter
//! that misses something deflates (2); either way the larger one is kept, so
//! the estimate errs toward a smaller pool. A tracked figure above everything
//! in use on the device is an over-count and is charged as everything in use.
//! The two are logged side by side and a gap over [`DISAGREE_WARN_BYTES`] is a
//! WARN.
//!
//! **Headroom floor.** The process keeps growing after sizing (CUDA graphs,
//! JIT, lazy buffers): on the GB10 TP2 reference start by about 2.9 GiB, 2.0
//! of it beyond the 0.9 GiB of reserves. So whatever was measured, the pool
//! leaves the reserves plus [`HEADROOM_BYTES`] of what is free now. The
//! headroom is an absolute size because that growth is; it is capped at the
//! memory the operator left outside the utilization ceiling, so a device
//! Atlas has to itself is sized exactly as before at every utilization.
//!
//! The floor is a backstop, not the fix. It bounds what a mis-measured
//! footprint can cost (free-now is read directly) and keeps co-tenants from
//! filling the margin a growing process needs, but a start it binds settles
//! about 1.5 GiB above empty on the reference pair, where a normal start
//! settles at 2.0. The reference start (4.6 GiB of co-tenants, 3.9 GiB left
//! beyond the reserves) is 0.4 GiB clear of it.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::own_footprint::{DeviceSource, OwnFootprint};

const GIB: usize = 1 << 30;
/// Two own-footprint measures further apart than this are logged at WARN.
pub(super) const DISAGREE_WARN_BYTES: usize = 512 << 20;
/// How far the free-memory measure may exceed the tracked one before that is
/// a WARN. The counters miss memory the process holds outside them (a quiet
/// GB10 start measured 1.59 and 1.65 GiB on the two ranks), so a smaller gap
/// is the normal state, not a co-tenant allocating during the load.
pub(super) const UNTRACKED_WARN_BYTES: usize = 5 * GIB / 2;
/// What the pool leaves free beyond the reserves, for growth after sizing.
pub(super) const HEADROOM_BYTES: usize = 7 * GIB / 2;
/// The same when the tracked figure is only the allocation ledger: then the
/// floor is all that bounds a release during the load, so a start it binds is
/// left what the reference start is left.
pub(super) const UNVERIFIED_HEADROOM_BYTES: usize = 4 * GIB;

pub(super) fn gib(bytes: usize) -> f64 {
    bytes as f64 / GIB as f64
}

/// Everything the arithmetic needs, in bytes, measured by the caller.
#[derive(Clone, Copy, Debug)]
pub(super) struct Inputs {
    pub total: usize,
    pub free_now: usize,
    pub utilization: f64,
    /// Inference reserve plus derived (lazy BF16) reserve.
    pub reserve: usize,
    /// Free memory at GPU-context init; `None` under the mock backend.
    pub baseline_free: Option<usize>,
    /// `ATLAS_KV_EXTERNAL_RESERVE_GB`, when set above zero.
    pub manual_external: Option<usize>,
    /// The process's own allocations since context init, when the backend
    /// keeps that account.
    pub tracked: Option<OwnFootprint>,
}

/// Which free-memory measure produced [`KvBudget::basis_own`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Basis {
    /// Raw used minus `ATLAS_KV_EXTERNAL_RESERVE_GB`.
    Manual,
    /// Baseline-free minus free-now.
    Delta,
    /// No baseline: raw used, co-tenants included.
    Raw,
    /// A baseline that cannot be right (delta zero, or above raw used).
    Implausible,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Disagreement {
    /// Tracked exceeds the free-memory measure by this much: under `Delta`
    /// something else released memory during the load, under `Manual` the
    /// override discounts more than co-tenants hold.
    TrackedAbove(usize),
    /// The delta exceeds tracked by this much: something else allocated
    /// during the load, or the process holds memory the counters miss.
    TrackedBelow(usize),
    /// Tracked exceeds everything in use on the device by this much, and
    /// charging everything in use raised the footprint.
    OverCount(usize),
}

#[derive(Clone, Copy, Debug)]
pub(super) struct KvBudget {
    /// `total − free_now`: everything in use, co-tenants included.
    pub used_raw: usize,
    pub basis: Basis,
    pub basis_own: usize,
    /// The tracked measure as read; it is charged capped at `used_raw`.
    pub footprint: Option<OwnFootprint>,
    /// Pre-KV footprint the budget was charged with.
    pub own: usize,
    pub total_budget: usize,
    pub headroom: usize,
    /// What the utilization ceiling alone allows the pool.
    pub by_budget: usize,
    /// True when the headroom floor, not the utilization ceiling, set `bytes`.
    pub headroom_limited: bool,
    pub bytes: usize,
}

impl KvBudget {
    pub(super) fn disagreement(&self) -> Option<Disagreement> {
        let tracked = self.footprint?.total();
        if tracked > self.used_raw {
            return (self.own > self.basis_own)
                .then_some(Disagreement::OverCount(tracked - self.used_raw));
        }
        if tracked > self.basis_own.saturating_add(DISAGREE_WARN_BYTES) {
            Some(Disagreement::TrackedAbove(tracked - self.basis_own))
        } else if self.basis == Basis::Delta
            && self.basis_own > tracked.saturating_add(UNTRACKED_WARN_BYTES)
        {
            Some(Disagreement::TrackedBelow(self.basis_own - tracked))
        } else {
            None
        }
    }

    /// What other processes hold, by the footprint the budget was charged.
    fn co_tenants(&self) -> usize {
        self.used_raw.saturating_sub(self.own)
    }

    /// How the "No memory left for KV cache" error should end.
    pub(super) fn no_room_advice(&self, free_now: usize) -> String {
        if !self.headroom_limited {
            return "Raise --gpu-memory-utilization or use a smaller model.".into();
        }
        format!(
            "The utilization budget still allows {:.1} GB for KV, but only {:.1} GB \
             is free now and {:.1} GB of that is kept for growth after sizing: other \
             processes hold {:.1} GB. Free memory on the host; raising \
             --gpu-memory-utilization will not help.",
            gib(self.by_budget),
            gib(free_now),
            gib(self.headroom),
            gib(self.co_tenants()),
        )
    }
}

pub(super) fn size(i: &Inputs) -> KvBudget {
    let used_raw = i.total.saturating_sub(i.free_now);
    let (basis, basis_own) = match (i.manual_external, i.baseline_free) {
        (Some(external), _) => (Basis::Manual, used_raw.saturating_sub(external)),
        (None, Some(baseline)) => match baseline.saturating_sub(i.free_now) {
            delta if delta > 0 && delta <= used_raw => (Basis::Delta, delta),
            _ => (Basis::Implausible, used_raw),
        },
        (None, None) => (Basis::Raw, used_raw),
    };
    // A process cannot hold more than is in use system-wide, so a tracked
    // figure above that is charged as everything in use: the conservative
    // reading of a counter that over-counts, as for an implausible delta.
    let tracked = i.tracked.map_or(0, |t| t.total().min(used_raw));
    let own = basis_own.max(tracked);
    let total_budget = (i.total as f64 * i.utilization) as usize;
    let headroom = match i.tracked {
        Some(t) if t.device_source == DeviceSource::Ledger => UNVERIFIED_HEADROOM_BYTES,
        _ => HEADROOM_BYTES,
    }
    .min(i.total.saturating_sub(total_budget));
    let by_budget = total_budget.saturating_sub(own).saturating_sub(i.reserve);
    let by_free = i
        .free_now
        .saturating_sub(i.reserve)
        .saturating_sub(headroom);
    KvBudget {
        used_raw,
        basis,
        basis_own,
        footprint: i.tracked,
        own,
        total_budget,
        headroom,
        by_budget,
        headroom_limited: by_free < by_budget,
        bytes: by_budget.min(by_free),
    }
}

/// One line for the log.
#[derive(Debug)]
pub(super) struct Note {
    pub warn: bool,
    pub text: String,
}

/// What to say about a sizing. `pool_from_budget` is false when something
/// else sizes the pool (`--high-speed-swap`) and the budget is not used.
pub(super) fn notes(i: &Inputs, b: &KvBudget, pool_from_budget: bool) -> Vec<Note> {
    let mut out = Vec::new();
    let mut say = |warn: bool, text: String| out.push(Note { warn, text });
    match (b.basis, i.manual_external, i.baseline_free) {
        (Basis::Manual, Some(external), _) => say(
            false,
            format!(
                "ATLAS_KV_EXTERNAL_RESERVE_GB={:.2} (manual override): discounting \
                 external/co-tenant memory from KV budget — used_so_far {:.1} GB → \
                 Atlas-own {:.1} GB",
                gib(external),
                gib(b.used_raw),
                gib(b.basis_own),
            ),
        ),
        (Basis::Delta, _, Some(baseline)) => say(
            false,
            format!(
                "KV budget self-relative (auto): baseline-free {:.1} GB − free-now \
                 {:.1} GB = Atlas-own {:.1} GB; co-tenants {:.1} GB excluded \
                 (set ATLAS_KV_EXTERNAL_RESERVE_GB to override)",
                gib(baseline),
                gib(i.free_now),
                gib(b.basis_own),
                gib(b.used_raw - b.basis_own),
            ),
        ),
        (Basis::Implausible, _, Some(baseline)) => say(
            true,
            format!(
                "KV budget auto-measure implausible (baseline {:.1} GB, free-now \
                 {:.1} GB, used {:.1} GB) — using raw used_so_far",
                gib(baseline),
                gib(i.free_now),
                gib(b.used_raw),
            ),
        ),
        _ => {}
    }
    if let Some(f) = b.footprint {
        let tracked = f.total();
        say(
            false,
            format!(
                "KV own-footprint cross-check: free-memory measure {:.2} GB, tracked \
                 {:.2} GB (device {:.2} GB by {} + host {:.2} GB) → {:.2} GB pre-KV",
                gib(b.basis_own),
                gib(tracked),
                gib(f.device),
                f.device_source.label(),
                gib(f.host),
                gib(b.own),
            ),
        );
        let by_ledger = f.device_source == DeviceSource::Ledger;
        if by_ledger {
            say(
                true,
                format!(
                    "KV own-footprint: the driver's per-process accounting is not \
                     available (libnvidia-ml.so.1 is missing from this container, or \
                     does not list this process), so the tracked figure is the \
                     allocation ledger — a lower bound that misses library workspaces \
                     and driver rounding. Memory another process releases during the \
                     load can still be sized into the KV pool; only the {:.1} GB kept \
                     free beyond the reserves bounds that",
                    gib(b.headroom),
                ),
            );
        }
        match b.disagreement() {
            Some(Disagreement::OverCount(gap)) => say(
                true,
                format!(
                    "KV own-footprint: tracked {:.2} GB is {:.2} GB more than the \
                     {:.2} GB in use on the whole device — its counters over-count \
                     here; charging everything in use to Atlas, co-tenants included \
                     (the KV pool is smaller than it needs to be)",
                    gib(tracked),
                    gib(gap),
                    gib(b.used_raw),
                ),
            ),
            Some(Disagreement::TrackedAbove(gap)) if b.basis == Basis::Manual => say(
                true,
                format!(
                    "KV own-footprint: ATLAS_KV_EXTERNAL_RESERVE_GB leaves {:.2} GB as \
                     Atlas-own, but Atlas allocated {:.2} GB — the override discounts \
                     {:.2} GB more than other processes hold; sizing from the tracked \
                     figure",
                    gib(b.basis_own),
                    gib(tracked),
                    gib(gap),
                ),
            ),
            Some(Disagreement::TrackedAbove(gap)) => say(
                true,
                format!(
                    "KV own-footprint: Atlas allocated {:.2} GB but free memory fell by \
                     only {:.2} GB — {:.2} GB was released by something else during the \
                     load and would have been sized into the KV pool; sizing from the \
                     tracked figure",
                    gib(tracked),
                    gib(b.basis_own),
                    gib(gap),
                ),
            ),
            // Under the ledger the delta is expected to be the larger one; the
            // WARN above already says what that means.
            Some(Disagreement::TrackedBelow(gap)) if !by_ledger => say(
                true,
                format!(
                    "KV own-footprint: free memory fell by {:.2} GB but Atlas's own \
                     counters account for {:.2} GB — {:.2} GB was taken by something \
                     else during the load, or is Atlas memory the counters miss; \
                     keeping the larger figure (the KV pool may be smaller than it \
                     needs to be)",
                    gib(b.basis_own),
                    gib(tracked),
                    gib(gap),
                ),
            ),
            _ => {}
        }
    }
    if b.headroom_limited && pool_from_budget {
        say(
            true,
            format!(
                "KV pool limited by free memory: {:.2} GB free now; keeping {:.2} GB \
                 for growth after sizing plus the {:.2} GB reserve unallocated → \
                 {:.2} GB for KV, not the {:.2} GB the utilization budget allows. \
                 Other processes hold {:.2} GB.",
                gib(i.free_now),
                gib(b.headroom),
                gib(i.reserve),
                gib(b.bytes),
                gib(b.by_budget),
                gib(b.co_tenants()),
            ),
        );
    }
    out
}

/// `ATLAS_KV_EXTERNAL_RESERVE_GB` as bytes; unset, unparsable and values
/// that are not above zero all mean "no override".
pub(super) fn external_reserve_bytes(value: Option<&str>) -> Option<usize> {
    let gb = value?.parse::<f64>().ok().filter(|&gb| gb > 0.0)?;
    Some((gb * GIB as f64) as usize)
}

/// Take the measurements, size the budget, and say what was decided.
pub(super) fn measure(
    gpu: &dyn GpuBackend,
    total: usize,
    free_now: usize,
    utilization: f64,
    reserve: usize,
    pool_from_budget: bool,
) -> KvBudget {
    let inputs = Inputs {
        total,
        free_now,
        utilization,
        reserve,
        baseline_free: spark_runtime::gpu::baseline_free_bytes(),
        manual_external: external_reserve_bytes(
            std::env::var("ATLAS_KV_EXTERNAL_RESERVE_GB")
                .ok()
                .as_deref(),
        ),
        tracked: gpu.own_footprint(),
    };
    let budget = size(&inputs);
    for note in notes(&inputs, &budget, pool_from_budget) {
        if note.warn {
            tracing::warn!("{}", note.text);
        } else {
            tracing::info!("{}", note.text);
        }
    }
    budget
}

/// Blocks `budget` buys when each pool token also carries `aux_per_token`
/// bytes of per-sequence state kept outside the pool
/// (`TransformerLayer::aux_bytes_per_token`, the QSA indexer carry). With
/// no such state this is `PagedKvCache::compute_num_blocks`.
pub(super) fn blocks_with_aux(
    kv: &spark_runtime::kv_cache::KvCacheConfig,
    budget: usize,
    aux_per_token: usize,
) -> anyhow::Result<usize> {
    let per_block = kv.block_bytes_kv_all_layers();
    anyhow::ensure!(per_block > 0, "KV cache block size is zero");
    Ok(budget / (per_block + aux_per_token * kv.block_size))
}

#[cfg(test)]
#[path = "kv_budget_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "kv_budget_notes_tests.rs"]
mod notes_tests;
