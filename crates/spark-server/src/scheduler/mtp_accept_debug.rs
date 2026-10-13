// SPDX-License-Identifier: AGPL-3.0-only

//! Per-batch-width MTP acceptance telemetry (`ATLAS_MTP_ACCEPT_DEBUG`).
//!
//! # Why this exists
//!
//! The C=8 bar is arithmetic in ONE quantity: expected tokens per verify step
//! divided by the verify step's cost relative to a plain decode step. The
//! numerator is `1 + p1 + p1*p2c + ...`, i.e. `1 + mean_accepted`. Before this
//! module nothing reported it at the shipped operating point:
//! `k4_record_positional` (the only p1 source) is gated on `k_drafts == 3`,
//! and the default ladder runs `k_drafts == 2` for n in [5, 8]. The na
//! histogram (`k4_record_outcome`) is width-blind — it mixes every n in one
//! set of counters, so an accept A/B at C=8 could not be attributed.
//!
//! # What it reports
//!
//! One line per `PERIOD` recorded verifies PER BATCH WIDTH:
//! `p1` (fraction of steps whose FIRST draft matched the target — measured
//! before the accept chain short-circuits, so it is unconditional),
//! `mean_na` (mean accepted drafts) and `tok_step = 1 + mean_na`.
//!
//! Counters are relaxed atomics and the log fires off one thread at a time;
//! there is no D2H and no stream sync, so the only cost in a timed leg is the
//! periodic `tracing::info!`. Still gated: presence of `ATLAS_MTP_ACCEPT_DEBUG`.

use std::sync::atomic::{AtomicU64, Ordering};

use spark_model::speculative::deep_depth;

use super::mtp_deep_depth::DepthLevers;

/// Widths tracked individually; anything wider folds into the last bucket.
///
/// MUST cover the MTP dispatch cap ([`spark_model::speculative::mtp_max_seqs`],
/// default 32) or the fold ALIASES distinct widths onto one bucket. It did:
/// this was 17 while the cap was raised 16 -> 32 (`69658b873`, 2026-07-30),
/// so every width in 16..=128 accumulated into bucket 16 and the flush
/// reported whichever width happened to trip [`PERIOD`]. The `diag_c32`
/// serve log shows the single bucket flushing as
/// `n=18/25/26/28/29/30/31/32` inside one run, at a mean p1 of 0.676 —
/// while a clean n=16 bucket reads 0.77-0.89.
///
/// Two consequences, both live and both on the rung controller:
/// * the n=16 sample [`super::adaptive_rung`] steers the shipped rung from
///   was a MIXTURE of n=16 and n>16 statistics. n>16 accepts materially
///   worse, so the mixture biases `p1` (and through it the token ratio)
///   DOWN, i.e. toward `k=1` — the conservative side;
/// * a flush tripped by an n>16 caller is handed to `observe(n>16, ..)`,
///   which the `9..=16` BAND discards outright, so under mixed-width
///   traffic the n=16 controller silently LOSES ticks it paid for.
///
/// Kill switch `ATLAS_MTP_ACCEPT_FOLD_AT_16` (presence — house convention,
/// `=0` is NOT off) restores the pre-fix fold for an A/B.
const MAX_N: usize = 33;

/// PRESENCE check for `ATLAS_MTP_ACCEPT_FOLD_AT_16`: restores the pre-fix
/// bucket fold (every width >= 16 into one bucket) so the correction is
/// A/B-able against the binary that measured the ladder. Read once per
/// process.
fn fold_at_16() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_MTP_ACCEPT_FOLD_AT_16").is_some())
}

/// The bucket index for batch width `n` — SSOT for [`record`] and for the
/// aliasing test. Widths above the tracked range fold onto the last bucket.
fn bucket_idx(n: usize) -> usize {
    if fold_at_16() {
        n.min(16)
    } else {
        n.min(MAX_N - 1)
    }
}
/// Verifies per flush. A flush is both a log line and one controller tick
/// for [`super::adaptive_rung`], so it sets the rung's reaction time: 128
/// verifies = 8 steps at n=16 ~ 1 s, which puts the probe interval (8 ticks)
/// inside a single short agentic burst. Was 200 before wave 28, when the
/// flush had no consumer but the log.
const PERIOD: u64 = 128;

/// A flush that took longer than this to accumulate its `PERIOD` verifies is
/// NOT a sample of current traffic and is dropped rather than fed to the rung
/// controller. Measured 2026-08-01: the width the batch actually sits at
/// fills a bucket in ~1.1 s, while a width the batch only grazes (n=15 at
/// C=16) took over TWO MINUTES — long enough to span a whole change of
/// workload. That stale bucket flushed prose-era statistics in the middle of
/// a tool-shaped leg and flipped the rung back to `k=1`, which is exactly the
/// failure this guard removes. Only the flush timestamp is read, so the cost
/// is one clock read per `PERIOD` verifies, not per verify.
const MAX_SAMPLE_SPAN_MS: u64 = 5_000;

struct Bucket {
    steps: AtomicU64,
    d1: AtomicU64,
    na: AtomicU64,
    k: AtomicU64,
    /// Millis since [`EPOCH`] when this bucket's CURRENT accumulation window
    /// opened (stamped when `steps` goes 0 -> 1), so the span below is the
    /// exact window the sample covers — correct for the first flush too.
    window_start_ms: AtomicU64,
}

const fn new_bucket() -> Bucket {
    Bucket {
        steps: AtomicU64::new(0),
        d1: AtomicU64::new(0),
        na: AtomicU64::new(0),
        k: AtomicU64::new(0),
        window_start_ms: AtomicU64::new(0),
    }
}

/// Process-start reference for the flush clock (monotonic, unaffected by
/// wall-clock steps).
static EPOCH: std::sync::LazyLock<std::time::Instant> =
    std::sync::LazyLock::new(std::time::Instant::now);

#[allow(clippy::declare_interior_mutable_const)]
const INIT: Bucket = new_bucket();
static BUCKETS: [Bucket; MAX_N] = [INIT; MAX_N];

/// Record one sequence's verify outcome at batch width `n`.
///
/// `d1_match` must be the UNCONDITIONAL first-position draft match
/// (`drafts[0] == verified[0]`), not `num_accepted >= 1` — they agree today
/// but the second form silently becomes conditional if a future verdict path
/// short-circuits before comparing.
///
/// `k_drafts` is the RETAINED depth of THIS sequence, which D-Cut makes ragged
/// within one batch; the reported value is the period's DEEPEST retained depth
/// at this width (identical to the uniform value when D-Cut is off, and never
/// an arbitrary last-writer). The full per-step shape is on the `MTP D-Cut`
/// line — `p1` stays unconditional and `mean_na`/`tok_step` are measured over
/// the shape that actually ran, which is the quantity the C=8 arithmetic wants.
/// ★ Accumulation is UNCONDITIONAL as of wave 28 — `ATLAS_MTP_ACCEPT_DEBUG`
/// now gates only the log line. The counters are the SSOT for accept
/// statistics and [`super::adaptive_rung`] steers the n=16 rung from them, so
/// gating the accounting would make the shipped rung depend on whether
/// telemetry happened to be switched on. The counters are relaxed atomics
/// with no D2H and no stream sync, so always-on costs nothing measurable.
pub(super) fn record(n: usize, k_drafts: usize, d1_match: bool, num_accepted: usize) {
    let b = &BUCKETS[bucket_idx(n)];
    b.k.fetch_max(k_drafts as u64, Ordering::Relaxed);
    b.na.fetch_add(num_accepted as u64, Ordering::Relaxed);
    if d1_match {
        b.d1.fetch_add(1, Ordering::Relaxed);
    }
    let prev_steps = b.steps.fetch_add(1, Ordering::Relaxed);
    if prev_steps == 0 {
        b.window_start_ms
            .store(EPOCH.elapsed().as_millis() as u64, Ordering::Relaxed);
    }
    if prev_steps + 1 >= PERIOD {
        let steps = b.steps.swap(0, Ordering::Relaxed).max(1);
        let d1 = b.d1.swap(0, Ordering::Relaxed);
        let na = b.na.swap(0, Ordering::Relaxed);
        let k = b.k.swap(0, Ordering::Relaxed);
        let mean_na = na as f64 / steps as f64;
        let p1 = d1 as f64 / steps as f64;
        let span_ms = (EPOCH.elapsed().as_millis() as u64)
            .saturating_sub(b.window_start_ms.load(Ordering::Relaxed));
        let fresh = span_ms <= MAX_SAMPLE_SPAN_MS;
        // ONE accounting path: the rung controller consumes this flush
        // rather than maintaining its own counters (SSOT). Stale flushes are
        // logged but NOT steered on — see MAX_SAMPLE_SPAN_MS.
        if fresh {
            super::adaptive_rung::observe(n, k as usize, p1, mean_na);
        }
        if spark_model::speculative::mtp_accept_debug() {
            tracing::info!(
                "MTP accept n={n} k_drafts={k} verifies={steps} p1={p1:.3} \
                 mean_na={mean_na:.3} tok_step={:.3} token_ratio={:.4} \
                 span_ms={span_ms} fresh={fresh}",
                1.0 + mean_na,
                super::adaptive_rung::token_ratio(
                    p1,
                    super::adaptive_rung::p2_cond_from(p1, mean_na).unwrap_or(0.0),
                ),
            );
        }
    }
}

/// Per-request view of the same quantities [`record`] accumulates
/// (`p1`, `mean_na`, `tok_step = 1 + mean_na`), plus serial-vs-MTP step
/// fraction and depth-regime re-probe count for the Done-line.
/// Not a new telemetry product — the request-finished line is the sink.
#[derive(Debug, Default, Clone, Copy)]
pub struct RequestAccept {
    serial_steps: u64,
    mtp_steps: u64,
    d1: u64,
    na: u64,
    /// Verifies that accepted at least `i + 1` drafts (the per-position
    /// survival curve that sizes the draft cap).
    survive: [u64; SURVIVAL_POSITIONS],
    pub regime_reprobes: u64,
    // Per-request K=3/K=5 controller.  Zero is deliberately the deep state so
    // `Default` starts every request with an informative K=5 probe.
    depth_mode: u8,
    last_verify_drafts: usize,
    depth_steps: u16,
    depth_full_accepts: u32,
    depth_k3_accepts: u32,
    shallow_steps_since_probe: u16,
    depth_switches: u16,
    // Per-request 1..=3-draft controller (`mtp_depth_ladder`), steering a
    // ceiling of 2 or 3 drafts (qwen4_exp K=2..4).
    ladder: super::mtp_depth_ladder::DepthLadder,
    // `ATLAS_MTP_DYNAMIC_DEPTH`: the deep ceiling (`mtp_deep_depth`), and the
    // per-position acceptance over the verifies that drafted each position.
    deep: super::mtp_deep_depth::DeepDepth,
    pos: super::mtp_deep_depth::PositionAccept,
}

const SURVIVAL_POSITIONS: usize = 7;
const DEPTH_DEEP: u8 = 0;
const DEPTH_SHALLOW: u8 = 1;
const DEPTH_WINDOW: u16 = 12;
const DEPTH_REPROBE_STEPS: u16 = 128;

impl RequestAccept {
    pub fn record_serial(&mut self) {
        self.serial_steps = self.serial_steps.saturating_add(1);
        self.ladder.note_serial();
    }

    /// `emitted` is tokens committed this verify (1 + accepted drafts).
    /// `d1_match` agrees with `emitted > 1` today (see [`record`]).
    pub fn record_verify_emitted(&mut self, emitted: usize) {
        self.mtp_steps = self.mtp_steps.saturating_add(1);
        let accepted = emitted.saturating_sub(1) as u64;
        self.na = self.na.saturating_add(accepted);
        if accepted > 0 {
            self.d1 = self.d1.saturating_add(1);
        }
        for s in self.survive.iter_mut().take(accepted as usize) {
            *s = s.saturating_add(1);
        }
    }

    /// Desired proposal depth for a single request.  This controller only
    /// selects the two measured efficient kernels: K=3 (two drafts) and K=5
    /// (four drafts).  Intermediate K=4 was slower than both on GB10.
    ///
    /// Under a 2..=3-draft ceiling the same lever arms the 1..=3 ladder
    /// (`mtp_depth_ladder`: measured per-position survival over measured
    /// step wall).
    ///
    /// `ATLAS_MTP_DYNAMIC_DEPTH` (any ceiling past 3) supersedes both: the
    /// deep controller (`mtp_deep_depth`) steers between its floor and
    /// `max_drafts`.
    pub fn depth_drafts(&self, max_drafts: usize, levers: DepthLevers) -> usize {
        let enabled = levers.adapt;
        if levers.deep && max_drafts > super::mtp_depth_ladder::MAX_DEPTH {
            self.deep
                .drafts(deep_depth::deep_floor(max_drafts), max_drafts)
        } else if enabled && max_drafts >= 4 && self.depth_mode == DEPTH_SHALLOW {
            2
        } else if enabled && (2..=super::mtp_depth_ladder::MAX_DEPTH).contains(&max_drafts) {
            self.ladder.drafts(max_drafts)
        } else {
            max_drafts
        }
    }

    /// Feed a true verify outcome to the per-request depth controller.
    ///
    /// A K=5 verify tells us both yields at identical context: the full
    /// accepted prefix and `min(accepted, 2)`, the tokens K=3 would have kept.
    /// We retain K=5 only when its token-yield ratio pays for its measured
    /// ~20% higher target-forward cost.  K=3 periodically probes K=5 again,
    /// and promotes early when its own acceptance becomes high.
    /// `max_drafts` is the ceiling the depth was chosen under (the deep
    /// controller's `--num-drafts`).
    pub fn record_depth_verify(
        &mut self,
        drafts: usize,
        accepted: usize,
        max_drafts: usize,
        levers: DepthLevers,
    ) {
        let enabled = levers.adapt;
        self.last_verify_drafts = drafts;
        self.pos.record(drafts, accepted);
        if levers.deep && max_drafts > super::mtp_depth_ladder::MAX_DEPTH {
            let floor = deep_depth::deep_floor(max_drafts);
            self.deep.record(drafts, accepted, floor, max_drafts);
            return;
        }
        if !enabled {
            return;
        }
        self.record_depth_verify_inner(drafts, accepted);
    }

    fn record_depth_verify_inner(&mut self, drafts: usize, accepted: usize) {
        if drafts <= super::mtp_depth_ladder::MAX_DEPTH {
            self.ladder.record(
                drafts,
                accepted,
                super::mtp_depth_ladder::MAX_DEPTH,
                std::time::Instant::now(),
            );
        }
        if drafts >= 4 {
            self.depth_steps = self.depth_steps.saturating_add(1);
            self.depth_full_accepts = self
                .depth_full_accepts
                .saturating_add(accepted.min(drafts) as u32);
            self.depth_k3_accepts = self.depth_k3_accepts.saturating_add(accepted.min(2) as u32);

            if self.depth_steps >= DEPTH_WINDOW {
                let steps = self.depth_steps as f64;
                let full_yield = 1.0 + self.depth_full_accepts as f64 / steps;
                let k3_yield = 1.0 + self.depth_k3_accepts as f64 / steps;
                let yield_ratio = full_yield / k3_yield;
                // Measured K5/K3 step-cost ratio is 1.20 on dual GB10.  A
                // small margin avoids oscillation on a boundary workload.
                if yield_ratio < 1.18 {
                    if self.depth_mode != DEPTH_SHALLOW {
                        self.depth_switches = self.depth_switches.saturating_add(1);
                        tracing::info!(
                            "MTP depth adapt: K5 -> K3 (yield_ratio={yield_ratio:.3}, window={})",
                            self.depth_steps
                        );
                    }
                    self.depth_mode = DEPTH_SHALLOW;
                    self.shallow_steps_since_probe = 0;
                } else {
                    self.depth_mode = DEPTH_DEEP;
                }
                self.reset_depth_window();
            }
        } else if drafts == 2 && self.depth_mode == DEPTH_SHALLOW {
            self.depth_steps = self.depth_steps.saturating_add(1);
            self.depth_k3_accepts = self.depth_k3_accepts.saturating_add(accepted.min(2) as u32);
            self.shallow_steps_since_probe = self.shallow_steps_since_probe.saturating_add(1);

            if self.depth_steps >= DEPTH_WINDOW {
                let mean = self.depth_k3_accepts as f64 / self.depth_steps as f64;
                // Near-saturated K3 cannot observe positions 3/4, so promote
                // to a K5 measurement window.  Low-acceptance prose measured
                // ~1.0 here; high-acceptance requests measured ~1.75.
                if mean >= 1.70 || self.shallow_steps_since_probe >= DEPTH_REPROBE_STEPS {
                    self.depth_mode = DEPTH_DEEP;
                    self.depth_switches = self.depth_switches.saturating_add(1);
                    tracing::info!(
                        "MTP depth adapt: K3 -> K5 probe (mean_k3={mean:.3}, since_probe={})",
                        self.shallow_steps_since_probe
                    );
                    self.shallow_steps_since_probe = 0;
                }
                self.reset_depth_window();
            }
        }
    }

    fn reset_depth_window(&mut self) {
        self.depth_steps = 0;
        self.depth_full_accepts = 0;
        self.depth_k3_accepts = 0;
    }

    /// Total draft tokens ACCEPTED for this request — the per-request quantity
    /// `usage.completion_tokens_details.accepted_prediction_tokens` reports.
    /// Raw count, not a rate: the wire field's meaning is "predicted tokens
    /// that matched generation", and `na` is exactly that sum.
    pub fn accepted_total(&self) -> u64 {
        self.na
    }

    pub fn note_regime_reprobe(&mut self) {
        self.regime_reprobes = self.regime_reprobes.saturating_add(1);
    }

    fn total_steps(&self) -> u64 {
        self.serial_steps.saturating_add(self.mtp_steps)
    }

    pub fn serial_frac(&self) -> f64 {
        let n = self.total_steps();
        if n == 0 {
            0.0
        } else {
            self.serial_steps as f64 / n as f64
        }
    }

    pub fn mtp_frac(&self) -> f64 {
        let n = self.total_steps();
        if n == 0 {
            0.0
        } else {
            self.mtp_steps as f64 / n as f64
        }
    }

    pub fn p1(&self) -> f64 {
        if self.mtp_steps == 0 {
            0.0
        } else {
            self.d1 as f64 / self.mtp_steps as f64
        }
    }

    pub fn mean_na(&self) -> f64 {
        if self.mtp_steps == 0 {
            0.0
        } else {
            self.na as f64 / self.mtp_steps as f64
        }
    }

    pub fn tok_step(&self) -> f64 {
        1.0 + self.mean_na()
    }

    pub fn done_suffix(&self) -> String {
        let steps = self.mtp_steps.max(1) as f64;
        let survival: Vec<String> = self
            .survive
            .iter()
            .map(|&s| format!("{:.2}", s as f64 / steps))
            .collect();
        format!(
            "serial={:.2} mtp={:.2} p1={:.3} mean_na={:.3} tok_step={:.3} regime_reprobes={} depth={} depth_switches={} acc={} surv={}",
            self.serial_frac(),
            self.mtp_frac(),
            self.p1(),
            self.mean_na(),
            self.tok_step(),
            self.regime_reprobes,
            match self.last_verify_drafts {
                1 => "k2",
                2 => "k3",
                3 => "k4",
                4 => "k5",
                5 => "k6",
                6 => "k7",
                7 => "k8",
                _ if self.depth_mode == DEPTH_SHALLOW => "k3",
                _ => "k5",
            },
            self.depth_switches
                .saturating_add(self.ladder.switches)
                .saturating_add(self.deep.switches),
            self.pos.suffix(),
            survival.join(","),
        )
    }

    pub fn log_done(n: usize, reason: &str, tps: f64, ttft_ms: f64, acct: &Self) {
        tracing::info!(
            "Done: {n} tokens ({reason}) {tps:.1} tok/s, TTFT={ttft_ms:.1}ms, {}",
            acct.done_suffix()
        );
    }
}

#[cfg(test)]
#[path = "mtp_accept_debug_tests.rs"]
mod tests;
