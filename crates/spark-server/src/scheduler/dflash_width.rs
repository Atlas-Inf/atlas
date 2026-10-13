// SPDX-License-Identifier: AGPL-3.0-only

//! Cost-aware DFlash verify width (`ATLAS_DFLASH_ADAPTIVE_WIDTH=1`).
//!
//! A verify step's cost on GLM grows with the rows it verifies: every row
//! routes to eight experts, and the MoE is bound by reading their weights.
//! Draft positions deep in the block pay off only when the text is
//! predictable (counting, code), so a fixed γ over-verifies prose and a
//! short one under-verifies code. Each sequence keeps decayed per-position
//! conditional acceptance rates; each step takes the width that maximizes
//! expected emitted tokens per millisecond of the measured step cost for
//! that many owners. Owners share one width (the owner-batched verify is
//! uniform), so the batch trades their curves against each other.
//!
//! `ATLAS_DFLASH_FIXED_WIDTH=w` pins the width instead (cost sweeps).
//! `ATLAS_DFLASH_CONF_WIDTH=1` sizes it from the drafter's confidence in
//! this step's drafts where that is measured (`dflash_conf_width`).

use std::sync::OnceLock;

/// Draft positions tracked (γ = 8 → seven drafts).
pub(crate) const MAX_DRAFTS: usize = 7;
/// Estimator tuning: per-step decay (0.98 ≈ the last fifty verifies; code sits near the
/// single-owner break-even, so a noisy estimate there narrows wrongly), the
/// prior conditional acceptance and its weight in pseudo-steps, an
/// optimism bonus `ucb / sqrt(1 + verified)` that re-probes positions a
/// narrow width stopped observing (acceptance is bursty: a code block
/// accepts deep drafts that a prose-trained estimate would never try), and
/// the floor on the conditional acceptance of positions past a full accept.
/// `ATLAS_DFLASH_WIDTH_PARAMS=decay,prior_h,prior_w,ucb[,burst_h]` overrides
/// them (`burst_h` 0 turns the floor off).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Params {
    decay: f32,
    prior_h: f32,
    prior_w: f32,
    ucb: f32,
    burst_h: f32,
}

const DEFAULT_PARAMS: Params = Params {
    decay: 0.98,
    prior_h: 0.85,
    prior_w: 3.0,
    ucb: 0.2,
    burst_h: 0.9,
};

fn parse_params(spec: Option<&str>) -> Params {
    let v: Vec<f32> = spec
        .map(|s| s.split(',').filter_map(|x| x.trim().parse().ok()).collect())
        .unwrap_or_default();
    match v[..] {
        [decay, prior_h, prior_w, ucb] => Params {
            decay,
            prior_h,
            prior_w,
            ucb,
            ..DEFAULT_PARAMS
        },
        [decay, prior_h, prior_w, ucb, burst_h] => Params {
            decay,
            prior_h,
            prior_w,
            ucb,
            burst_h,
        },
        _ => DEFAULT_PARAMS,
    }
}

fn params() -> Params {
    static P: OnceLock<Params> = OnceLock::new();
    *P.get_or_init(|| parse_params(std::env::var("ATLAS_DFLASH_WIDTH_PARAMS").ok().as_deref()))
}

/// Decayed conditional acceptance per draft position.
#[derive(Clone, Debug, Default)]
pub(crate) struct DraftSurvival {
    verified: [f32; MAX_DRAFTS],
    accepted: [f32; MAX_DRAFTS],
    /// Drafts of the last verify when it accepted every one short of the
    /// cap: a burst in progress, so the positions it never tried are hot.
    burst: Option<usize>,
}

impl DraftSurvival {
    /// One verify of `drafted` drafts that accepted the first `accepted`.
    /// Position `j` is observed only when every earlier draft was accepted.
    pub(crate) fn record(&mut self, drafted: usize, accepted: usize) {
        self.burst = (accepted >= drafted && drafted < MAX_DRAFTS).then_some(drafted);
        let decay = params().decay;
        for j in 0..MAX_DRAFTS {
            self.verified[j] *= decay;
            self.accepted[j] *= decay;
            if j < drafted && j <= accepted {
                self.verified[j] += 1.0;
                if j < accepted {
                    self.accepted[j] += 1.0;
                }
            }
        }
    }

    fn hazard(&self, j: usize) -> f32 {
        let p = params();
        let mean = (self.accepted[j] + p.prior_h * p.prior_w) / (self.verified[j] + p.prior_w);
        (mean + p.ucb / (1.0 + self.verified[j]).sqrt()).min(1.0)
    }

    /// Expected tokens emitted by a verify of `width` drafts (bonus included).
    pub(crate) fn expected(&self, width: usize) -> f32 {
        let (mut survive, mut total) = (1.0, 1.0);
        for j in 0..width.min(MAX_DRAFTS) {
            let hot = self.burst.is_some_and(|d| j >= d);
            survive *= if hot {
                self.hazard(j).max(params().burst_h)
            } else {
                self.hazard(j)
            };
            total += survive;
        }
        total
    }
}

/// Measured verify step cost in ms for `owners` sequences of `rows` rows
/// each: median scheduler step interval (verify + re-propose) on GLM-5.3
/// Flash TP2 GB10, prose, 2026-09-28 (`ATLAS_DFLASH_FIXED_WIDTH` sweeps; 5..=8
/// owners on the eight-sequence profile).
fn step_ms(owners: usize, rows: usize) -> f32 {
    // Rows 2..=8, owners 1..=8. A lone owner's 2-row verify takes a slower
    // path than 3 rows. Owners x rows past the 32-row verify budget never
    // run (infinite cost).
    const X: f32 = f32::INFINITY;
    const MS: [[f32; 7]; 8] = [
        [83.8, 73.5, 81.7, 88.7, 97.0, 104.1, 111.2],
        [93.2, 109.4, 123.1, 144.0, 155.3, 165.2, 175.9],
        [114.1, 141.0, 161.2, 176.3, 191.6, 202.7, 216.3],
        [132.1, 163.1, 186.2, 206.4, 222.1, 236.2, 248.4],
        [167.9, 189.5, 218.6, 241.4, 259.3, X, X],
        [184.0, 210.2, 240.4, 263.6, X, X, X],
        [197.8, 229.0, 262.6, X, X, X, X],
        [210.2, 246.1, 279.6, X, X, X, X],
    ];
    MS[owners.clamp(1, 8) - 1][rows.clamp(2, 8) - 2]
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_ADAPTIVE_WIDTH").as_deref() == Ok("1"))
}

/// `ATLAS_DFLASH_WIDTH_LOG=1`: one line per verify (cost sweeps).
pub(crate) fn log_verify(owners: usize, drafts: usize) {
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_WIDTH_LOG").as_deref() == Ok("1")) {
        tracing::info!("DFLASH VERIFY owners={owners} rows={}", drafts + 1);
    }
}

fn fixed() -> Option<usize> {
    static W: OnceLock<Option<usize>> = OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("ATLAS_DFLASH_FIXED_WIDTH")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|w| (1..=MAX_DRAFTS).contains(w))
    })
}

/// Width (drafts per owner, 1..=`max`) for owners verifying together, each
/// holding at least `max` drafts: its acceptance history and the drafter's
/// confidence in its drafts (`ActiveSeq::draft_conf`). `None` leaves the
/// caller's policy in place.
pub(crate) fn choose<'a>(
    owners: impl ExactSizeIterator<Item = (&'a DraftSurvival, &'a [f32])> + Clone,
    max: usize,
) -> Option<usize> {
    let max = max.min(MAX_DRAFTS);
    if max == 0 {
        return None;
    }
    if let Some(w) = fixed() {
        return Some(w.min(max));
    }
    if let Some(w) = super::dflash_conf_width::choose(owners.clone().map(|(_, conf)| conf), max) {
        return Some(w);
    }
    enabled().then(|| best(owners.map(|(survival, _)| survival), max))
}

/// The width maximizing the owners' expected tokens per step millisecond.
fn best<'a>(owners: impl ExactSizeIterator<Item = &'a DraftSurvival> + Clone, max: usize) -> usize {
    let n = owners.len();
    let rate = |w: usize| owners.clone().map(|s| s.expected(w)).sum::<f32>() / step_ms(n, w + 1);
    (1..=max)
        .max_by(|&a, &b| rate(a).total_cmp(&rate(b)))
        .unwrap_or(1)
}

#[cfg(test)]
#[path = "dflash_width_tests.rs"]
mod tests;
