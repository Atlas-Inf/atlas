// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash verify width from the drafter's own confidence
//! (`ATLAS_DFLASH_CONF_WIDTH=1`).
//!
//! `dflash_width` sizes a verify from acceptance history, so it verifies the
//! same rows whether or not the drafter was sure of this block: prose pays
//! for rows that are rejected, and a sure run inside prose is cut short. The
//! DFlash2 selector reports, per draft, the log-probability of its pick among
//! the candidates it scored; it rides with the drafts in
//! `ActiveSeq::pending_draft_conf` and is cut with them. A calibration table
//! learned while serving turns that into the chance the target accepts the
//! draft once it accepted the ones before it; the running
//! product is the draft's survival. The step verifies the width that
//! maximizes the owners' expected tokens less the price of its rows, a row
//! being worth `ATLAS_DFLASH_CONF_TAU` tokens (default 0.3): for one owner
//! that cuts the draft at the first position whose survival is below it.
//! Rows are priced with today's step cost ([`step_ms`]); the throughput-
//! optimal price per row is the tokens per millisecond the step achieves
//! times the row's cost, 0.21..0.38 tokens across prose, code and four
//! streams, over which the gain simulated from logged acceptance is flat
//! within half a point, so one fixed price serves them all.
//!
//! Verification is exact at any width, so only the rows per step change.
//! The drafter has already run; the cut rows are simply never launched.
//!
//! The table cannot drift: every cell decays each verify back towards its
//! confidence bin's rate over all depths (the first draft is verified at any
//! confidence, so every bin stays observed), and every
//! `ATLAS_DFLASH_CONF_PROBE`-th step (default 64, 0 = never) verifies the
//! full width, which observes the positions the rule had cut.
//!
//! `ATLAS_DFLASH_CONF_LOG=1` logs each verify's confidences beside its
//! outcome (with or without the width rule) for offline fitting.
//!
//! Prior art: the design follows knapcio's draft-shape truncation
//! `GLM_DRAFT_TRUNC` (knapcio, <https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4>,
//! `overlay/glm_draft_trunc.py` @ 982e258): the selector's log max softmax
//! per draft, a table of acceptance per draft position and confidence bin,
//! expected tokens as the sum of running survival products, and one uniform
//! width maximizing expected tokens less a per-row price. Ideas, no code.
//! Ours: the online decayed calibration pooled over depths, the fixed `TAU`
//! price, and the periodic full-width probe. `EDGES` and `PRIOR` come from
//! that repository's measured table (MIT): see their doc comments and
//! docs/glm-prior-art.md.

use std::sync::{Mutex, OnceLock};

use super::dflash_width::MAX_DRAFTS;

/// Confidence bin edges (log-probability of the pick); bin `b` holds
/// confidences in `[EDGES[b - 1], EDGES[b])`. These are the upper ten edges
/// (`edges[7..]`) of knapcio's `overlay/glm_bav_table_seg.json`
/// (<https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4> @ d80f4fd;
/// Copyright (c) 2026 knapcio, MIT; notice in docs/glm-prior-art.md).
const EDGES: [f32; 10] = [
    -1.5, -1.0, -0.7, -0.5, -0.35, -0.22, -0.12, -0.06, -0.03, -0.01,
];
const BINS: usize = EDGES.len() + 1;
/// Acceptance per bin before any verify: a drafter's top-1 probability
/// overstates acceptance (knapcio's calibration of this drafter, the same
/// table as `EDGES`, reads 0.96 only above p = 0.99), so the prior is a
/// rounded reading of that table's first-position row (`g[0][7..]`). Serving
/// replaces it within a few hundred verifies.
const PRIOR: [f32; BINS] = [
    0.12, 0.16, 0.25, 0.35, 0.42, 0.48, 0.55, 0.62, 0.68, 0.76, 0.95,
];
/// Pseudo-observations behind the prior in a bin's all-depth rate, and
/// behind that rate in one depth's cell.
const PRIOR_WEIGHT: f32 = 8.0;
const POOL_WEIGHT: f32 = 8.0;
/// Per-verify decay of every cell (about the last two hundred verifies).
const DECAY: f32 = 0.995;

fn bin(conf: f32) -> usize {
    EDGES.iter().filter(|&&edge| conf >= edge).count()
}

/// Decayed acceptance counts per draft position and confidence bin.
#[derive(Clone, Debug)]
pub(crate) struct Calibration {
    verified: [[f32; BINS]; MAX_DRAFTS],
    accepted: [[f32; BINS]; MAX_DRAFTS],
}

impl Default for Calibration {
    fn default() -> Self {
        Self {
            verified: [[0.0; BINS]; MAX_DRAFTS],
            accepted: [[0.0; BINS]; MAX_DRAFTS],
        }
    }
}

impl Calibration {
    /// Chance the target accepts draft `depth` at confidence bin `b`, given
    /// it accepted every draft before it.
    fn rate(&self, depth: usize, b: usize) -> f32 {
        let column =
            |counts: &[[f32; BINS]; MAX_DRAFTS]| counts.iter().map(|row| row[b]).sum::<f32>();
        let pooled = (column(&self.accepted) + PRIOR_WEIGHT * PRIOR[b])
            / (column(&self.verified) + PRIOR_WEIGHT);
        (self.accepted[depth][b] + POOL_WEIGHT * pooled) / (self.verified[depth][b] + POOL_WEIGHT)
    }

    /// Survival of each draft: the chance it and every draft before it are
    /// accepted. Non-increasing, so a cut keeps a prefix.
    pub(crate) fn survival(&self, conf: &[f32]) -> [f32; MAX_DRAFTS] {
        let mut out = [0.0; MAX_DRAFTS];
        let mut alive = 1.0;
        for (depth, &c) in conf.iter().take(MAX_DRAFTS).enumerate() {
            alive *= self.rate(depth, bin(c));
            out[depth] = alive;
        }
        out
    }

    /// One verify of the first `drafted` drafts that accepted `accepted`.
    /// Position `j` is observed only when every earlier draft was accepted.
    pub(crate) fn record(&mut self, conf: &[f32], drafted: usize, accepted: usize) {
        for cells in [&mut self.verified, &mut self.accepted] {
            cells.iter_mut().flatten().for_each(|cell| *cell *= DECAY);
        }
        let observed = drafted.min(accepted + 1).min(MAX_DRAFTS);
        for (depth, &c) in conf.iter().take(observed).enumerate() {
            if c.is_finite() {
                self.verified[depth][bin(c)] += 1.0;
                if depth < accepted {
                    self.accepted[depth][bin(c)] += 1.0;
                }
            }
        }
    }
}

/// Tokens one verify row must be expected to yield, and the step period of
/// the full-width verify.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Params {
    pub(crate) tau: f32,
    pub(crate) probe: u64,
}

pub(crate) const DEFAULT_PARAMS: Params = Params {
    tau: 0.3,
    probe: 64,
};

/// Step ms for `owners` owners verifying `rows` rows each, from the release
/// engine's nsys step profile of 2026-09-30 (GLM-5.3 Flash TP2 GB10, prose,
/// rank 0 step period by verify rows): one owner 44.4 ms + 5.68 ms a row
/// (3..=8 rows; 5.3 ms of each row is MoE), four owners 109.8 ms + 3.2 ms an
/// owner-row (16..=32 rows). Two and three owners interpolate; more than four
/// keep four's per-row cost. A lone owner's 2-row verify takes the slower
/// path the 2026-09-28 sweep measured (10.3 ms over its 3-row one); a batch
/// past the 32-row verify budget never runs. Only differences between widths
/// at one owner count decide a width.
pub(super) fn step_ms(owners: usize, rows: usize) -> f32 {
    const ROW_BUDGET: usize = 32;
    const LONE_TWO_ROWS_MS: f32 = 10.3;
    let owners = owners.max(1);
    if owners * rows > ROW_BUDGET {
        return f32::INFINITY;
    }
    if owners == 1 && rows == 2 {
        return step_ms(1, 3) + LONE_TWO_ROWS_MS;
    }
    let t = (owners - 1) as f32 / 3.0;
    let base = 44.4 + (109.8 - 44.4) * t;
    let per_row = 5.68 + (3.2 - 5.68) * t.min(1.0);
    base + per_row * (owners * rows) as f32
}

/// Milliseconds one more row costs a lone owner: the row `tau` prices.
fn row_ms() -> f32 {
    step_ms(1, 4) - step_ms(1, 3)
}

/// The calibration and the verify counter behind the periodic full width.
#[derive(Default)]
pub(crate) struct Policy {
    pub(crate) calibration: Calibration,
    steps: u64,
}

impl Policy {
    /// Drafts per owner (1..=`max`) for owners with these confidences
    /// verifying together; `None` when any owner's drafts are not measured
    /// (the caller's policy stands).
    pub(crate) fn choose<'a>(
        &mut self,
        owners: impl ExactSizeIterator<Item = &'a [f32]> + Clone,
        max: usize,
        params: Params,
    ) -> Option<usize> {
        let max = max.min(MAX_DRAFTS);
        let measured =
            |conf: &[f32]| conf.len() >= max && conf[..max].iter().all(|c| c.is_finite());
        if max == 0 || owners.len() == 0 || !owners.clone().all(measured) {
            return None;
        }
        self.steps += 1;
        if params.probe > 0 && self.steps.is_multiple_of(params.probe) {
            return Some(max);
        }
        let n = owners.len();
        let price = params.tau / row_ms();
        let mut expected = n as f32;
        let (mut width, mut value) = (1, f32::NEG_INFINITY);
        for w in 1..=max {
            expected += owners
                .clone()
                .map(|conf| self.calibration.survival(&conf[..max])[w - 1])
                .sum::<f32>();
            let v = expected - price * step_ms(n, w + 1);
            if v > value {
                (width, value) = (w, v);
            }
        }
        Some(width)
    }
}

fn flag(name: &str) -> bool {
    std::env::var(name).as_deref() == Ok("1")
}

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_DFLASH_CONF_WIDTH"))
}

fn log_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| flag("ATLAS_DFLASH_CONF_LOG"))
}

fn params() -> Params {
    static P: OnceLock<Params> = OnceLock::new();
    *P.get_or_init(|| {
        let parsed = |name: &str| std::env::var(name).ok().and_then(|v| v.trim().parse().ok());
        Params {
            tau: parsed("ATLAS_DFLASH_CONF_TAU")
                .filter(|tau: &f32| tau.is_finite() && *tau >= 0.0)
                .unwrap_or(DEFAULT_PARAMS.tau),
            probe: std::env::var("ATLAS_DFLASH_CONF_PROBE")
                .ok()
                .and_then(|v| v.trim().parse().ok())
                .unwrap_or(DEFAULT_PARAMS.probe),
        }
    })
}

fn policy() -> std::sync::MutexGuard<'static, Policy> {
    static POLICY: OnceLock<Mutex<Policy>> = OnceLock::new();
    POLICY
        .get_or_init(Mutex::default)
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

/// [`Policy::choose`] on the serving policy; `None` with the rule off.
pub(crate) fn choose<'a>(
    owners: impl ExactSizeIterator<Item = &'a [f32]> + Clone,
    max: usize,
) -> Option<usize> {
    if !enabled() {
        return None;
    }
    policy().choose(owners, max, params())
}

/// Feed one owner's verify outcome to the calibration (`conf` is the
/// drafter's confidence in the drafts that owner held).
pub(crate) fn record(conf: &[f32], drafted: usize, accepted: usize) {
    if log_enabled() {
        tracing::info!("DFLASH CONF drafted={drafted} accepted={accepted} conf={conf:.3?}");
    }
    if enabled() && conf.len() >= drafted {
        policy().calibration.record(conf, drafted, accepted);
    }
}

#[cfg(test)]
#[path = "dflash_conf_width_tests.rs"]
mod tests;
