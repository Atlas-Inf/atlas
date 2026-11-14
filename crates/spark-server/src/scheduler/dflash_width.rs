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

/// A batched DFlash step's draft depth. Every member verifies the same
/// width and only members holding that many drafts batch, so past two
/// sequences the depth is at most 7 (8 rows each): the batched verify is
/// bound by the weight stream's L2 reuse past ~24 rows, and 16 rows for
/// each of 8 sequences halved the C8 rate (59 vs 114 tok/s). Two sequences
/// keep the deepest width both hold, but never less than 7, so a long-block
/// drafter's sequence (DSpine, 15 drafts) still batches with a short one's
/// (DFlash2, 7) under `--draft-model-alt`. A 7-draft drafter is unchanged.
///
/// Past eight sequences the depth narrows with the batch so a step's rows
/// stay within one exact batched verify (128 rows, `VERIFY_ROW_CAP`):
/// [`wide_depth`].
pub(crate) fn batch_depth(nd: usize, n: usize, held: impl Iterator<Item = usize>) -> usize {
    match n {
        0 | 1 => nd,
        2 => held.min().map_or(nd, |h| nd.min(h.max(MAX_DRAFTS))),
        _ => nd.min(wide_depth(n)).max(1),
    }
}

/// Draft depth for `n` live sequences: `ATLAS_DFLASH_BATCH_DEPTH`
/// (`n:depth,...`, ascending; the first entry whose `n` covers the batch
/// applies, the last beyond it) or [`WIDE_DEPTHS`].
pub(crate) fn wide_depth(n: usize) -> usize {
    static T: OnceLock<Vec<(usize, usize)>> = OnceLock::new();
    let table = T.get_or_init(|| {
        parse_depths(std::env::var("ATLAS_DFLASH_BATCH_DEPTH").ok().as_deref())
            .unwrap_or_else(|| WIDE_DEPTHS.to_vec())
    });
    depth_for(table, n)
}

/// Measured on the 27B (Qwen3.8-27B NVFP4 + DFlash2, one GB10, exact batched
/// verify, think off, one wave of 300 tokens; aggregate tok/s):
///
/// | live | 7 drafts | 3 drafts | 1 draft | plain decode |
/// |---|---|---|---|---|
/// | 8  | 131 | | | 82 |
/// | 16 | 95  | 115 | | 110 |
/// | 32 | 102 | 119 | ~123 | 146 |
///
/// A verify row costs about 3.2 ms up to 64 rows and 4.2 ms at 128 (8 rows
/// 72 ms, 64 rows 206 ms, 128 rows 535 ms), and the drafter's propose about
/// 8 ms a sequence, so past eight sequences deep blocks lose and past
/// sixteen plain decode delivers the most: depth 0 there (`spec_width_ok`).
pub(crate) const WIDE_DEPTHS: [(usize, usize); 3] = [(8, 7), (16, 3), (usize::MAX, 0)];

fn depth_for(table: &[(usize, usize)], n: usize) -> usize {
    table
        .iter()
        .find(|&&(upto, _)| n <= upto)
        .or(table.last())
        .map_or(MAX_DRAFTS, |&(_, d)| d.min(MAX_DRAFTS))
}

/// Whether `n` live sequences speculate at all: a depth of 0 in the table
/// is plain batched decode, which past the speculation width delivers more
/// (every verify row costs about a decode row; see [`WIDE_DEPTHS`]).
pub(crate) fn spec_width_ok(n: usize) -> bool {
    n <= 2 || wide_depth(n) > 0
}

fn parse_depths(spec: Option<&str>) -> Option<Vec<(usize, usize)>> {
    let v: Vec<(usize, usize)> = spec?
        .split(',')
        .filter_map(|e| {
            let (n, d) = e.trim().split_once(':')?;
            Some((n.parse().ok()?, d.parse().ok()?))
        })
        .collect();
    (!v.is_empty()).then_some(v)
}

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

/// The tensor-parallel ranks the verify runs on, set once at startup
/// ([`set_verify_ranks`]); unset is the pair.
static VERIFY_RANKS: OnceLock<usize> = OnceLock::new();

/// Record the ranks a verify step runs on, which select the measured step
/// costs the widths are priced by: the TP2 pair's or the TP4 chain's.
pub(crate) fn set_verify_ranks(ranks: usize) {
    let _ = VERIFY_RANKS.set(ranks);
}

/// Whether the verify runs on the chain (more than two ranks).
pub(super) fn on_chain() -> bool {
    VERIFY_RANKS.get().is_some_and(|&ranks| ranks > 2)
}

/// What a lone owner's 2-row verify costs over its 3-row one on the fused
/// two-row path (`moe::forward_k2`): the pair's 2026-09-28 sweep measured
/// 10.3 ms. The chain has not measured it and is charged the same. A pair
/// under expert TP has no such path ([`lone_two_on_row_line`]).
pub(super) const LONE_TWO_ROWS_MS: f32 = 10.3;

/// Whether the verify runs on a two-rank pair under expert TP, set once at
/// startup ([`set_pair_expert_tp`]); unset is not.
static PAIR_EXPERT_TP: OnceLock<bool> = OnceLock::new();

/// Record that the verify runs on a two-rank pair under expert TP.
pub(crate) fn set_pair_expert_tp(on: bool) {
    let _ = PAIR_EXPERT_TP.set(on);
}

/// Whether a lone owner's 2-row verify is priced on its row line instead of
/// carrying [`LONE_TWO_ROWS_MS`]: on the pair under expert TP, where it runs
/// the grouped path its 3-row verify does and measured 42.0 ms against 47.4
/// (SparkGLM 0.3.2, prose, 2026-11-11, `ATLAS_DFLASH_FIXED_WIDTH=1`). A
/// single GPU and a pair without expert TP still take the fused path.
pub(super) fn lone_two_on_row_line() -> bool {
    !on_chain() && PAIR_EXPERT_TP.get().copied().unwrap_or(false)
}

/// Verify step cost in ms on the TP4 chain for `owners` owners of `rows`
/// rows each: per owner count, a fixed part and a cost per owner-row, fitted
/// to the median scheduler step interval by (owners, rows) of the chain's
/// rank 0 (integ ba843fb33, recipe 4x1m-tp4 16161cb, `ATLAS_DFLASH_WIDTH_LOG`,
/// four and eight concurrent streams of mixed code / prose / structured /
/// counting, 2026-10-18; 1470 verifies). Every (owners, rows) cell with ten
/// or more verifies lies within 6% of the fit; to 56 rows (five to eight owners of
/// five to seven, a 64-row budget then) within 8%. Past the verify budget
/// (`glm_long_owner::ROW_BUDGET`) it is infinite.
pub(super) fn chain_step_ms(owners: usize, rows: usize) -> f32 {
    const BASE: [f32; 8] = [19.0, 30.0, 37.0, 50.0, 56.0, 62.8, 66.8, 68.0];
    let owners = owners.clamp(1, 8);
    if owners * rows > spark_model::layer::glm_long_owner::ROW_BUDGET {
        return f32::INFINITY;
    }
    let lone_two = if owners == 1 && rows <= 2 {
        LONE_TWO_ROWS_MS
    } else {
        0.0
    };
    BASE[owners - 1] + PER_OWNER_ROW[owners - 1] * (owners * rows) as f32 + lone_two
}

/// [`chain_step_ms`]'s cost per owner-row, by owner count.
const PER_OWNER_ROW: [f32; 8] = [4.35, 2.7, 2.5, 2.0, 2.1, 2.1, 2.1, 2.0];

/// What a dead row (`glm_long_owner::DEAD_ROW`) still costs: a row's work
/// besides its routed experts, which a dead row skips. Not yet measured on
/// the chain; kindling's TP4 step model (`experimental/adaptive-k`,
/// `VLLM_ADAPTIVE_K_MODEL` 27.0,0.635,0.542) charges 0.54 ms a token besides
/// the experts it touches.
const DEAD_ROW_MS: f32 = 0.55;

/// [`chain_step_ms`] when only `live` of the `owners * rows` rows are live,
/// the rest dead rows.
pub(super) fn chain_step_ms_live(owners: usize, rows: usize, live: usize) -> f32 {
    let dead = (owners * rows).saturating_sub(live) as f32;
    let saved = PER_OWNER_ROW[owners.clamp(1, 8) - 1] - DEAD_ROW_MS;
    chain_step_ms(owners, rows) - dead * saved.max(0.0)
}

/// Measured verify step cost in ms for `owners` sequences of `rows` rows
/// each: median scheduler step interval (verify + re-propose) on GLM-5.3
/// Flash TP2 GB10, prose, 2026-09-28 (`ATLAS_DFLASH_FIXED_WIDTH` sweeps; 5..=8
/// owners on the eight-sequence profile); on the chain, [`chain_step_ms`].
fn step_ms(owners: usize, rows: usize) -> f32 {
    if on_chain() {
        return chain_step_ms(owners, rows.clamp(2, 8));
    }
    // Rows 2..=8, owners 1..=8. A lone owner's 2-row verify takes a slower
    // path than 3 rows, except on the pair under expert TP, where it is the
    // 3-row cell less the 5.4 ms measured on 2026-11-11. Owners x rows past
    // the 32-row verify budget never run (infinite cost).
    const X: f32 = f32::INFINITY;
    if owners <= 1 && rows <= 2 && lone_two_on_row_line() {
        return 68.1;
    }
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

/// `ATLAS_DFLASH_WIDTH_LOG=1`: one line per verify (cost sweeps), with each
/// owner's verified drafts when some rows are dead (`verified`).
pub(crate) fn log_verify(owners: usize, drafts: usize, verified: &[usize]) {
    static ON: OnceLock<bool> = OnceLock::new();
    if *ON.get_or_init(|| std::env::var("ATLAS_DFLASH_WIDTH_LOG").as_deref() == Ok("1")) {
        if verified.iter().any(|&d| d < drafts) {
            tracing::info!(
                "DFLASH VERIFY owners={owners} rows={} live={}",
                drafts + 1,
                verified.iter().map(|d| d + 1).sum::<usize>()
            );
        } else {
            tracing::info!("DFLASH VERIFY owners={owners} rows={}", drafts + 1);
        }
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

/// With dead rows (the chain; `glm_long_owner::DEAD_ROW`): the batch width
/// and each owner's verified drafts (1..=width, in `owners` order), from the
/// drafter's confidence (`dflash_conf_width::choose_live`). `None` leaves
/// [`choose`] in place: off the chain, a pinned width, or unmeasured drafts.
pub(crate) fn choose_live<'a>(
    owners: impl ExactSizeIterator<Item = &'a [f32]> + Clone,
    max: usize,
) -> Option<(usize, Vec<usize>)> {
    let max = max.min(MAX_DRAFTS);
    if max == 0 || fixed().is_some() || !on_chain() {
        return None;
    }
    super::dflash_conf_width::choose_live(owners, max)
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
