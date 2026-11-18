// SPDX-License-Identifier: AGPL-3.0-only

//! Per-owner DFlash verify widths on the pair
//! (`ATLAS_GLM_PAIR_LIVE_WIDTHS=1`, default off).
//!
//! Owners of one owner-batched verify share a width (`dflash_width`), so a
//! batch of prose and code either verifies rows the prose owner's drafter
//! never believed in or cuts the code owner short. The chain already lets
//! each owner verify its own number of drafts inside the uniform batch, the
//! rest of its rows dead (`glm_long_owner::DEAD_ROW`: a dead row skips its
//! routed experts, most of what a row costs). With this switch the pair does
//! the same (`dflash_width::choose_live`), priced by the pair's own step
//! costs below; the chain keeps its model (`dflash_width::chain_step_ms_live`).
//!
//! Verification is exact at any width and a dead row's pick is never read,
//! so only the rows a step pays for change. Every owner still verifies at
//! least its first draft, the batch stays within the verify row budget, and
//! the periodic full-width probe verifies every draft
//! (`dflash_conf_width::Policy::choose_live_pair`). The head alone reads the
//! switch: a worker learns a dead row from its token's mark.
//!
//! `ATLAS_DFLASH_WIDTH_LOG=1` logs each such verify's live rows
//! (`dflash_width::log_verify`), and `ATLAS_GLM_PAIR_DEAD_ROW_MS` overrides
//! what a dead row is charged ([`PAIR_DEAD_ROW_MS`]) while that is measured.

use std::sync::OnceLock;

/// `ATLAS_GLM_PAIR_LIVE_WIDTHS=1`. Read once per process.
pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PAIR_LIVE_WIDTHS").as_deref() == Ok("1"))
}

/// [`uniform_step_ms`]'s fixed part for one owner, and what each further
/// owner adds to it.
const BASE_MS: f32 = 34.5;
const OWNER_MS: f32 = 11.3;

/// [`uniform_step_ms`]'s cost per live owner-row, by owner count (four's for
/// more owners).
const LIVE_ROW_MS: [f32; 4] = [6.0, 5.2, 4.2, 3.7];

/// What a dead row (`glm_long_owner::DEAD_ROW`) still costs on the pair: a
/// row's work besides its routed experts, which a dead row skips. Measured
/// with the live rows' costs ([`uniform_step_ms`]): 1.27, 1.44 and 1.35 ms
/// at two, three and four owners. `ATLAS_GLM_PAIR_DEAD_ROW_MS` overrides it
/// for a sweep.
pub(super) const PAIR_DEAD_ROW_MS: f32 = 1.35;

fn live_row_ms(owners: usize) -> f32 {
    LIVE_ROW_MS[owners.clamp(1, LIVE_ROW_MS.len()) - 1]
}

/// Step ms on the pair (verify, the owners' tails and the batched propose)
/// for `owners` owners of `rows` live rows each: per owner count, a fixed
/// part and a cost per owner-row, fitted to the medians of
/// `GLM OWNER STEP_TIMING`'s total and of a lone sequence's verify + propose
/// (SparkGLM 0.3.4 engine, TP2 pair under expert TP, 2026-11-14;
/// `ATLAS_DFLASH_FIXED_WIDTH` sweeps). Measured, rows per owner -> ms:
///
/// - one owner: 2 -> 46.8, 3 -> 53.5, 4 -> 61.0, 5 -> 65.9, 6 -> 69.9,
///   8 -> 82.5
/// - two owners: 2 -> 66.3, 3 -> 79.5, 4 -> 89.0, 5 -> 101, 7 -> 117,
///   8 -> 128
/// - four owners: 2 -> 100.6, 3 -> 119..128, 4 -> 134..139, 5 -> 153..157,
///   6 -> 166, 7 -> 177..181, 8 -> 189
///
/// The fixed parts and row costs are the least-squares fit of step ms to
/// live and dead rows over 4,312 owner-batched verifies with this switch on
/// (`DFLASH VERIFY ... live=`, the same day, mixed prose and structured
/// output at two to four streams): 45.9 + 5.18 a live row at two owners,
/// 57.9 + 4.19 at three, 68.5 + 3.71 at four, median error 2.6 to 4.4 ms.
/// The uniform cells above lie within 10% of it (a range by its middle).
/// Past the verify budget (`glm_long_owner::ROW_BUDGET`) it is infinite.
///
/// Not `dflash_conf_width::pair_step_ms`, the 2026-09-30 costs the shared
/// width is priced by: that one is left as it is.
pub(super) fn uniform_step_ms(owners: usize, rows: usize) -> f32 {
    let owners = owners.max(1);
    if owners * rows > spark_model::layer::glm_long_owner::ROW_BUDGET {
        return f32::INFINITY;
    }
    BASE_MS + OWNER_MS * (owners - 1) as f32 + live_row_ms(owners) * (owners * rows) as f32
}

/// [`uniform_step_ms`] when only `live` of the `owners * rows` rows are live
/// and a dead row costs `dead_row_ms`: each dead row saves a live row's cost
/// less its own.
fn step_ms_live_at(owners: usize, rows: usize, live: usize, dead_row_ms: f32) -> f32 {
    let dead = (owners.max(1) * rows).saturating_sub(live) as f32;
    let saved = live_row_ms(owners) - dead_row_ms;
    uniform_step_ms(owners, rows) - dead * saved.max(0.0)
}

/// [`uniform_step_ms`] when only `live` of the `owners * rows` rows are
/// live, the rest dead rows at this process's dead-row cost.
pub(super) fn step_ms_live(owners: usize, rows: usize, live: usize) -> f32 {
    step_ms_live_at(owners, rows, live, dead_row_ms())
}

/// `ATLAS_GLM_PAIR_DEAD_ROW_MS` as set: a finite cost of at least 0 ms, else
/// [`PAIR_DEAD_ROW_MS`].
fn parse_dead_row_ms(raw: Option<&str>) -> f32 {
    raw.and_then(|v| v.trim().parse::<f32>().ok())
        .filter(|ms| ms.is_finite() && *ms >= 0.0)
        .unwrap_or(PAIR_DEAD_ROW_MS)
}

/// What this process charges a dead row. Read once.
fn dead_row_ms() -> f32 {
    static MS: OnceLock<f32> = OnceLock::new();
    *MS.get_or_init(|| {
        parse_dead_row_ms(std::env::var("ATLAS_GLM_PAIR_DEAD_ROW_MS").ok().as_deref())
    })
}

#[cfg(test)]
#[path = "dflash_pair_live_tests.rs"]
mod tests;
