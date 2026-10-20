// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request dynamic MTP draft ceiling for deep speculation
//! (`ATLAS_MTP_DYNAMIC_DEPTH=1`, `spark_model::speculative::deep_depth`),
//! steering between a floor (default 3) and `--num-drafts` (up to 7 on
//! qwen4_exp's exact lane).
//!
//! The statistic is the acceptance of the DEEPEST position the ceiling
//! allows, over the verifies that actually drafted it:
//!
//! ```text
//!   a_d = #(verifies with >= d drafts that accepted >= d) / #(verifies with >= d drafts)
//! ```
//!
//! That is P(accepted >= d | position d drafted), the per-position rate vLLM
//! reports (it always drafts every position). It is the extra yield the
//! deepest verify row buys, which is what the next row would have to beat.
//! Verifies that drafted fewer positions (the drafter's confidence stop,
//! D-Cut, a row budget) say nothing about position `d` and are not counted:
//! under a confidence stop a deep ceiling costs drafter and verify rows only
//! on confident runs, so the controller steers what it costs when it binds.
//!
//! Every [`WINDOW`] verifies at one ceiling, with at least [`MIN_DRAFTED`]
//! observations of position `d`:
//!
//! | a_d | step |
//! |---|---|
//! | > 0.60 | +2 |
//! | > 0.45 | +1 |
//! | < 0.15 | -2 |
//! | < 0.25 | -1 |
//!
//! (the thresholds of the RecoverSSM/vLLM K=3..7 recipe). A request starts at
//! the floor, so the first deep rows are earned, not assumed. Fewer than
//! [`MIN_DRAFTED`] observations hold the ceiling: the confidence stop rarely
//! reaches it, so it rarely costs anything.

/// The scheduler levers the per-request depth controllers answer to.
#[derive(Debug, Default, Clone, Copy)]
pub struct DepthLevers {
    /// `ATLAS_MTP_SINGLE_DEPTH_ADAPT`: the n=1 ladders (GLM K3/K5, 1..=3).
    pub adapt: bool,
    /// `ATLAS_MTP_DYNAMIC_DEPTH`: the deep ceiling (this module).
    pub deep: bool,
}

/// Whether a step of `n` sequences under a `ceiling`-draft ceiling drafts
/// past the ladder: the lever, a ceiling past 3, and a width
/// `deep_depth::deep_max_seqs()` admits.
pub fn deep_step(levers: DepthLevers, ceiling: usize, n: usize) -> bool {
    levers.deep
        && ceiling > super::mtp_depth_ladder::MAX_DEPTH
        && n <= spark_model::speculative::deep_depth::deep_max_seqs()
}

/// Verifies per decision.
pub const WINDOW: u16 = 48;
/// Observations of the deepest position a decision needs.
pub const MIN_DRAFTED: u16 = 12;
const PROMOTE_2: f32 = 0.60;
const PROMOTE_1: f32 = 0.45;
const DEMOTE_1: f32 = 0.25;
const DEMOTE_2: f32 = 0.15;

#[derive(Debug, Default, Clone, Copy)]
pub struct DeepDepth {
    /// Current ceiling; 0 until the first verify (= the floor).
    depth: u8,
    steps: u16,
    drafted: u16,
    hits: u16,
    pub switches: u16,
}

/// The ceiling step for deepest-position acceptance `a`.
fn step_for(a: f32) -> isize {
    if a > PROMOTE_2 {
        2
    } else if a > PROMOTE_1 {
        1
    } else if a < DEMOTE_2 {
        -2
    } else if a < DEMOTE_1 {
        -1
    } else {
        0
    }
}

impl DeepDepth {
    /// Draft ceiling under `floor..=max`.
    pub fn drafts(&self, floor: usize, max: usize) -> usize {
        let floor = floor.clamp(1, max.max(1));
        if self.depth == 0 {
            floor
        } else {
            (self.depth as usize).clamp(floor, max.max(1))
        }
    }

    /// Feed one verify that verified `drafts` drafts and accepted `accepted`.
    pub fn record(&mut self, drafts: usize, accepted: usize, floor: usize, max: usize) {
        let cur = self.drafts(floor, max);
        self.depth = cur as u8;
        self.steps += 1;
        if drafts >= cur {
            self.drafted += 1;
            self.hits += u16::from(accepted >= cur);
        }
        if self.steps < WINDOW {
            return;
        }
        let a = self.hits as f32 / self.drafted.max(1) as f32;
        let next = if self.drafted < MIN_DRAFTED {
            cur
        } else {
            let lo = floor.clamp(1, max.max(1)) as isize;
            (cur as isize + step_for(a)).clamp(lo, max.max(1) as isize) as usize
        };
        if next != cur {
            self.switches = self.switches.saturating_add(1);
            tracing::info!(
                "MTP dynamic depth: {cur} -> {next} drafts (a_{cur}={a:.2} over {} of {} verifies)",
                self.drafted,
                self.steps
            );
        }
        self.depth = next as u8;
        self.steps = 0;
        self.drafted = 0;
        self.hits = 0;
    }
}

/// Per-position acceptance over the verifies that DRAFTED the position:
/// `acc_i = #(accepted >= i) / #(drafts >= i)` — vLLM's per-position rate.
/// `RequestAccept`'s `surv` divides by every MTP step instead (bootstraps,
/// decode rows, confidence-stopped and ladder-shallow verifies count as
/// rejections there), so a deep position the confidence stop or the depth
/// controller seldom reaches reads far lower in `surv` than it accepts.
#[derive(Debug, Default, Clone, Copy)]
pub struct PositionAccept {
    drafted: [u64; POSITIONS],
    hit: [u64; POSITIONS],
}

/// Positions tracked (7 drafts).
pub const POSITIONS: usize = 7;

impl PositionAccept {
    pub fn record(&mut self, drafts: usize, accepted: usize) {
        for i in 0..drafts.min(POSITIONS) {
            self.drafted[i] += 1;
            self.hit[i] += u64::from(accepted > i);
        }
    }

    /// `acc_i` for every position, `-` where none was drafted.
    pub fn suffix(&self) -> String {
        let rates: Vec<String> = (0..POSITIONS)
            .map(|i| match self.drafted[i] {
                0 => "-".to_string(),
                d => format!("{:.2}", self.hit[i] as f64 / d as f64),
            })
            .collect();
        rates.join(",")
    }
}

/// Fit per-sequence draft counts `caps` (each >= 0) into `budget` verify
/// rows (`Σ (caps[i] + 1)` for a sequence with drafts, 1 for one without):
/// the deepest sequence gives up a draft until the batch fits, never below
/// one draft (D-Cut's floor). Lower indices win ties, so the result is a
/// pure function of the inputs.
pub fn fit_row_budget(caps: &mut [usize], budget: usize) {
    let rows = |c: &[usize]| c.iter().map(|&d| d + 1).sum::<usize>();
    while rows(caps) > budget {
        let Some((i, _)) = caps
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d > 1)
            .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(&a.0)))
        else {
            return;
        };
        caps[i] -= 1;
    }
}

#[cfg(test)]
#[path = "mtp_deep_depth_tests.rs"]
mod tests;
