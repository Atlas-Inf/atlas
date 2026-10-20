// SPDX-License-Identifier: AGPL-3.0-only

//! Per-request MTP draft-depth controller for a 2..=3-draft ceiling
//! (qwen4_exp K=2..4 under `ATLAS_QWEN4EXP_MTP_DEPTH`), armed by the same
//! lever as the GLM K3/K5 controller (`ATLAS_MTP_SINGLE_DEPTH_ADAPT=1`,
//! single sequence only).
//!
//! It picks the depth `d` in `1..=max` that maximizes delivered tokens per
//! unit of step wall:
//!
//! ```text
//!   yield(d) = 1 + S_1 + ... + S_d        S_i = P(accepted >= i)
//!   score(d) = yield(d) / wall(d)
//! ```
//!
//! `S_i` is measured, not modelled: every verify at depth >= i observes
//! position i, and `P(accepted >= i)` does not depend on how many drafts
//! follow position i. `wall(d)` is the measured interval between two verify
//! records at depth `d` (at n=1 that interval is one whole step: the
//! propose of `d` drafts, the K=d+1 verify, the emit), smoothed; a depth
//! not yet run is extrapolated from the nearest measured one by
//! `ATLAS_MTP_DEPTH_ROW_COST` (default 0.2) of a step per extra row.
//!
//! Exploration: a request starts at the deepest depth (every position is
//! observed), re-decides every [`WINDOW`] verifies with a [`HYSTERESIS`]
//! margin, and while shallower re-probes the deepest depth for one window
//! every [`REPROBE`] verifies — sooner when its deepest observed position
//! accepts above [`PROMOTE`].

use std::time::{Duration, Instant};

/// Deepest draft count this controller steers.
pub const MAX_DEPTH: usize = 3;
/// Verifies per decision.
const WINDOW: u16 = 16;
/// Verifies at a shallower depth between deep probes.
const REPROBE: u16 = 128;
/// Survival of the deepest observed position that pulls a probe forward.
const PROMOTE: f32 = 0.85;
/// Relative score margin a different depth must clear.
const HYSTERESIS: f32 = 0.03;
/// EWMA weights for survival and step wall.
const SURV_ALPHA: f32 = 0.1;
const WALL_ALPHA: f32 = 0.2;
/// An interval longer than this spans something other than one step
/// (serial decode, admission, a prefill chunk) and is not a wall sample.
const MAX_STEP: Duration = Duration::from_secs(2);

fn row_cost() -> f32 {
    static C: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
    *C.get_or_init(|| {
        std::env::var("ATLAS_MTP_DEPTH_ROW_COST")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|c: &f32| c.is_finite() && *c >= 0.0)
            .unwrap_or(0.2)
    })
}

#[derive(Debug, Default, Clone, Copy)]
pub struct DepthLadder {
    /// Current depth; 0 until the first verify (= the ceiling).
    depth: u8,
    /// `S_i` for positions 1..=3 and how many verifies observed each.
    surv: [f32; MAX_DEPTH],
    seen: [u32; MAX_DEPTH],
    /// Smoothed step wall (ms) by depth (index = drafts); 0 = never run.
    wall_ms: [f32; MAX_DEPTH + 1],
    last: Option<Instant>,
    window: u16,
    since_probe: u16,
    pub switches: u16,
}

impl DepthLadder {
    /// Drafts to propose next under a ceiling of `max`.
    pub fn drafts(&self, max: usize) -> usize {
        if self.depth == 0 {
            max
        } else {
            (self.depth as usize).min(max)
        }
    }

    /// A serial step ran: the next interval is not a verify step.
    pub fn note_serial(&mut self) {
        self.last = None;
    }

    /// Feed one verify of `drafts` drafts that accepted `accepted`.
    pub fn record(&mut self, drafts: usize, accepted: usize, max: usize, now: Instant) {
        let max = max.clamp(1, MAX_DEPTH);
        let d = drafts.clamp(1, MAX_DEPTH);
        if let Some(t) = self.last {
            let dt = now.saturating_duration_since(t);
            if dt <= MAX_STEP {
                let ms = dt.as_secs_f32() * 1e3;
                let w = &mut self.wall_ms[d];
                *w = if *w == 0.0 {
                    ms
                } else {
                    *w + WALL_ALPHA * (ms - *w)
                };
            }
        }
        self.last = Some(now);
        for i in 0..d {
            let s = if accepted > i { 1.0 } else { 0.0 };
            self.surv[i] = if self.seen[i] == 0 {
                s
            } else {
                self.surv[i] + SURV_ALPHA * (s - self.surv[i])
            };
            self.seen[i] = self.seen[i].saturating_add(1);
        }
        if self.depth == 0 {
            self.depth = max as u8;
        }
        self.window += 1;
        if self.window < WINDOW {
            return;
        }
        self.window = 0;
        let cur = (self.depth as usize).min(max);
        let next = if cur < max {
            self.since_probe = self.since_probe.saturating_add(WINDOW);
            if self.since_probe >= REPROBE || self.surv[cur - 1] >= PROMOTE {
                self.since_probe = 0;
                max
            } else {
                self.choose(cur, max)
            }
        } else {
            self.choose(cur, max)
        };
        if next != cur {
            self.switches = self.switches.saturating_add(1);
            tracing::info!(
                "MTP depth ladder: {cur} -> {next} drafts (S={:.2},{:.2},{:.2} wall_ms={:.1},{:.1},{:.1})",
                self.surv[0],
                self.surv[1],
                self.surv[2],
                self.wall_ms[1],
                self.wall_ms[2],
                self.wall_ms[3],
            );
        }
        self.depth = next as u8;
    }

    /// Step wall at depth `d`, extrapolated from the nearest measured one.
    fn wall(&self, d: usize) -> Option<f32> {
        if self.wall_ms[d] > 0.0 {
            return Some(self.wall_ms[d]);
        }
        let near = (1..=MAX_DEPTH)
            .filter(|&e| self.wall_ms[e] > 0.0)
            .min_by_key(|&e| e.abs_diff(d))?;
        let f = row_cost();
        Some(self.wall_ms[near] * (1.0 + f * d as f32) / (1.0 + f * near as f32))
    }

    fn score(&self, d: usize) -> Option<f32> {
        let yield_ = 1.0 + self.surv[..d].iter().sum::<f32>();
        self.wall(d).map(|w| yield_ / w)
    }

    fn choose(&self, cur: usize, max: usize) -> usize {
        let Some(base) = self.score(cur) else {
            return cur;
        };
        let (mut best, mut best_score) = (cur, base);
        for d in 1..=max {
            if self.seen[d - 1] == 0 {
                continue; // never observed: no yield estimate
            }
            if let Some(s) = self.score(d)
                && s > best_score
            {
                best = d;
                best_score = s;
            }
        }
        if best != cur && best_score > base * (1.0 + HYSTERESIS) {
            best
        } else {
            cur
        }
    }
}

#[cfg(test)]
#[path = "mtp_depth_ladder_tests.rs"]
mod tests;
