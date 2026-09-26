// SPDX-License-Identifier: AGPL-3.0-only

//! Adaptive per-sequence γ for generic DFlash2 (`ATLAS_DFLASH_ADAPTIVE_GAMMA=1`).
//!
//! Evidence (reiner job 260, C=1): γ=12 wins on code/JSON (47.9 / 42.6 vs
//! 37.6 / 39.3 at γ=8) but loses on prose (14.2 vs 14.9) and long code
//! (26.2 vs 29.0); mean accepted was 5.1 at γ=8 and 7.8 at γ=12, ~1.4 for
//! prose. A per-sequence EMA of accepted draft tokens picks between the
//! floor γ=8 and the configured γ_max (12 in the shipped set) — with
//! hysteresis so a sequence near the boundary does not oscillate.
//!
//! Every buffer and pool stays sized for the configured γ_max; a γ=8
//! sequence uses a prefix of its γ_max-shaped allocations. The lever off is
//! byte-identical to the legacy path: `propose_gamma` is initialised to
//! the configured γ and is never updated.

/// Floor γ for the adaptive arm. The shipped set is {8, 12}; γ_max is the
/// head's configured `--dflash-gamma`.
pub(crate) const ADAPTIVE_GAMMA_LO: usize = 8;
/// Thresholds are ACCEPTANCE RATIOS — EMA of accepted drafts over the
/// current γ's draft count (γ − 1) — because the raw accepted count scales
/// with γ (job 260 code: 5.1 of 7 at γ=8, 7.8 of 11 at γ=12; prose ~1.4 of
/// 7). Rise from the floor above 0.65 (code 0.73 rises, prose 0.20 stays);
/// fall from γ_max below 0.45. The gap is the hysteresis band.
const RATIO_RISE: f32 = 0.65;
const RATIO_FALL: f32 = 0.45;

/// Exponential moving average of ACCEPTED DRAFT tokens per verify step —
/// the bonus token is never counted (`num_accepted` already excludes it).
/// 0.75/0.25 weights ~4-step memory: fast enough to follow a mode change
/// (prose → code mid-generation), slow enough that one bad step at γ=12
/// does not immediately collapse back to γ=8.
pub fn update_ema(ema: f32, accepted: usize) -> f32 {
    0.75 * ema + 0.25 * (accepted as f32)
}

/// Per-sequence γ transition: up to `hi` when the acceptance ratio is
/// strong, down to `lo` when weak, else hold. Returns the next γ and the
/// EMA rescaled to it (`ema · (next − 1) / (cur − 1)`), so the ratio is
/// continuous across a switch — without the rescale a γ=12 → 8 fall at
/// ratio 0.44 would read 0.69 at γ=8 and bounce straight back up. Pure —
/// called once per sequence after `after_verify` ingests that step's
/// accepted count.
pub fn next_gamma(cur: usize, ema: f32, lo: usize, hi: usize) -> (usize, f32) {
    let drafts = cur.saturating_sub(1).max(1) as f32;
    let ratio = ema / drafts;
    let next = if cur < hi && ratio > RATIO_RISE {
        hi
    } else if cur > lo && ratio < RATIO_FALL {
        lo
    } else {
        cur
    };
    let rescaled = ema * (next.saturating_sub(1).max(1) as f32) / drafts;
    (next, rescaled)
}

/// Propose γ at alloc time: adaptive starts at the floor (conservative —
/// prose stays cheap, code climbs within a few steps); fixed mode keeps
/// the configured γ so every call site can read `propose_gamma`
/// unconditionally.
pub(crate) fn initial_gamma(adaptive: bool, configured: usize) -> usize {
    if adaptive && configured > ADAPTIVE_GAMMA_LO {
        ADAPTIVE_GAMMA_LO
    } else {
        configured
    }
}

/// Group key order for the batched split: ascending γ puts the cheap
/// group first. Returns the distinct γ values present in `gammas`.
pub(crate) fn distinct_gammas(gammas: &[usize]) -> Vec<usize> {
    let mut set: Vec<usize> = Vec::with_capacity(2);
    for &g in gammas {
        if !set.contains(&g) {
            set.push(g);
        }
    }
    set.sort_unstable();
    set
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ema_weights_and_ignores_bonus() {
        let e = update_ema(0.0, 8);
        assert_eq!(e, 2.0); // 0.25 * 8
        let e = update_ema(e, 8);
        assert!((e - 3.5).abs() < 1e-6);
        // Bonus token must never be passed in — assert the formula treats
        // `accepted` as draft tokens only by construction (caller contract).
        let e = update_ema(4.0, 0);
        assert!((e - 3.0).abs() < 1e-6);
    }

    #[test]
    fn next_gamma_hysteresis_on_acceptance_ratio() {
        let g = |cur, ema| next_gamma(cur, ema, 8, 12).0;
        // At γ=8 (7 drafts): rise above ratio 0.65 (ema > 4.55).
        assert_eq!(g(8, 5.1), 12); // job-260 code at γ=8: 0.73
        assert_eq!(g(8, 4.5), 8); // 0.643, just inside the band: hold
        assert_eq!(g(8, 3.5), 8); // inside the band
        assert_eq!(g(8, 1.4), 8); // job-260 prose
        // At γ=12 (11 drafts): fall below ratio 0.45 (ema < 4.95).
        assert_eq!(g(12, 7.8), 12); // job-260 code at γ=12: 0.71
        assert_eq!(g(12, 5.0), 12); // 0.4545, just inside the band: hold
        assert_eq!(g(12, 4.0), 8);
        // Off-band values still land in {lo, hi}.
        assert_eq!(g(10, 7.0), 12);
        assert_eq!(g(10, 3.0), 8);
    }

    #[test]
    fn switch_rescales_ema_so_it_does_not_bounce() {
        // Fall 12 -> 8 at ratio 0.44: the rescaled EMA keeps ratio 0.44 at
        // γ=8 (inside the band), so the next step holds at 8.
        let (g, e) = next_gamma(12, 4.84, 8, 12);
        assert_eq!(g, 8);
        assert!((e - 4.84 * 7.0 / 11.0).abs() < 1e-5);
        assert_eq!(next_gamma(8, e, 8, 12).0, 8);
        // Rise 8 -> 12 at ratio 0.70 keeps 0.70 at γ=12: holds at 12.
        let (g, e) = next_gamma(8, 4.9, 8, 12);
        assert_eq!(g, 12);
        assert_eq!(next_gamma(12, e, 8, 12).0, 12);
        // No switch: EMA unchanged.
        assert_eq!(next_gamma(8, 3.0, 8, 12), (8, 3.0));
    }

    #[test]
    fn initial_gamma_arms_at_floor_only_when_adaptive() {
        assert_eq!(initial_gamma(false, 12), 12);
        assert_eq!(initial_gamma(true, 12), 8);
        // γ_max <= floor: adaptive has nowhere to go, keep configured.
        assert_eq!(initial_gamma(true, 8), 8);
        assert_eq!(initial_gamma(true, 4), 4);
        assert_eq!(initial_gamma(false, 16), 16);
    }

    #[test]
    fn distinct_gammas_groups_sorted() {
        assert_eq!(distinct_gammas(&[8, 8, 12]), vec![8, 12]);
        assert_eq!(distinct_gammas(&[12, 8, 12, 8]), vec![8, 12]);
        assert_eq!(distinct_gammas(&[12, 12]), vec![12]);
        assert!(distinct_gammas(&[]).is_empty());
    }
}
