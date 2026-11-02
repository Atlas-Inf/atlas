// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn curve(drafted: usize, accepted: usize, steps: usize) -> DraftSurvival {
    let mut s = DraftSurvival::default();
    for _ in 0..steps {
        s.record(drafted, accepted);
    }
    s
}

/// Hazard of a position with no evidence: the prior plus the full bonus.
fn unobserved() -> f32 {
    (DEFAULT_PARAMS.prior_h + DEFAULT_PARAMS.ucb).min(1.0)
}

#[test]
fn record_observes_only_reached_positions() {
    let s = curve(7, 2, 60);
    // Positions 0 and 1 always accepted, 2 always rejected, 3.. never reached.
    assert!(s.hazard(0) > 0.95 && s.hazard(1) > 0.95);
    assert!(s.hazard(2) < 0.2);
    assert!((s.hazard(3) - unobserved()).abs() < 1e-6);
}

#[test]
fn unobserved_positions_drift_back_to_the_prior() {
    let mut s = curve(7, 0, 60);
    assert!(s.hazard(0) < 0.2);
    for _ in 0..600 {
        s.record(0, 0);
    }
    assert!((s.hazard(0) - unobserved()).abs() < 0.02);
}

#[test]
fn predictable_text_verifies_wide_and_prose_narrow() {
    let counting = curve(7, 7, 60);
    let prose = curve(7, 1, 60);
    assert_eq!(best([&counting].into_iter(), 7), 7);
    assert!(best([&prose].into_iter(), 7) <= 3);
    // The batch shares one width: four prose owners narrow it at least as
    // far as one, since every extra row costs more with more owners.
    let four = [&prose, &prose, &prose, &prose];
    assert!(best(four.into_iter(), 7) <= best([&prose].into_iter(), 7));
}

#[test]
fn width_respects_the_cap() {
    let counting = curve(7, 7, 60);
    assert_eq!(best([&counting].into_iter(), 3), 3);
}

#[test]
fn a_full_accept_reopens_the_width() {
    // Bursty text: rejections at the fourth draft narrowed the width to
    // three, then a verify of three drafts accepted all three.
    let mut s = curve(7, 3, 20);
    assert!(best([&s].into_iter(), 7) <= 3);
    s.record(3, 3);
    assert!(best([&s].into_iter(), 7) > 3);
}

#[test]
fn eight_owners_stay_within_the_row_budget() {
    let counting = curve(7, 7, 60);
    let eight = [&counting; 8];
    // 8 owners x 4 rows is the widest batch the 32-row budget admits.
    assert_eq!(best(eight.into_iter(), 7), 3);
}

#[test]
fn width_params_parse_with_an_optional_burst_floor() {
    assert_eq!(parse_params(None), DEFAULT_PARAMS);
    assert_eq!(parse_params(Some("garbage")), DEFAULT_PARAMS);
    // Four values keep the production burst floor.
    let four = parse_params(Some("0.98, 0.6, 3, 0.2"));
    assert_eq!((four.prior_h, four.burst_h), (0.6, DEFAULT_PARAMS.burst_h));
    // A fifth sets it; 0 turns it off.
    let five = parse_params(Some("0.98,0.6,3,0.2,0"));
    assert_eq!((five.prior_h, five.burst_h), (0.6, 0.0));
}
