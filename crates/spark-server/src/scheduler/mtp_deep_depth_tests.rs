// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn window(d: &mut DeepDepth, drafts: usize, hit_every: usize, floor: usize, max: usize) {
    for i in 0..WINDOW as usize {
        let accepted = if i % hit_every == 0 { drafts } else { 0 };
        d.record(drafts, accepted, floor, max);
    }
}

#[test]
fn a_request_starts_at_the_floor() {
    let d = DeepDepth::default();
    assert_eq!(d.drafts(3, 7), 3);
    assert_eq!(d.drafts(3, 2), 2, "the floor never exceeds the ceiling");
}

#[test]
fn high_deepest_acceptance_promotes_by_two_then_one() {
    let mut d = DeepDepth::default();
    window(&mut d, 3, 1, 3, 7); // a = 1.0
    assert_eq!(d.drafts(3, 7), 5);
    // a = 1/2 = 0.5: one step.
    window(&mut d, 5, 2, 3, 7);
    assert_eq!(d.drafts(3, 7), 6);
    window(&mut d, 6, 1, 3, 7);
    assert_eq!(d.drafts(3, 7), 7, "clamped at --num-drafts");
    assert_eq!(d.switches, 3);
}

#[test]
fn low_deepest_acceptance_demotes_to_the_floor() {
    let mut d = DeepDepth::default();
    window(&mut d, 3, 1, 3, 7);
    window(&mut d, 5, 1, 3, 7);
    assert_eq!(d.drafts(3, 7), 7);
    // a = 1/5 = 0.2: one step down.
    window(&mut d, 7, 5, 3, 7);
    assert_eq!(d.drafts(3, 7), 6);
    // a = 1/8 = 0.125: two steps down.
    window(&mut d, 6, 8, 3, 7);
    assert_eq!(d.drafts(3, 7), 4);
    window(&mut d, 4, 48, 3, 7);
    assert_eq!(d.drafts(3, 7), 3, "never below the floor");
}

#[test]
fn the_middle_band_holds() {
    let mut d = DeepDepth::default();
    window(&mut d, 3, 1, 3, 7);
    let before = d.drafts(3, 7);
    // a = 1/3: inside [0.25, 0.45].
    window(&mut d, before, 3, 3, 7);
    assert_eq!(d.drafts(3, 7), before);
}

#[test]
fn verifies_that_never_drafted_the_deepest_position_do_not_count() {
    // The confidence stop cut every chain at 2 drafts: position 3 is never
    // observed, so the ceiling holds instead of reading 0% acceptance.
    let mut d = DeepDepth::default();
    for _ in 0..4 {
        window(&mut d, 2, 1, 3, 7);
    }
    assert_eq!(d.drafts(3, 7), 3);
    assert_eq!(d.switches, 0);
}

#[test]
fn row_budget_trims_the_deepest_first() {
    let mut caps = vec![7, 7, 3, 0];
    fit_row_budget(&mut caps, 16);
    assert_eq!(caps.iter().map(|d| d + 1).sum::<usize>(), 16);
    assert_eq!(caps, vec![4, 5, 3, 0]);

    let mut caps = vec![7, 7];
    fit_row_budget(&mut caps, 3);
    assert_eq!(caps, vec![1, 1], "never below one draft");

    let mut caps = vec![4, 4, 4];
    fit_row_budget(&mut caps, 14);
    assert_eq!(caps, vec![3, 4, 4], "ties: the lowest index gives first");
}

#[test]
fn position_acceptance_divides_by_the_verifies_that_drafted_it() {
    let mut p = PositionAccept::default();
    p.record(3, 3);
    p.record(3, 1);
    p.record(2, 2); // confidence stop: position 3 not drafted
    p.record(1, 0);
    assert_eq!(p.suffix(), "0.75,0.67,0.50,-,-,-,-");
}
