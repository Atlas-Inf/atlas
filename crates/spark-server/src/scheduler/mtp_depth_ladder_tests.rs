// SPDX-License-Identifier: AGPL-3.0-only

use std::time::{Duration, Instant};

use super::*;

/// Run `n` verifies through `l`, each `wall(d)` ms after the previous one,
/// accepting `acc(i, d)` of the `d` drafts the ladder asks for.
fn drive(
    l: &mut DepthLadder,
    t: &mut Instant,
    n: usize,
    wall: impl Fn(usize) -> f32,
    acc: impl Fn(usize, usize) -> usize,
) {
    for i in 0..n {
        let d = l.drafts(MAX_DEPTH);
        *t += Duration::from_secs_f32(wall(d) / 1e3);
        l.record(d, acc(i, d), MAX_DEPTH, *t);
    }
}

#[test]
fn starts_at_the_ceiling() {
    let l = DepthLadder::default();
    assert_eq!(l.drafts(3), 3);
    assert_eq!(l.drafts(2), 2);
}

#[test]
fn high_acceptance_with_cheap_rows_stays_deep() {
    let mut l = DepthLadder::default();
    let mut t = Instant::now();
    // ~0.9 survival per position, +10% wall per row: depth 3 wins.
    drive(
        &mut l,
        &mut t,
        64,
        |d| 40.0 * (1.0 + 0.1 * d as f32),
        |i, d| {
            if i % 10 == 0 { 0 } else { d }
        },
    );
    assert_eq!(l.drafts(3), 3);
    assert_eq!(l.switches, 0);
}

#[test]
fn low_acceptance_with_costly_rows_steps_down_to_one() {
    let mut l = DepthLadder::default();
    let mut t = Instant::now();
    // Position 1 accepts half the time, later positions almost never; each
    // extra row costs 30% of a step.
    drive(
        &mut l,
        &mut t,
        48,
        |d| 40.0 * (1.0 + 0.3 * d as f32),
        |i, _| usize::from(i % 2 == 0),
    );
    assert_eq!(l.drafts(3), 1);
    assert!(l.switches >= 1);
}

#[test]
fn a_shallow_ladder_re_probes_the_ceiling() {
    let mut l = DepthLadder::default();
    let mut t = Instant::now();
    drive(
        &mut l,
        &mut t,
        32,
        |d| 40.0 * (1.0 + 0.3 * d as f32),
        |i, _| usize::from(i % 2 == 0),
    );
    assert_eq!(l.drafts(3), 1);
    // Stay shallow until the probe interval elapses, then go deep once.
    let mut probed = false;
    for _ in 0..(REPROBE / WINDOW + 1) {
        drive(
            &mut l,
            &mut t,
            WINDOW as usize,
            |_| 52.0,
            |i, _| usize::from(i % 2 == 0),
        );
        probed |= l.drafts(3) == 3;
    }
    assert!(probed, "no deep probe within {REPROBE} verifies");
}

#[test]
fn saturated_shallow_depth_promotes_early() {
    let mut l = DepthLadder::default();
    let mut t = Instant::now();
    drive(
        &mut l,
        &mut t,
        32,
        |d| 40.0 * (1.0 + 0.3 * d as f32),
        |i, _| usize::from(i % 2 == 0),
    );
    assert_eq!(l.drafts(3), 1);
    // Position 1 now always accepts: the next decision probes deep.
    drive(&mut l, &mut t, WINDOW as usize, |_| 52.0, |_, d| d);
    assert_eq!(l.drafts(3), 3);
}

#[test]
fn stalls_are_not_wall_samples() {
    let mut l = DepthLadder::default();
    let t = Instant::now();
    l.record(3, 3, 3, t);
    l.record(3, 3, 3, t + Duration::from_secs(5));
    assert_eq!(l.wall_ms[3], 0.0);
    l.note_serial();
    l.record(3, 3, 3, t + Duration::from_secs(6));
    assert_eq!(l.wall_ms[3], 0.0);
    l.record(3, 3, 3, t + Duration::from_millis(6050));
    assert!((l.wall_ms[3] - 50.0).abs() < 0.5);
}
