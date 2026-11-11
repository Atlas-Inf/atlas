// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn union_keeps_primary_order_and_drops_duplicates() {
    assert_eq!(
        union_layers(&[5, 19, 33, 47, 61], &[1, 16, 31, 46, 61]),
        vec![5, 19, 33, 47, 61, 1, 16, 31, 46]
    );
    assert_eq!(union_layers(&[1, 2], &[1, 2]), vec![1, 2]);
}

/// Feed `rounds` rounds of `tokens(head, round)` on whatever head `r` picks.
fn run(r: &mut Route, rounds: usize, tokens: impl Fn(usize, usize) -> usize) -> Vec<usize> {
    (0..rounds)
        .map(|i| {
            let h = r.pick();
            assert_eq!(r.pick(), h, "pick is idempotent until record");
            r.last = h;
            r.record(tokens(h, i));
            h
        })
        .collect()
}

/// Primary = DFlash2 (8-row blocks), alt = DSpine (16).
fn pair() -> Route {
    Route::new([8, 16])
}

#[test]
fn prose_never_leaves_the_short_head() {
    let mut r = pair();
    // An occasional full accept is not enough.
    let heads = run(&mut r, 200, |_, i| if i % 5 == 0 { 8 } else { 2 });
    assert!(heads.iter().all(|&h| h == 0), "{heads:?}");
}

#[test]
fn code_moves_to_the_long_head_and_stays() {
    let mut r = pair();
    let heads = run(&mut r, 100, |h, i| match (h, i % 3) {
        (0, 0) => 4,
        (0, _) => 8,
        (_, 0) => 3,
        _ => 13,
    });
    let first = heads.iter().position(|&h| h == 1).expect("switches");
    assert!(first <= 6, "{heads:?}");
    assert!(heads[first..].iter().all(|&h| h == 1), "{heads:?}");
}

#[test]
fn the_long_head_hands_back_when_it_stops_going_past_the_cap() {
    let mut r = pair();
    run(&mut r, 20, |h, _| if h == 0 { 8 } else { 12 });
    assert_eq!(r.pick(), 1);
    let heads = run(&mut r, 12, |h, _| if h == 0 { 3 } else { 5 });
    assert_eq!(*heads.last().unwrap(), 0, "{heads:?}");
    assert!(heads[..JUDGE].iter().all(|&h| h == 1), "{heads:?}");
}

#[test]
fn equal_blocks_never_switch() {
    let mut r = Route::new([8, 8]);
    assert!(run(&mut r, 50, |_, _| 8).iter().all(|&h| h == 0));
}
