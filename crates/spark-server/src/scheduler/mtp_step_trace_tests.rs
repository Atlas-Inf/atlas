// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn rec_with(seqs: &[(usize, u8, usize)]) -> Rec {
    let mut r = Rec {
        stepped: true,
        nd: 3,
        ..Rec::default()
    };
    for &(slot, path, drafts) in seqs {
        r.seq(slot, path, drafts);
    }
    r
}

// The line `scripts/dev/step_trace_summary.py` parses: one entry per
// sequence that took part, emitted tokens from the seq_len delta, the
// sequence's tokens generated before the step, and the path it took.
#[test]
fn line_carries_every_participant() {
    let mut rec = rec_with(&[(3, b'v', 3), (5, b'd', 0)]);
    rec.forward(5);
    let before = vec![(3, 100, 7), (5, 40, 1), (9, 10, 2)];
    // Slot 9 was preempted mid-step: it is not in `after` and is dropped.
    let after = vec![(5, 41), (3, 103)];
    let line = format_line(1234, 567, &rec, &before, &after).expect("a step ran");
    assert_eq!(
        line,
        "MTP STEP t=1234 dur=567 kind=mtp n=2 rows=5 fwd=1 nd=3 deep=0 dl=1 \
         s=3:7:v3:3,5:1:d0:1"
    );
}

// A plain decode step records no shape; its sequences read as `n` rows.
#[test]
fn plain_decode_rows_default_to_serial() {
    let rec = Rec::default();
    let before = vec![(0, 10, 4)];
    let after = vec![(0, 11)];
    let line = format_line(1, 2, &rec, &before, &after).unwrap();
    assert!(line.contains("kind=dec"), "{line}");
    assert!(line.ends_with("s=0:4:n0:1"), "{line}");
    assert!(line.contains("dl=0"), "{line}");
}

// A tick where nothing decoded (prefill only) logs nothing.
#[test]
fn idle_tick_is_silent() {
    let rec = Rec::default();
    let before = vec![(0, 10, 4)];
    let after = vec![(0, 10)];
    assert!(format_line(1, 2, &rec, &before, &after).is_none());
}

// The first record of a slot wins: a later fallback note must not relabel
// the path that actually carried the sequence.
#[test]
fn first_path_record_wins() {
    let mut rec = rec_with(&[(2, b'b', 0)]);
    rec.seq(2, b's', 3);
    assert_eq!(rec.path_of(2), Some((b'b', 0)));
    // Bootstraps count as draftless rows.
    let line = format_line(0, 0, &rec, &[(2, 5, 0)], &[(2, 6)]).unwrap();
    assert!(line.contains("dl=1"), "{line}");
}

#[test]
fn kind_override_and_rode_rows() {
    let mut rec = Rec {
        kind: Some("mixed"),
        ..Rec::default()
    };
    rec.seq(4, b'r', 0);
    let line = format_line(0, 0, &rec, &[(4, 9, 3), (6, 9, 3)], &[(4, 12), (6, 10)]).unwrap();
    assert!(line.contains("kind=mixed"), "{line}");
    assert!(line.ends_with("s=4:3:r0:3,6:3:m0:1"), "{line}");
}
