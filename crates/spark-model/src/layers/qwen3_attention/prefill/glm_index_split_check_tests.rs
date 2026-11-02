// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for the `glm_index_split` check: what it swaps with the peer
//! and how it names the cause of a difference.

use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;

use super::super::tests::{Pair, split, with_ctx};
use super::*;

/// The owner under test on rank 1: 258 rows of 16 bytes, so rank 1 selects
/// rows 64..192 and 256..258 and receives 0..64 and 192..256.
const ROWS: usize = 258;
const RB: usize = 16;
const CLEAN: Verdict = Verdict {
    first: None,
    own: 0,
    received: 0,
};

fn bytes(words: &[u64]) -> Vec<u8> {
    words.iter().flat_map(|w| w.to_le_bytes()).collect()
}

/// `ROWS` rows whose bytes all differ from their neighbours', with `rows`
/// changed.
fn rows_with(changed: &[usize]) -> Vec<u8> {
    let mut rows: Vec<u8> = (0..ROWS * RB).map(|i| (i % 251) as u8).collect();
    for &row in changed {
        rows[row * RB + 3] ^= 0x40;
    }
    rows
}

fn evidence(split: &[u8], replicated: &[u8], query: &[u8]) -> Evidence {
    Evidence::of(split, replicated, [query, b"weights"], ROWS, RB)
}

/// Run the check on rank 1 holding `held` split rows and `replicated` rows,
/// with the peer answering `peer` and then `theirs`. Returns the result and
/// the payloads this rank sent after the two quarter swaps.
fn check(
    held: &[u8],
    replicated: &[u8],
    peer: Verdict,
    theirs: Evidence,
) -> (Result<()>, Vec<Vec<u8>>) {
    let gpu = MockGpuBackend::new();
    let pair = Pair::new(&gpu, 1);
    // The two quarter swaps land nothing new.
    pair.peer.lock().unwrap().extend([
        vec![],
        vec![],
        bytes(&peer.words()),
        bytes(&theirs.words()),
    ]);
    let result = with_ctx(&gpu, Some(&pair), false, 2, |ctx| {
        let selected = ctx.buffers.expert_down_out();
        let scratch = selected.offset(ROWS * RB);
        let (query, weights) = (ctx.buffers.ssm_deinterleaved(), ctx.buffers.ssm_gates());
        gpu.copy_h2d(held, selected).unwrap();
        gpu.copy_h2d(replicated, scratch).unwrap();
        gpu.copy_h2d(b"queries", query).unwrap();
        gpu.copy_h2d(b"weights", weights).unwrap();
        let owner = OwnerRows {
            selected,
            row_bytes: RB,
            scratch,
            inputs: [(query, 7), (weights, 7)],
        };
        split(ROWS, 1, true).exchange(&owner, 3, ctx, 7)
    });
    let sent = pair.calls().into_iter().skip(2);
    let sent = sent.map(|(send, dst, add, sent)| {
        assert_eq!((dst - send, add), (sent.len() as u64, false));
        sent
    });
    (result, sent.collect())
}

fn message(result: Result<()>) -> String {
    format!("{:#}", result.unwrap_err())
}

#[test]
fn a_clean_check_swaps_one_verdict_and_no_evidence() {
    let rows = rows_with(&[]);
    let (ok, sent) = check(&rows, &rows, CLEAN, evidence(&rows, &rows, b"queries"));
    ok.unwrap();
    assert_eq!(sent, [bytes(&[0, 0, 0])]);
}

#[test]
fn rows_changed_in_the_exchange_are_a_transport_fault() {
    // Rows 5..=7 and 200 arrived unlike the rows the peer selected and sent.
    let (sent_rows, held) = (rows_with(&[]), rows_with(&[5, 6, 7, 200]));
    let theirs = evidence(&sent_rows, &sent_rows, b"queries");
    let (err, sent) = check(&held, &sent_rows, CLEAN, theirs);
    let msg = message(err);
    assert!(
        msg.contains("differs from the replicated selection"),
        "{msg}"
    );
    assert!(
        msg.contains("(first differing row: here Some(5), peer None)"),
        "{msg}"
    );
    assert!(msg.contains("cause=transport;"), "{msg}");
    assert!(
        msg.contains(
            "here 0 own + 4 received rows differ [5..=7 200] from byte Q0+83, 4 after a re-read"
        ),
        "{msg}"
    );
    assert!(msg.contains("peer 0 own + 0 received"), "{msg}");
    assert!(
        msg.contains(
            r#"exchanged rows differ in ["Q0", "Q3"], replicated rows in [], queries same, weights same"#
        ),
        "{msg}"
    );
    // The verdict (first row + 1, own, received), then this rank's evidence.
    let mine = evidence(&held, &sent_rows, b"queries");
    assert_eq!(sent, [bytes(&[6, 0, 4]), bytes(&mine.words())]);
}

#[test]
fn ranks_selecting_differently_from_their_own_inputs_are_an_inputs_fault() {
    // The peer selected row 200 unlike this rank's replicated row, from
    // queries unlike this rank's; the exchange delivered it faithfully.
    let (mine_rows, peer_rows) = (rows_with(&[]), rows_with(&[200]));
    let theirs = evidence(&peer_rows, &peer_rows, b"QUERIES");
    let msg = message(check(&peer_rows, &mine_rows, CLEAN, theirs).0);
    assert!(msg.contains("cause=inputs;"), "{msg}");
    assert!(
        msg.contains("here 0 own + 1 received rows differ [200] from byte Q3+131,"),
        "{msg}"
    );
    assert!(
        msg.contains(
            r#"exchanged rows differ in [], replicated rows in ["Q3"], queries differ, weights same"#
        ),
        "{msg}"
    );
}

#[test]
fn own_rows_unlike_the_full_pass_are_a_launch_fault() {
    // Rank 1 selected row 100 and the tail row 257 itself.
    let (full, held) = (rows_with(&[]), rows_with(&[100, 257]));
    let theirs = evidence(&held, &full, b"queries");
    let (err, sent) = check(&held, &full, CLEAN, theirs);
    let msg = message(err);
    assert!(msg.contains("cause=launch;"), "{msg}");
    assert!(
        msg.contains("here 2 own + 0 received rows differ [100 257] from byte Q1+579,"),
        "{msg}"
    );
    assert_eq!(sent[0], bytes(&[101, 2, 0]));
}

#[test]
fn a_difference_only_the_peer_saw_fails_here_too() {
    let rows = rows_with(&[]);
    let peer = Verdict {
        first: Some(71),
        own: 0,
        received: 2,
    };
    // The peer holds rows unlike the ones this rank selected and sent it.
    let theirs = evidence(&rows_with(&[71, 72]), &rows, b"queries");
    let (err, sent) = check(&rows, &rows, peer, theirs);
    let msg = message(err);
    assert!(
        msg.contains("(first differing row: here None, peer Some(71))"),
        "{msg}"
    );
    assert!(msg.contains("cause=transport;"), "{msg}");
    assert!(msg.contains("peer 0 own + 2 received"), "{msg}");
    assert!(msg.contains(r#"exchanged rows differ in ["Q1"]"#), "{msg}");
    assert_eq!(sent.len(), 2);
    assert_eq!(sent[0], bytes(&[0, 0, 0]));
}

#[test]
fn causes_combine_and_a_difference_nothing_explains_says_so() {
    let rows = rows_with(&[]);
    let mine = evidence(&rows, &rows, b"queries");
    let saw = |own, received| Verdict {
        first: Some(1),
        own,
        received,
    };
    let cause = |here, peer, theirs: &Evidence| causes(here, peer, &mine, theirs).0;
    assert_eq!(cause(saw(0, 1), CLEAN, &mine), "unexplained");
    assert_eq!(cause(CLEAN, saw(3, 0), &mine), "launch");
    let other = rows_with(&[70, 257]);
    let theirs = evidence(&other, &other, b"queries");
    assert_eq!(cause(saw(1, 1), CLEAN, &theirs), "launch+transport+inputs");
    let (_, [exchanged, replicated]) = causes(CLEAN, CLEAN, &mine, &theirs);
    // The rows past the quarters are never exchanged.
    assert_eq!((exchanged, replicated), (vec!["Q1"], vec!["Q1", "tail"]));
}

#[test]
fn verdicts_count_own_and_received_rows() {
    let s = split(ROWS, 1, true);
    assert_eq!(Verdict::of(&[], &s), CLEAN);
    let v = Verdict::of(&[0, 63, 64, 191, 192, 255, 256, 257], &s);
    let want = Verdict {
        first: Some(0),
        own: 4,
        received: 4,
    };
    assert_eq!(v, want);
    assert_eq!(v.words(), [1, 4, 4]);
    assert_eq!(Verdict::from_words(&v.words()), v);
    assert_eq!(Verdict::from_words(&[0, 0, 0]), CLEAN);
    // Rank 0 receives the other two quarters.
    let v = Verdict::of(&[0, 64, 128, 192, 256], &split(ROWS, 0, true));
    assert_eq!((v.own, v.received), (3, 2));
}

#[test]
fn evidence_fingerprints_each_region_and_round_trips() {
    let rows = rows_with(&[]);
    let base = evidence(&rows, &rows, b"queries");
    assert_eq!(Evidence::from_words(&base.words()), base);
    assert_eq!(base.words().len(), Evidence::WORDS);
    // One changed row moves exactly its region's fingerprint.
    for (row, region) in [(0, 0), (63, 0), (64, 1), (191, 2), (255, 3), (256, 4)] {
        let e = evidence(&rows, &rows_with(&[row]), b"queries");
        let moved: Vec<_> = (0..5)
            .filter(|&k| e.replicated[k] != base.replicated[k])
            .collect();
        assert_eq!(moved, [region], "row {row}");
        assert_eq!((e.split, e.inputs), (base.split, base.inputs));
    }
    let e = evidence(&rows_with(&[200]), &rows, b"queriez");
    assert_eq!(e.replicated, base.replicated);
    assert_ne!(e.split[3], base.split[3]);
    assert_eq!(e.split[..3], base.split[..3]);
    assert_ne!(e.inputs[0], base.inputs[0]);
    assert_eq!(e.inputs[1], base.inputs[1]);
}

#[test]
fn fingerprints_see_every_byte_and_the_length() {
    let data: Vec<u8> = (0..37).collect();
    let base = fingerprint(&data);
    for i in 0..data.len() {
        for bit in [1u8, 0x80] {
            let mut d = data.clone();
            d[i] ^= bit;
            assert_ne!(fingerprint(&d), base, "byte {i} bit {bit}");
        }
    }
    assert_ne!(fingerprint(&data[..36]), base);
    assert_ne!(fingerprint(&[0u8; 8]), fingerprint(&[0u8; 16]));
    assert_ne!(fingerprint(&[]), fingerprint(&[0]));
    // Rows swapped with each other are not the same buffer.
    let (a, b) = ([1u8; 8], [2u8; 8]);
    assert_ne!(fingerprint(&[a, b].concat()), fingerprint(&[b, a].concat()));
}

#[test]
fn differing_rows_and_their_runs() {
    let a = [1u8, 2, 3, 4, 5, 6];
    assert_eq!(differing_rows(&a, &a, 2), Vec::<usize>::new());
    assert_eq!(differing_rows(&a, &[1, 2, 3, 4, 5, 7], 2), [2]);
    assert_eq!(differing_rows(&a, &[1, 2, 0, 4, 0, 6], 2), [1, 2]);
    assert_eq!(runs(&[], 8), "");
    assert_eq!(runs(&[7], 8), "7");
    assert_eq!(
        runs(&[151, 152, 153, 300, 302, 303], 8),
        "151..=153 300 302..=303"
    );
    assert_eq!(runs(&[1, 3, 5, 6, 9], 2), "1 3 (+2 runs)");
}

#[test]
fn the_first_differing_byte_is_named_by_region_and_offset() {
    let rows = rows_with(&[]);
    let at = |changed: &[usize]| first_byte(&rows_with(changed), &rows, ROWS, RB);
    assert_eq!(at(&[]), "none");
    // A changed row differs at its byte 3; quarters are 64 rows of 16 bytes.
    assert_eq!(at(&[0]), "Q0+3");
    assert_eq!(at(&[63, 64]), "Q0+1011");
    assert_eq!(at(&[64, 200]), "Q1+3");
    assert_eq!(at(&[255]), "Q3+1011");
    assert_eq!(at(&[257]), "tail+19");
    // Fewer than four rows: every byte is past the (empty) quarters.
    assert_eq!(first_byte(&[1, 2, 3], &[1, 2, 4], 3, 1), "tail+2");
}
