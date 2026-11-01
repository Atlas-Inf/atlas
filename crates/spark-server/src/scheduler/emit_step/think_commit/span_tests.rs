// SPDX-License-Identifier: AGPL-3.0-only

//! A verify span's rows see, row by row, what serial commits leave: the
//! replay ([`SpanShadow`]) after row `i` holds the state `emit_token` holds
//! after committing rows `0..=i` one at a time at their own positions, and
//! the commit path ([`emit_span_row`]) checks each row at that position too.

use super::shadow::ThinkState;
use super::{CommitEnv, SpanShadow, span_row_position};
use crate::scheduler::emit_step::{emit_span_row, emit_token};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;

const END: u32 = 900;
const OPEN: u32 = 600;
const CLOSE: u32 = 601;
/// `<`, `parameter`, `=`, `>`, `</` (`update_tool_param_state`).
const LT: u32 = 27;
const PARAM: u32 = 15704;
const EQ: u32 = 28;
const GT: u32 = 29;
const LT_SLASH: u32 = 510;

/// A content-phase sequence with a required tool call.
fn content_seq(min_tokens: usize) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![5, 6], 64, None, 0);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = min_tokens;
    a.eos_tokens = vec![END];
    a.inside_thinking = false;
    a.think_ended = true;
    a.tool_call_start_token = Some(OPEN);
    a.tool_call_end_token = Some(CLOSE);
    a.require_tool_call = true;
    a
}

/// Below `min_tokens` and with no tool obligation: only the ceiling stops an
/// end token.
fn ceiling_seq() -> ActiveSeq {
    let mut a = content_seq(50);
    a.require_tool_call = false;
    a
}

/// What a row leaves: the commit rule's state, the output, whether it ended.
type Snap = (ThinkState, Vec<u32>, bool);

fn snap(a: &ActiveSeq) -> Snap {
    (
        ThinkState::of(a).picks_only(),
        a.output_tokens.clone(),
        a.finished,
    )
}

/// Serial: commit `toks` one at a time, row `i` at its own position. Stops
/// after the row that finishes the sequence, as every commit path does.
fn serial(mut a: ActiveSeq, toks: &[u32], span_end: usize, s: &SchedCtx) -> Vec<Snap> {
    let mut out = Vec::new();
    for (i, &t) in toks.iter().enumerate() {
        a.seq.seq_len = span_row_position(span_end, toks.len() + 1, i);
        emit_token(&mut a, t, None, s);
        out.push(snap(&a));
        if a.finished {
            break;
        }
    }
    out
}

/// The replay over one span of `toks.len() + 1` rows (the last row, never
/// committed into a later pick, is the bonus).
fn replayed(mut a: ActiveSeq, toks: &[u32], span_end: usize, s: &SchedCtx) -> Vec<Snap> {
    a.seq.seq_len = span_end;
    let start = snap(&a);
    let span = SpanShadow::begin(&mut a, CommitEnv::of(s), toks.len() + 1);
    let mut out = Vec::new();
    for (i, &t) in toks.iter().enumerate() {
        assert!(span.pick(&mut a, t, i), "row {i} refused");
        out.push(snap(&a));
        if a.finished {
            break;
        }
    }
    span.end(&mut a);
    assert_eq!(snap(&a), start, "the span is restored");
    out
}

#[test]
fn a_tool_call_opened_and_closed_inside_one_span_replays_serial_tool_state() {
    // `<tool_call>`, `<parameter=k>` value `</parameter>`, `</tool_call>`, all
    // inside one span: DRY's tool-body zeroing, B1's margin stage and the
    // opener-bias strip read this state at every later row.
    let toks = [
        7, OPEN, LT, PARAM, EQ, 8, GT, 9, 10, LT_SLASH, PARAM, GT, CLOSE, 11,
    ];
    let s = SchedCtx::for_test();
    let want = serial(content_seq(0), &toks, 40, &s);
    let got = replayed(content_seq(0), &toks, 40, &s);
    assert_eq!(got.len(), toks.len());
    assert_eq!(got, want);
    // The span crosses every transition the pick stages read.
    let probe = |i: usize, f: fn(&ActiveSeq) -> bool| {
        let mut a = content_seq(0);
        for &t in &toks[..=i] {
            emit_token(&mut a, t, None, &s);
        }
        f(&a)
    };
    assert!(probe(1, |a| a.inside_tool_body
        && a.tool_call_opened
        && !a.require_tool_call));
    assert!(probe(7, |a| a.inside_parameter_body
        && a.param_body_chars_emitted == 1));
    assert!(probe(11, |a| a.inside_tool_body && !a.inside_parameter_body));
    assert!(probe(12, |a| !a.inside_tool_body && a.tool_call_completed));
}

#[test]
fn an_end_token_near_the_context_ceiling_is_checked_at_its_own_row() {
    // `max_seq_len` 100, a 4-row span ending at 100: rows sit at 97..=100.
    // Row 0's end token is below min_tokens and far enough from the ceiling,
    // so serial decode discards it; checked at the span's end it was a hard
    // stop. Row 2 (position 99) is the ceiling.
    let mut s = SchedCtx::for_test();
    s.limits.max_seq_len = 100;
    let toks = [END, 7, 8];
    let want = serial(ceiling_seq(), &toks, 100, &s);
    let got = replayed(ceiling_seq(), &toks, 100, &s);
    assert_eq!(got, want);
    let ends: Vec<(Vec<u32>, bool)> = want.into_iter().map(|(_, o, f)| (o, f)).collect();
    assert_eq!(
        ends,
        [
            (vec![5, 6], false), // the end token is discarded, not a stop
            (vec![5, 6, 7], false),
            (vec![5, 6, 7, 8], true),
        ]
    );
}

#[test]
fn the_commit_path_checks_each_row_at_its_own_position() {
    // The accepted prefix [END, 7, 8] after the rewind: seq_len 99, rows at
    // 97..=99. Committed at 99 the end token stopped the turn at once.
    let mut s = SchedCtx::for_test();
    s.limits.max_seq_len = 100;
    let toks = [END, 7, 8];
    let mut a = ceiling_seq();
    a.seq.seq_len = 99;
    for (i, &t) in toks.iter().enumerate() {
        emit_span_row(&mut a, t, None, &s, i, toks.len());
        if a.finished {
            break;
        }
    }
    let mut want = ceiling_seq();
    for (i, &t) in toks.iter().enumerate() {
        want.seq.seq_len = 97 + i;
        emit_token(&mut want, t, None, &s);
    }
    assert_eq!(snap(&a), snap(&want));
    assert_eq!(a.output_tokens, [5, 6, 7, 8]);
    assert!(a.finished, "the last row meets the ceiling");
}
