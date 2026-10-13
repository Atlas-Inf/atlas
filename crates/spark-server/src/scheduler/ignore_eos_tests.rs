// SPDX-License-Identifier: AGPL-3.0-only

//! `ignore_eos` (vLLM): the model's end tokens do not end the request.
//! `InferenceRequest::take_end_tokens` leaves them out of the sequence's
//! `eos_tokens`, so serial decode (`process_decode_logits`) and the MTP/verify
//! emission (`emit_token`) both record, count and stream a sampled end token
//! like any other token, and the request runs to `max_tokens` ("length").
//! The request's own `stop` tokens still end it. Without `ignore_eos` an end
//! token below `min_tokens` is discarded (`min_tokens_eos_tests`) — a step
//! that yields no output, which is what made min_tokens benchmarks slower
//! than vLLM's `ignore_eos` runs.

use super::min_tokens_eos_tests::{END, seq, step};
use crate::api::{GrammarSpec, InferenceRequest};
use crate::scheduler::{
    emit_step::{compile_grammar_state, emit_token},
    lifecycle::derive_finish_reason,
    sched_ctx::SchedCtx,
    test_support::{GRAMMAR_EOS, grammar_engine, test_request},
    types::{ActiveSeq, ResponseSink},
};

/// A single-token user `stop` string.
const STOP: u32 = 7;

/// The end tokens a request gets from the model's `[END]`.
fn end_tokens(ignore_eos: bool, stop: &[u32]) -> Vec<u32> {
    let (response_tx, _rx) = tokio::sync::oneshot::channel();
    let mut req = test_request!(Blocking, response_tx,);
    if let InferenceRequest::Blocking {
        ignore_eos: i,
        stop_tokens,
        ..
    } = &mut req
    {
        *i = ignore_eos;
        *stop_tokens = stop.to_vec();
    }
    req.take_end_tokens(&[END])
}

/// `seq` for an `ignore_eos` request.
fn ignoring(output: usize, min_tokens: usize) -> ActiveSeq {
    let mut a = seq(output, min_tokens);
    a.eos_tokens = end_tokens(true, &[]);
    a
}

fn finish_reason(a: &ActiveSeq) -> &'static str {
    derive_finish_reason(
        a.guard_stop,
        a.output_tokens.last().copied(),
        &a.eos_tokens,
        a.tool_call_end_token,
        a.remaining,
        a.seq.seq_len,
        0,
    )
}

#[test]
fn ignore_eos_drops_only_the_model_end_tokens() {
    // Default: byte-identical to the merge the prefill steps used to inline.
    assert_eq!(end_tokens(false, &[]), vec![END]);
    assert_eq!(end_tokens(false, &[STOP]), vec![STOP, END]);
    assert!(end_tokens(true, &[]).is_empty());
    assert_eq!(end_tokens(true, &[STOP]), vec![STOP]);
}

#[test]
fn an_end_token_is_an_ordinary_token_alike_by_serial_and_mtp() {
    let sched = SchedCtx::for_test();
    for (output, floor) in [(3, 5), (5, 5), (0, 0), (3, 0)] {
        for serial in [true, false] {
            let a = step(ignoring(output, floor), serial, &sched);
            let at = format!("serial={serial} at {output}/{floor}");
            assert!(!a.finished, "{at}: an end token ended the request");
            assert_eq!(a.output_tokens.len(), output + 1, "{at}: not counted");
            assert_eq!(a.output_tokens.last(), Some(&END), "{at}");
            assert_eq!(a.remaining, 63, "{at}: no budget drawn");
        }
    }
}

#[test]
fn the_request_runs_to_max_tokens_and_reports_length() {
    let sched = SchedCtx::for_test();
    for serial in [true, false] {
        let mut a = ignoring(3, 0);
        a.remaining = 2;
        a = step(a, serial, &sched);
        assert!(!a.finished, "serial={serial}");
        a = step(a, serial, &sched);
        assert!(a.finished, "serial={serial}: max_tokens reached");
        assert_eq!(a.output_tokens[3..], [END, END], "serial={serial}");
        assert_eq!(finish_reason(&a), "length", "serial={serial}");

        // The same end token ends a default request: "stop".
        let a = step(seq(3, 0), serial, &sched);
        assert!(a.finished);
        assert_eq!(finish_reason(&a), "stop", "serial={serial}");
    }
}

#[test]
fn the_role_boundary_hard_stop_is_an_ordinary_token() {
    // `<|im_start|>` is an end token and the MTP path's ChatML hard stop;
    // under `ignore_eos` it is neither, on both paths.
    let mut sched = SchedCtx::for_test();
    sched.limits.im_start_hard_stop = Some(END);
    for serial in [true, false] {
        let a = step(ignoring(3, 0), serial, &sched);
        assert!(!a.finished, "serial={serial}");
        assert_eq!(a.output_tokens.len(), 4, "serial={serial}");
    }
}

#[test]
fn a_user_stop_token_still_ends_the_request() {
    let sched = SchedCtx::for_test();
    let mut a = ignoring(3, 0);
    a.eos_tokens = end_tokens(true, &[STOP]);
    emit_token(&mut a, END, None, &sched);
    assert!(!a.finished);
    emit_token(&mut a, STOP, None, &sched);
    assert!(a.finished);
    assert_eq!(finish_reason(&a), "stop");
}

#[test]
fn a_grammar_passes_the_end_token_through() {
    // Prefill still exempts the model's EOS from the matcher: a finished JSON
    // answer decodes end tokens to max_tokens on the MTP path too, instead of
    // terminating the matcher or failing the strict grammar.
    let sched = SchedCtx::for_test();
    let mut a = ignoring(0, 0);
    a.grammar_state = compile_grammar_state(
        &mut Some(grammar_engine()),
        &Some(GrammarSpec::JsonObject),
        &[GRAMMAR_EOS],
        false,
        &mut ResponseSink::Blocking(None),
    )
    .unwrap();
    for t in [b'{', b'}'] {
        emit_token(&mut a, t as u32, None, &sched);
    }
    for _ in 0..2 {
        emit_token(&mut a, GRAMMAR_EOS, None, &sched);
    }
    assert!(!a.finished && a.engine_error.is_none());
    assert!(a.grammar_state.as_ref().is_some_and(|g| !g.is_terminated()));
    assert_eq!(
        a.output_tokens,
        [b'{' as u32, b'}' as u32, GRAMMAR_EOS, GRAMMAR_EOS]
    );
}
