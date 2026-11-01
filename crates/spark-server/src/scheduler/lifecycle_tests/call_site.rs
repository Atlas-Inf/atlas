// SPDX-License-Identifier: AGPL-3.0-only

//! `finish_sequence` call-site proofs. Split from `lifecycle_tests.rs` (500-LoC cap).

use super::*;

#[test]
fn call_site_passes_the_real_budget() {
    // Budget exhausted ⇒ "length". Red if finish_sequence stops passing
    // `a.remaining` (e.g. a constant, or the decoy `min_tokens` = 7).
    let (a, rx) = test_seq(vec![5, 6, 42], 0, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
    // Budget left ⇒ NOT "length" (same decoy: min_tokens=7 ≠ remaining).
    let (a, rx) = test_seq(vec![5, 6, 42], 7, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "stop");
}

#[test]
fn call_site_passes_the_real_seq_len() {
    // Context-ceiling stop with budget left ⇒ "length". Red if the
    // call site stops passing `a.seq.seq_len` / the served ceiling.
    let (a, rx) = test_seq(vec![5, 6, 42], 500, None, MAX_SEQ_LEN - 1);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
}

#[test]
fn call_site_passes_the_real_last_token_and_eos() {
    // Red if the call site stops passing `output_tokens.last()` or
    // `a.eos_tokens`.
    let (a, rx) = test_seq(vec![5, 6, 151645], 0, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "stop");
    let (a, rx) = test_seq(vec![5, 6, 151658], 3, None, 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "tool_calls");
}

#[test]
fn call_site_passes_the_real_guard() {
    // Timeout is the one guard with a distinct wire reason — proves
    // `a.guard_stop` reaches the decision.
    let (a, rx) = test_seq(vec![5, 6, 42], 3, Some(GUARD_STOP_REQUEST_TIMEOUT), 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, FINISH_REASON_TIMEOUT);
    // And a degeneration guard reaches the response as "length" — the
    // truncation signal. Asserted at the CALL SITE, not just over the pure
    // function, because that is where the wire value the client actually
    // receives is decided.
    let (a, rx) = test_seq(vec![5, 6, 42], 3, Some("fuzzy_repetition"), 10);
    assert_eq!(finish_and_recv(a, rx).finish_reason, "length");
}

#[test]
fn terminal_model_error_is_sent_as_error_without_successful_done() {
    let (mut a, mut rx) = test_seq(vec![5, 6, 42], 7, None, 10);
    a.engine_error = Some("MTP proposal failed: indexed context envelope".into());
    finish_sequence(&StubModel::default(), &mut a, MAX_SEQ_LEN);
    let response = rx
        .try_recv()
        .expect("terminal model error must reach the blocking client");
    let error = match response {
        Ok(_) => panic!("a terminal model error became a successful response"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("indexed context envelope"));
}
