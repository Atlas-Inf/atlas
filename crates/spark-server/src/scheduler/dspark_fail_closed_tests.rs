// SPDX-License-Identifier: AGPL-3.0-only

//! Lightning DSpark fail-closed regression tests. Split out of
//! `lifecycle_tests.rs` for the file-size cap. The `StubModel` used here
//! lives in `lifecycle_tests.rs` and carries the REAL Model identity hook
//! (an admitted product policy via OnceLock).

use super::lifecycle_tests::{MAX_SEQ_LEN, StubModel, derive, finish_and_recv, test_seq};

#[test]
fn dspark_product_fail_closed_reports_server_truncation_not_stop() {
    // The Lightning product fail-closed guard must surface as the
    // server-truncation family ("length"), never a natural "stop": a
    // client seeing "stop" accepts the partial answer as complete.
    // Last token 42 is a normal (non-EOS) token; budget remains.
    assert_eq!(
        derive(
            Some(crate::scheduler::types::GUARD_STOP_DSPARK_PRODUCT_FAIL_CLOSED),
            Some(42),
            100
        ),
        "length"
    );
    // Same when the budget also ran out on that step (deterministic
    // precedence: both paths agree it is a truncation).
    assert_eq!(
        derive(
            Some(crate::scheduler::types::GUARD_STOP_DSPARK_PRODUCT_FAIL_CLOSED),
            Some(42),
            0
        ),
        "length"
    );
    // An EOS sampled on the failing step would still report "stop"
    // (token-derived natural stops outrank guards) — that is the
    // documented precedence and acceptable here because the model did
    // emit a terminal token.
}

// Drives the REAL production fail-closed action
// (fail_dspark_product_sequence_closed — the single function every
// scheduler proposal-failure call site routes through) over the REAL
// Model identity hook, using this module's StubModel.

#[test]
fn fail_dspark_product_sequence_closed_marks_guard_and_finishes_for_product() {
    let mut guard: Option<&'static str> = None;
    let mut finished = false;
    let acted = crate::scheduler::helpers::fail_dspark_product_sequence_closed(
        &StubModel { product: true },
        7,
        &mut guard,
        &mut finished,
        "test-site",
    );
    assert!(acted, "product serve must fail closed");
    assert!(finished, "sequence must be finished");
    assert_eq!(
        guard,
        Some(crate::scheduler::types::GUARD_STOP_DSPARK_PRODUCT_FAIL_CLOSED),
        "the client-visible truncation guard must be set"
    );
}

#[test]
fn fail_dspark_product_sequence_closed_leaves_generic_untouched() {
    let mut guard: Option<&'static str> = None;
    let mut finished = false;
    let acted = crate::scheduler::helpers::fail_dspark_product_sequence_closed(
        &StubModel::default(),
        7,
        &mut guard,
        &mut finished,
        "test-site",
    );
    assert!(!acted, "generic DFlash/MTP keeps legacy behavior");
    assert!(!finished);
    assert!(guard.is_none());
}
#[test]
fn guard_cuts_report_length_because_the_model_did_not_finish() {
    // POSITIVE case. A guard cut is a server-side truncation: the model
    // was still mid-output. `"length"` is the OpenAI-spec slot for
    // "forcibly truncated" and is what every client's truncation handling
    // keys on (openai-python `LengthFinishReasonError`, aider's
    // continuation, Instructor, pydantic-ai).
    //
    // ★ This assertion was briefly INVERTED to `"stop"`, and that shipped
    // a measured regression: the agentic gate fell to 8/10 then 4/10
    // followed_directions because its `was_cut_off()` stopped firing and
    // runs ended at 3-10 turns instead of the 12-22 a recovery needs.
    // `"stop"` claims the model finished; for a mid-sentence repetition
    // cut that is false, and every client action keyed on it (accept,
    // validate, commit, end the run) is then wrong. Do not re-invert.
    for guard in [
        "fuzzy_repetition",
        "inter_tool_prose_budget",
        "tool_envelope_stuck",
        "simhash_semantic_loop",
        "token_loop_watchdog",
    ] {
        assert_eq!(
            derive(Some(guard), Some(42), 100),
            "length",
            "guard={guard}"
        );
        // A guard trip on the exact step the budget ran out is still a
        // truncation, and both paths agree — precedence is deterministic.
        assert_eq!(derive(Some(guard), Some(42), 0), "length", "guard={guard}");
    }
}

#[test]
fn product_fail_closed_action_surfaces_length_through_real_finish_path() {
    // PRODUCTION PATH regression: a live (unfinished, unguarded) sequence
    // is driven through the real fail-closed action — exactly what every
    // scheduler proposal-failure call site does — and then through the
    // real finish_sequence blocking terminal. The client must observe
    // "length" (server truncation), never "stop" (natural completion).
    let (mut a, rx) = test_seq(vec![5, 6, 42], 7, None, 10);
    a.finished = false;
    let acted = crate::scheduler::helpers::fail_dspark_product_sequence_closed(
        &StubModel { product: true },
        a.seq.slot_idx,
        &mut a.guard_stop,
        &mut a.finished,
        "production-path regression",
    );
    assert!(acted, "product identity must trigger the action");
    assert!(a.finished);
    let response = finish_and_recv(a, rx);
    assert_eq!(
        response.finish_reason, "length",
        "product fail-closed must surface as server truncation"
    );
}

#[test]
fn generic_model_action_is_a_noop_on_the_real_finish_path() {
    // Negative control on the same real path: a generic model's sequence
    // is untouched by the action and still finishes "stop" with budget
    // remaining — the legacy behavior is preserved end-to-end.
    let (mut a, rx) = test_seq(vec![5, 6, 42], 7, None, 10);
    a.finished = false;
    let acted = crate::scheduler::helpers::fail_dspark_product_sequence_closed(
        &StubModel::default(),
        a.seq.slot_idx,
        &mut a.guard_stop,
        &mut a.finished,
        "generic negative control",
    );
    assert!(!acted);
    assert!(
        !a.finished,
        "generic sequence must not be finished by the action"
    );
    a.finished = true;
    let response = finish_and_recv(a, rx);
    assert_eq!(response.finish_reason, "stop");
}

// ─── Production call-site handlers ─────────────────────────────────────
// These are the exact branch functions the scheduler call sites invoke
// (mtp_step bootstrap arms, verify_dflash_step re-propose arms, and the
// batched arms via fail_closed_if_lightning → the batched handler). A call
// site that stops routing through its handler, or a handler that stops
// failing product sequences closed, fails these tests.

#[test]
fn bootstrap_handler_fails_product_closed_for_both_outcomes() {
    for outcome in [
        crate::scheduler::helpers::ProposalOutcome::Empty,
        crate::scheduler::helpers::ProposalOutcome::Error,
    ] {
        let (mut a, _rx) = test_seq(vec![5, 6, 42], 7, None, 10);
        a.finished = false;
        assert!(
            crate::scheduler::helpers::handle_dspark_bootstrap_proposal_failure(
                &StubModel { product: true },
                &mut a,
                outcome
            ),
            "bootstrap handler must fail product closed ({outcome:?})"
        );
        assert!(a.finished);
        assert_eq!(
            a.guard_stop,
            Some(crate::scheduler::types::GUARD_STOP_DSPARK_PRODUCT_FAIL_CLOSED)
        );
    }
}

#[test]
fn repropose_handler_fails_product_closed_for_both_outcomes() {
    for outcome in [
        crate::scheduler::helpers::ProposalOutcome::Empty,
        crate::scheduler::helpers::ProposalOutcome::Error,
    ] {
        let (mut a, _rx) = test_seq(vec![5, 6, 42], 7, None, 10);
        a.finished = false;
        assert!(
            crate::scheduler::helpers::handle_dspark_repropose_failure(
                &StubModel { product: true },
                &mut a,
                outcome
            ),
            "re-propose handler must fail product closed ({outcome:?})"
        );
        assert!(a.finished);
        assert_eq!(
            a.guard_stop,
            Some(crate::scheduler::types::GUARD_STOP_DSPARK_PRODUCT_FAIL_CLOSED)
        );
    }
}

#[test]
fn batched_handler_fails_product_closed_for_every_arm_site() {
    for site in [
        "returned empty drafts",
        "fallback returned empty drafts",
        "fallback errored",
        "single returned empty drafts",
        "single errored",
        "batched proposer declined/errored",
        "stash save failed (single)",
    ] {
        let (mut a, _rx) = test_seq(vec![5, 6, 42], 7, None, 10);
        a.finished = false;
        assert!(
            crate::scheduler::helpers::handle_dspark_batched_proposal_failure(
                &StubModel { product: true },
                &mut a,
                site
            ),
            "batched handler must fail product closed (site={site})"
        );
        assert!(a.finished);
    }
    // Generic negative control on every handler.
    for site in ["returned empty drafts", "single errored"] {
        let (mut a, _rx) = test_seq(vec![5, 6, 42], 7, None, 10);
        a.finished = false;
        assert!(
            !crate::scheduler::helpers::handle_dspark_batched_proposal_failure(
                &StubModel::default(),
                &mut a,
                site
            ),
            "generic must be untouched (site={site})"
        );
        assert!(!a.finished);
    }
}

// ─── Streaming terminal contract ────────────────────────────────────────

#[tokio::test]
async fn product_fail_closed_streaming_done_carries_length_and_guard() {
    use tokio::sync::mpsc;
    let (tx, mut rx) = mpsc::channel(8);
    let (mut a, _blocking_rx) = test_seq(vec![5, 6, 42], 7, None, 10);
    a.finished = false;
    a.sink = crate::scheduler::types::ResponseSink::Streaming(tx);
    let acted = crate::scheduler::helpers::handle_dspark_repropose_failure(
        &StubModel { product: true },
        &mut a,
        crate::scheduler::helpers::ProposalOutcome::Error,
    );
    assert!(acted && a.finished);
    crate::scheduler::lifecycle::finish_sequence(&StubModel::default(), &mut a, MAX_SEQ_LEN);
    let done = tokio::time::timeout(std::time::Duration::from_secs(2), rx.recv())
        .await
        .expect("terminal frame within timeout")
        .expect("channel open");
    match done {
        crate::scheduler::StreamEvent::Done {
            finish_reason,
            guard_stop,
            ..
        } => {
            assert_eq!(
                finish_reason, "length",
                "streaming terminal must be a truncation"
            );
            assert_eq!(
                guard_stop,
                Some(crate::scheduler::types::GUARD_STOP_DSPARK_PRODUCT_FAIL_CLOSED),
                "streaming Done must carry the DSpark guard for diagnostics"
            );
        }
        _ => panic!("expected Done, got a non-Done terminal event"),
    }
}
