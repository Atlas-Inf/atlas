// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::scheduler::cancel_test_model::{TestModel, VerifyScript};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::verify_dflash_step::step_verify_dflash;
use std::sync::Arc;

fn verified(remaining: usize, drafts: &[u32], picks: &[u32]) -> (ActiveSeq, RespRx) {
    let model = TestModel {
        tokens: vec![],
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: Some(Arc::new(VerifyScript {
            picks: picks.to_vec(),
            ..Default::default()
        })),
    };
    let (mut a, rx) = test_seq(vec![], remaining, None, 32);
    a.finished = false;
    a.min_tokens = 0;
    let before = a.seq.seq_len;
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(None, None, None, None);
    step_verify_dflash(
        &model,
        &mut a,
        &sched,
        drafts,
        &[],
        drafts.len(),
        &ctx,
        true,
    );
    assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
    // The production scheduler records the committed model span, not emitted output.
    a.mtp_acct
        .record_verify_emitted(a.seq.seq_len.saturating_sub(before));
    (a, rx)
}

#[test]
fn prediction_usage_counts_only_returned_drafts_at_length_boundary() {
    for (remaining, expected) in [(1, 1), (2, 2), (5, 4)] {
        let (a, rx) = verified(remaining, &[41, 42, 43, 44], &[41, 42, 43, 44, 45]);
        assert_eq!(a.mtp_acct.accepted_total(), 4, "raw telemetry stays intact");
        let response = finish_and_recv(a, rx);
        assert_eq!(response.output_tokens.len(), remaining);
        assert_eq!(
            response.accepted_prediction_tokens, expected,
            "only drafts in completion"
        );
    }
}

#[test]
fn prediction_usage_counts_eos_draft_but_not_unemitted_tail() {
    let eos = EOS[0];
    let (a, rx) = verified(10, &[41, eos, 43], &[41, eos, 43, 44]);
    assert_eq!(a.mtp_acct.accepted_total(), 3);
    let response = finish_and_recv(a, rx);
    assert_eq!(response.output_tokens, vec![41, eos]);
    assert_eq!(response.accepted_prediction_tokens, 2);
}

#[test]
fn prediction_usage_rejected_draft_bonus_is_not_accepted() {
    let (a, rx) = verified(1, &[41], &[42, 43]);
    let response = finish_and_recv(a, rx);
    assert_eq!(response.output_tokens, vec![42]);
    assert_eq!(response.accepted_prediction_tokens, 0);
}

#[test]
fn prediction_usage_skipped_drafts_do_not_turn_bonus_into_acceptance() {
    for skipped in [99, EOS[0]] {
        let (mut a, rx) = test_seq(vec![], 10, None, 34);
        a.finished = false;
        a.min_tokens = 7;
        a.think_end_token = Some(99);
        let sched = SchedCtx::for_test();
        crate::scheduler::emit_step::emit_span_row(&mut a, skipped, None, &sched, 0, 2);
        assert!(a.output_tokens.is_empty(), "draft is suppressed");
        crate::scheduler::emit_step::emit_span_row(&mut a, 42, None, &sched, 1, 2);
        a.mtp_acct.record_verify_emitted(2);
        let response = finish_and_recv(a, rx);
        assert_eq!(response.output_tokens, vec![42]);
        assert_eq!(response.accepted_prediction_tokens, 0);
    }
}

#[test]
fn prediction_usage_cancel_before_draft_push_reports_zero() {
    let (mut a, rx) = test_seq(vec![], 10, None, 34);
    a.finished = false;
    a.cancel_flag = Some(Arc::new(std::sync::atomic::AtomicBool::new(true)));
    crate::scheduler::emit_step::emit_span_row(&mut a, 42, None, &SchedCtx::for_test(), 0, 2);
    a.mtp_acct.record_verify_emitted(2);
    let response = finish_and_recv(a, rx);
    assert!(response.output_tokens.is_empty());
    assert_eq!(response.accepted_prediction_tokens, 0);
}

#[tokio::test]
async fn prediction_usage_stream_done_uses_returned_draft_count() {
    let (mut a, _rx) = verified(1, &[41, 42, 43], &[41, 42, 43, 44]);
    let (tx, mut events) = tokio::sync::mpsc::channel(8);
    a.sink = ResponseSink::Streaming(tx);
    finish_sequence(&StubModel::default(), &mut a, MAX_SEQ_LEN);
    match events.recv().await.expect("terminal event") {
        crate::api::StreamEvent::Done {
            completion_tokens,
            accepted_prediction_tokens,
            ..
        } => {
            assert_eq!(completion_tokens, 1);
            assert_eq!(accepted_prediction_tokens, 1);
        }
        _ => panic!("expected Done"),
    }
}

mod k4;
