// SPDX-License-Identifier: AGPL-3.0-only

use super::{emit_token, retire_if_cancelled};
use crate::api::StreamEvent;
use crate::scheduler::{
    decode_logits_step::process_decode_logits,
    lifecycle::derive_finish_reason,
    sched_ctx::SchedCtx,
    test_support::test_seq,
    types::{ActiveSeq, ResponseSink},
};
use spark_runtime::gpu::DevicePtr;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use crate::scheduler::cancel_test_model as model;
use model::TestModel;

fn row() -> (ActiveSeq, tokio::sync::mpsc::Receiver<StreamEvent>) {
    let (mut a, _) = test_seq(vec![10], 20, None, 12);
    a.finished = false;
    a.min_tokens = 0;
    let (tx, rx) = tokio::sync::mpsc::channel(8);
    a.sink = ResponseSink::Streaming(tx);
    (a, rx)
}

fn decode(model: &TestModel, rows: &mut Vec<ActiveSeq>, sched: &SchedCtx) {
    process_decode_logits(
        model,
        rows,
        DevicePtr::NULL,
        std::time::Instant::now(),
        None,
        None,
        None,
        None,
        false,
        sched,
    );
}

#[test]
fn non_spec_preset_cancel_retires_without_committing_sampled_token() {
    let (mut a, mut rx) = row();
    a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
    a.guard_stop = Some("existing_guard");
    let before_time = a.last_token_time;
    let mut rows = vec![a];
    decode(
        &TestModel {
            tokens: vec![101],
            host_logits: false,
            cancel_after_sampling: None,
            cancel_after_row_commit: None,
            verify: None,
        },
        &mut rows,
        &SchedCtx::for_test(),
    );
    let a = &rows[0];
    assert!(
        a.finished,
        "preset cancellation must retire the non-spec row"
    );
    assert_eq!(a.output_tokens, vec![10]);
    assert_eq!(a.last_token, 10);
    assert_eq!(a.last_token_time, before_time);
    assert_eq!(a.remaining, 20);
    assert_eq!(a.seq.seq_len, 12, "no rollback or further state advance");
    assert_eq!(a.guard_stop, Some("existing_guard"));
    assert!(rx.try_recv().is_err(), "no stream token may be committed");
}

#[test]
fn non_spec_cancel_after_sampling_is_rechecked_before_commit() {
    let (mut a, mut rx) = row();
    let flag = Arc::new(AtomicBool::new(false));
    a.cancel_flag = Some(flag.clone());
    let mut rows = vec![a];
    decode(
        &TestModel {
            tokens: vec![101],
            host_logits: false,
            cancel_after_sampling: Some(flag),
            cancel_after_row_commit: None,
            verify: None,
        },
        &mut rows,
        &SchedCtx::for_test(),
    );
    assert!(rows[0].finished);
    assert_eq!(rows[0].output_tokens, vec![10]);
    assert!(rx.try_recv().is_err());
}

#[test]
fn non_spec_midbatch_cancel_preserves_other_rows_and_original_mapping() {
    let (a, mut rx0) = row();
    let (mut b, mut rx1) = row();
    let (c, mut rx2) = row();
    let flag = Arc::new(AtomicBool::new(false));
    b.cancel_flag = Some(flag.clone());
    let mut rows = vec![a, b, c];
    decode(
        &TestModel {
            tokens: vec![101, 102, 103],
            host_logits: false,
            cancel_after_sampling: None,
            cancel_after_row_commit: Some(flag),
            verify: None,
        },
        &mut rows,
        &SchedCtx::for_test(),
    );
    assert_eq!(
        rows.len(),
        3,
        "retirement must not compact current logits rows"
    );
    assert_eq!(rows[0].output_tokens, vec![10, 101]);
    assert_eq!(rows[1].output_tokens, vec![10]);
    assert_eq!(rows[2].output_tokens, vec![10, 103]);
    assert!(!rows[0].finished && rows[1].finished && !rows[2].finished);
    assert!(matches!(rx0.try_recv(), Ok(StreamEvent::Token(101))));
    assert!(rx1.try_recv().is_err());
    assert!(matches!(rx2.try_recv(), Ok(StreamEvent::Token(103))));
}

#[test]
fn emit_and_retirement_helper_preserve_budgets_and_finish_precedence() {
    for (remaining, guard, expected) in [
        (20, None, "stop"),
        (0, None, "length"),
        (20, Some("existing_guard"), "length"),
    ] {
        let (mut a, mut rx) = row();
        a.remaining = remaining;
        a.guard_stop = guard;
        a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
        emit_token(&mut a, 101, None, &SchedCtx::for_test());
        assert!(a.finished && retire_if_cancelled(&mut a));
        assert_eq!(a.output_tokens, vec![10]);
        assert_eq!(a.remaining, remaining);
        assert_eq!(
            derive_finish_reason(
                a.guard_stop,
                a.output_tokens.last().copied(),
                &a.eos_tokens,
                a.tool_call_end_token,
                a.remaining,
                a.seq.seq_len,
                2048
            ),
            expected
        );
        assert!(rx.try_recv().is_err());
    }
    let (mut a, mut rx) = row();
    a.cancel_flag = Some(Arc::new(AtomicBool::new(false)));
    assert!(!retire_if_cancelled(&mut a));
    emit_token(&mut a, 101, None, &SchedCtx::for_test());
    assert_eq!(a.output_tokens, vec![10, 101]);
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(101))));
    a.cancel_flag
        .as_ref()
        .unwrap()
        .store(true, Ordering::Release);
    emit_token(&mut a, 102, None, &SchedCtx::for_test());
    assert_eq!(a.output_tokens, vec![10, 101]);
    assert!(rx.try_recv().is_err());
}

#[test]
fn cancelled_host_sampling_preserves_request_adaptive_state_serial_and_parallel() {
    use crate::adaptive_sampler::GenerationZone;
    let model = TestModel {
        tokens: vec![],
        host_logits: true,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    };
    for n in [1, 3] {
        let mut rows = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..n {
            let (mut a, rx) = row();
            a.tool_call_opened = true;
            a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
            assert_eq!(a.adaptive.zone, GenerationZone::FreeText);
            rows.push(a);
            receivers.push(rx);
        }
        process_decode_logits(
            &model,
            &mut rows,
            DevicePtr::NULL,
            std::time::Instant::now(),
            None,
            None,
            None,
            None,
            true,
            &SchedCtx::for_test(),
        );
        for a in &rows {
            assert!(a.finished);
            assert_eq!(
                a.adaptive.zone,
                GenerationZone::FreeText,
                "cancelled rows must not enter mutable host sampler"
            );
            assert_eq!(a.output_tokens, vec![10]);
            assert_eq!(a.remaining, 20);
        }
        assert!(receivers.iter_mut().all(|rx| rx.try_recv().is_err()));
    }
    // Negative control: the same live row DOES update adaptive state and emit.
    let (mut a, mut rx) = row();
    a.tool_call_opened = true;
    let mut rows = vec![a];
    process_decode_logits(
        &model,
        &mut rows,
        DevicePtr::NULL,
        std::time::Instant::now(),
        None,
        None,
        None,
        None,
        true,
        &SchedCtx::for_test(),
    );
    assert_eq!(rows[0].adaptive.zone, GenerationZone::ToolCall);
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(101))));
}

#[test]
fn non_spec_without_cancel_still_enforces_token_and_context_limits() {
    for context_limit in [false, true] {
        let (mut a, _rx) = row();
        let mut sched = SchedCtx::for_test();
        if context_limit {
            sched.limits.max_seq_len = a.seq.seq_len;
        } else {
            a.remaining = 1;
        }
        let mut rows = vec![a];
        decode(
            &TestModel {
                tokens: vec![101],
                host_logits: false,
                cancel_after_sampling: None,
                cancel_after_row_commit: None,
                verify: None,
            },
            &mut rows,
            &sched,
        );
        let a = &rows[0];
        assert!(a.finished);
        assert_eq!(
            derive_finish_reason(
                a.guard_stop,
                a.output_tokens.last().copied(),
                &a.eos_tokens,
                a.tool_call_end_token,
                a.remaining,
                a.seq.seq_len,
                sched.limits.max_seq_len
            ),
            "length"
        );
    }
}

/// `stream:false` caller aborted before any result: the oneshot receiver is
/// dropped while the sender lives on in `ResponseSink::Blocking`. Retirement
/// must mark the sequence finished without committing more tokens.
#[test]
fn buffered_disconnect_retires_dead_receiver_at_emit() {
    let (mut a, rx) = test_seq(vec![10], 20, None, 12);
    a.finished = false;
    drop(rx);
    assert!(a.sink.receiver_closed(), "dropped oneshot must read closed");
    emit_token(&mut a, 101, None, &SchedCtx::for_test());
    assert!(a.finished, "dead buffered sequence must retire");
    assert_eq!(a.output_tokens, vec![10], "no token committed after drop");
    assert!(
        retire_if_cancelled(&mut a),
        "retire is idempotent for dead rx"
    );
}

/// Negative control: a live buffered receiver is not treated as cancelled, so
/// normal emission still happens.
#[test]
fn buffered_live_receiver_still_emits() {
    let (mut a, rx) = test_seq(vec![10], 20, None, 12);
    a.finished = false;
    assert!(!a.sink.receiver_closed());
    emit_token(&mut a, 101, None, &SchedCtx::for_test());
    assert_eq!(a.output_tokens, vec![10, 101]);
    assert!(!a.finished);
    // Blocking results are delivered by lifecycle finalization, not emission.
    // Keep the real receiver alive throughout this negative control.
    let _live_receiver = rx;
}

/// Decode path must observe buffered disconnect too, so a dead caller is
/// retired before logits commit rather than decoding to `max_tokens`.
#[test]
fn buffered_disconnect_retires_at_decode_boundary() {
    let (mut a, rx) = test_seq(vec![10], 20, None, 12);
    a.finished = false;
    a.min_tokens = 0;
    drop(rx);
    let mut rows = vec![a];
    decode(
        &TestModel {
            tokens: vec![101],
            host_logits: false,
            cancel_after_sampling: None,
            cancel_after_row_commit: None,
            verify: None,
        },
        &mut rows,
        &SchedCtx::for_test(),
    );
    assert!(rows[0].finished, "dead buffered row must retire at decode");
    assert_eq!(rows[0].output_tokens, vec![10]);
}

/// Streaming cancellation semantics must be unchanged: a closed stream
/// receiver alone does not retire (the streaming batch sender may still have
/// queued events), while the cooperative `cancel_flag` still does.
#[test]
fn streaming_cancel_semantics_unchanged_for_blocking_helper() {
    let (mut a, mut rx) = row();
    a.cancel_flag = None;
    // Fake a "closed-looking" streaming sink by dropping the receiver, then
    // confirm the sink helper never treats Streaming as closed.
    let (tx, dropped) = tokio::sync::mpsc::channel(8);
    drop(dropped);
    a.sink = ResponseSink::Streaming(tx);
    assert!(!a.sink.receiver_closed());
    assert!(!retire_if_cancelled(&mut a));
    a.cancel_flag = Some(Arc::new(AtomicBool::new(true)));
    assert!(retire_if_cancelled(&mut a));
    let _ = rx.try_recv();
}
