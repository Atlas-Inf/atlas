// SPDX-License-Identifier: AGPL-3.0-only
//! Actual scheduler entry and first-token-builder tests. No GPU/Model execution claim.
use crate::api::StreamEvent;
use crate::scheduler::{
    decode_logits_step::process_decode_logits, emit_step::emit_token,
    phase_promote_prefills::build_active_seq_from_prefill,
    prefill_a_step_params::build_prefill_in_progress, sched_ctx::SchedCtx, test_support::test_seq,
    types::ResponseSink,
};
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

use crate::scheduler::cancel_test_model as model;

const OPEN: u32 = 101; // Existing deterministic FP32 logits provider's winner.
const START: u32 = 900;
const END: u32 = 901;

fn exercise_emit(decode: bool, tools: bool, boundary: Option<u32>, suppress: bool, expected: bool) {
    let (mut a, _) = test_seq(vec![10], 20, None, 12);
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    a.sink = ResponseSink::Streaming(tx);
    a.finished = false;
    a.min_tokens = 0;
    a.enable_thinking = true;
    a.inside_thinking = true;
    a.think_ended = false;
    a.think_just_ended = false;
    a.thinking_budget = Some(128);
    a.thinking_tokens = 3;
    a.think_start_token = Some(START);
    a.think_end_token = Some(END);
    a.tool_call_start_token = Some(OPEN);
    a.tools_present = tools;
    a.suppress_tool_call = suppress;
    a.require_tool_call = false; // Actual auto-tools, not forced-tool-only behavior.
    let mut sched = SchedCtx::for_test();
    sched.limits.glm_tool_boundary = boundary;
    let mut rows = vec![a];
    if decode {
        process_decode_logits(
            &model::TestModel {
                tokens: vec![OPEN],
                host_logits: true,
                cancel_after_sampling: None,
                cancel_after_row_commit: None,
                verify: None,
            },
            &mut rows,
            DevicePtr::NULL,
            Instant::now(),
            Some(END),
            Some(START),
            None,
            Some(OPEN),
            None,
            false,
            &sched,
        );
    } else {
        emit_token(&mut rows[0], OPEN, None, &sched);
    }
    let a = &rows[0];
    // Actual ordinary sampling masks the native opener for generic/no-tools
    // requests. Its remaining zero logits tie at the final vocabulary ID.
    // MTP emission receives an already-issued token and does not resample it.
    let emitted = if decode && !expected { 2047 } else { OPEN };
    assert_eq!(
        a.output_tokens,
        [10, emitted],
        "qualified native opener must survive real sampling; generic mask stays"
    );
    assert_eq!(
        a.remaining, 19,
        "one generated token consumes one budget unit"
    );
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(token)) if token == emitted));
    assert!(rx.try_recv().is_err());
    assert!(!a.finished);
    assert_eq!(
        !a.inside_thinking, expected,
        "native GLM tool opener must end thinking"
    );
    assert_eq!(a.think_ended, expected);
    assert_eq!(a.thinking_tokens, if expected { 3 } else { 4 });
    assert!(
        !a.think_just_ended,
        "already emitted tool opener must not force another"
    );
    if expected {
        assert!(!a.force_end_thinking && a.inside_tool_body);
    }
}

#[test]
fn glm_tool_boundary_actual_mtp_emission() {
    for (tools, policy, closed) in [
        (false, Some(OPEN), false),
        (true, None, false),
        (true, Some(OPEN + 1), false),
        (true, Some(OPEN), true),
    ] {
        exercise_emit(false, tools, policy, false, closed);
    }
}

#[test]
fn glm_tool_boundary_actual_ordinary_decode() {
    for (tools, policy, suppress, closed) in [
        (false, Some(OPEN), false, false),
        (true, None, false, false),
        (true, Some(OPEN + 1), false, false),
        (true, Some(OPEN), true, false),
        (true, Some(OPEN), false, true),
    ] {
        exercise_emit(true, tools, policy, suppress, closed);
    }
}

#[test]
fn glm_tool_boundary_actual_first_token_builder() {
    for (tools, policy, closed) in [
        (false, Some(OPEN), false),
        (true, None, false),
        (true, Some(OPEN + 1), false),
        (true, Some(OPEN), true),
    ] {
        for max_tokens in [1, 20] {
            let (a, _) = test_seq(vec![], max_tokens, None, 28);
            let now = Instant::now();
            let p = build_prefill_in_progress(
                std::sync::Arc::new(vec![7; 28]),
                0,
                a.seq,
                28,
                max_tokens,
                0,
                vec![999],
                a.sink,
                None,
                now,
                0.0,
                0,
                1.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                vec![],
                true,
                Some(16),
                None,
                64,
                false,
                tools,
                false,
                false,
                None,
                None,
                None,
                None,
            );
            let a = build_active_seq_from_prefill(
                p,
                OPEN,
                false,
                false,
                0,
                max_tokens == 1,
                now,
                Some(END),
                Some(START),
                Some(OPEN),
                None,
                0,
                policy,
            );
            assert_eq!(a.output_tokens, [OPEN]);
            assert_eq!(a.remaining, max_tokens - 1);
            assert_eq!(a.finished, max_tokens == 1);
            assert_eq!(
                !a.inside_thinking, closed,
                "actual first token implicit boundary"
            );
            assert_eq!(a.think_ended, closed);
            assert!(!a.think_just_ended);
            assert_eq!(a.thinking_tokens, 0);
        }
    }
}
