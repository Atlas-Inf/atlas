// SPDX-License-Identifier: AGPL-3.0-only

//! Actual ordinary sampling and MTP emission after a native GLM EOS winner.
//! No inference/numerical claim: the existing provider supplies fixed logits.
use crate::api::StreamEvent;
use crate::scheduler::{
    decode_logits_step::process_decode_logits, emit_step::emit_token,
    lifecycle::derive_finish_reason, sched_ctx::SchedCtx, test_support::test_seq,
    types::ResponseSink,
};
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

use crate::scheduler::cancel_test_model as model;

const SAMPLED: u32 = 101; // Existing FP32 fixture winner, not a hardcoded GLM ID.
const THINK_END: u32 = 901;

fn run(decode: bool, glm: bool, eos: bool, min_tokens: usize, required_tool: bool) {
    let (mut a, _) = test_seq(vec![10], 20, None, 28);
    let (tx, mut rx) = tokio::sync::mpsc::channel(8);
    a.sink = ResponseSink::Streaming(tx);
    a.finished = false;
    a.min_tokens = min_tokens;
    a.eos_tokens = vec![if eos { SAMPLED } else { SAMPLED + 1 }];
    a.enable_thinking = true;
    a.inside_thinking = true;
    a.think_ended = false;
    a.think_just_ended = false;
    a.thinking_budget = Some(16);
    a.thinking_tokens = 1;
    a.force_end_thinking = false;
    a.think_start_token = Some(900);
    a.think_end_token = Some(THINK_END);
    a.tool_call_start_token = Some(903);
    a.tools_present = false;
    a.tool_request = false;
    a.require_tool_call = required_tool;
    let mut sched = SchedCtx::for_test();
    sched.limits.glm_tool_boundary = crate::glm_tool_boundary::native_opener(
        if glm { "glm5_next" } else { "qwen3" },
        true,
        Some(903),
    );
    let mut rows = vec![a];
    if decode {
        process_decode_logits(
            &model::TestModel {
                tokens: vec![SAMPLED],
                host_logits: true,
                cancel_after_sampling: None,
                cancel_after_row_commit: None,
                verify: None,
            },
            &mut rows,
            DevicePtr::NULL,
            Instant::now(),
            Some(THINK_END),
            Some(900),
            None,
            Some(903),
            None,
            false,
            &sched,
        );
    } else {
        emit_token(&mut rows[0], SAMPLED, None, &sched);
    }
    let a = &rows[0];
    let stop = glm && eos && min_tokens == 0 && !required_tool;
    assert_eq!(
        a.finished, stop,
        "decode={decode} glm={glm} eos={eos} min={min_tokens} required={required_tool}"
    );
    assert_eq!(
        a.remaining, 19,
        "one actual sampled token consumes one budget unit"
    );
    if stop {
        assert_eq!(a.output_tokens, [10, SAMPLED]);
        assert_eq!(
            a.thinking_tokens, 1,
            "terminal EOS is not a reasoning token"
        );
        assert!(a.inside_thinking && !a.think_ended && !a.think_just_ended);
        assert!(
            !a.force_end_thinking && !a.think_force_closed,
            "no fabricated close"
        );
        assert_eq!(
            derive_finish_reason(
                a.guard_stop,
                a.output_tokens.last().copied(),
                &a.eos_tokens,
                None,
                a.remaining,
                a.seq.seq_len,
                4096
            ),
            "stop"
        );
    } else if !eos {
        assert_eq!(a.output_tokens, [10, SAMPLED]);
        assert_eq!(a.thinking_tokens, 2);
    }
    if eos {
        assert!(rx.try_recv().is_err(), "native EOS must never be streamed");
    } else {
        assert!(matches!(rx.try_recv(), Ok(StreamEvent::Token(SAMPLED))));
    }
    assert!(rx.try_recv().is_err());
}

#[test]
fn glm_native_eos_actual_ordinary_stops_without_inventing_content() {
    run(true, true, true, 0, false);
}

#[test]
fn glm_native_eos_actual_mtp_stops_without_inventing_content() {
    run(false, true, true, 0, false);
}

#[test]
fn glm_native_eos_preserves_generic_non_eos_and_other_stop_guards() {
    for decode in [false, true] {
        run(decode, false, true, 0, false);
        run(decode, true, false, 0, false);
        run(decode, true, true, 10, false);
        run(decode, true, true, 0, true);
    }
}
