// SPDX-License-Identifier: AGPL-3.0-only

//! Post-thinking EOS guard tests for auto-tools turns (`should_suppress_post_think_eos`).

use super::{process_decode_logits, should_suppress_post_think_eos};
use crate::scheduler::cancel_test_model as model;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::test_seq;
use spark_runtime::gpu::DevicePtr;
use std::time::Instant;

fn post_think_state() -> crate::scheduler::types::ActiveSeq {
    let (mut a, _rx) = test_seq(vec![10], 20, None, 12);
    a.finished = false;
    a.think_ended = true;
    a.tool_request = true;
    a
}

#[test]
fn auto_tools_without_current_call_allow_short_plain_answer() {
    let a = post_think_state();
    assert!(!a.require_tool_call);
    assert!(!a.tool_call_opened);
    assert!(!a.inside_tool_body);
    assert!(!a.tool_call_completed);
    assert!(
        !should_suppress_post_think_eos(&a, 0),
        "sticky auto-tools availability must not hold EOS for a plain answer"
    );
}

#[test]
fn thinking_disabled_auto_tools_still_allow_short_plain_answer() {
    let mut a = post_think_state();
    a.enable_thinking = false;
    assert!(!should_suppress_post_think_eos(&a, 0));
}

#[test]
fn auto_tools_plain_eos_is_not_held_by_sticky_request() {
    const EOS: u32 = 101;
    let (mut a, _rx) = test_seq(vec![10], 20, None, 12);
    a.finished = false;
    a.min_tokens = 0;
    a.eos_tokens = vec![EOS];
    a.enable_thinking = false;
    a.think_ended = true;
    a.tool_request = true;
    a.tools_present = true;
    a.tool_call_start_token = Some(900);
    a.tool_call_end_token = Some(901);
    let mut rows = vec![a];
    let sched = SchedCtx::for_test();

    process_decode_logits(
        &model::TestModel {
            tokens: vec![EOS],
            host_logits: true,
            cancel_after_sampling: None,
            cancel_after_row_commit: None,
            verify: None,
        },
        &mut rows,
        DevicePtr::NULL,
        Instant::now(),
        Some(901),
        Some(900),
        None,
        Some(900),
        Some(901),
        false,
        &sched,
    );

    assert!(rows[0].finished, "a plain auto-tools EOS must finish");
    assert_eq!(rows[0].output_tokens, [10, EOS]);
}

#[test]
fn required_tool_call_still_holds_post_think_eos() {
    let mut a = post_think_state();
    a.require_tool_call = true;
    assert!(should_suppress_post_think_eos(&a, 0));
}

#[test]
fn opened_incomplete_tool_call_still_holds_post_think_eos() {
    let mut a = post_think_state();
    a.tool_call_opened = true;
    assert!(should_suppress_post_think_eos(&a, 0));
}

#[test]
fn active_tool_body_still_holds_post_think_eos() {
    let mut a = post_think_state();
    a.inside_tool_body = true;
    assert!(should_suppress_post_think_eos(&a, 0));
}

#[test]
fn completed_tool_call_can_end_after_tool_result() {
    let mut a = post_think_state();
    a.tool_call_opened = true;
    a.tool_call_completed = true;
    assert!(!should_suppress_post_think_eos(&a, 0));
}
