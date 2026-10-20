// SPDX-License-Identifier: AGPL-3.0-only

//! Actual logits pipeline + emission, with explicit raw logits. This proves
//! policy behavior, not that a native model assigned these logits to its end.
use crate::scheduler::{
    emit_step::emit_token,
    logit_processors::{LogitsContext, SamplingLevers, process_position_logits},
    sample_step::{PositionKind, penalty_params_for_with_floor},
    sched_ctx::SchedCtx,
    test_support::test_seq,
};
use std::sync::Arc;

// Reuse the actual build-time MODEL.toml parser. Its other returned policy
// fields are intentionally unused here; no duplicate metadata parser/default.
#[allow(dead_code)]
#[path = "../../../../atlas-kernels/build_parse_behavior.rs"]
mod build_behavior;

const END: u32 = 101;
const PREVIOUS: u32 = 10;

fn pick(glm: bool, mid_word: bool, floor: u32, forced_budget: bool) -> u32 {
    // Held: a dropped receiver reads as a client disconnect.
    let (mut seq, _rx) = test_seq(vec![PREVIOUS], 128, None, 28);
    seq.finished = false;
    seq.min_tokens = 0;
    seq.enable_thinking = true;
    seq.inside_thinking = true;
    seq.think_ended = false;
    seq.think_just_ended = false;
    seq.think_start_token = Some(900);
    seq.think_end_token = Some(END);
    seq.thinking_budget = Some(16);
    seq.thinking_tokens = if forced_budget { 16 } else { 1 };
    seq.force_end_thinking = forced_budget;
    seq.in_code_fence = false;
    seq.require_tool_call = false;
    seq.tools_present = false; // Natural close is NOT a tools-only policy.
    seq.repetition_penalty = 1.0;
    seq.presence_penalty = 0.0;
    seq.frequency_penalty = 0.0;
    seq.dry_multiplier = 0.0;
    seq.lz_penalty = 0.0;
    let mut sched = SchedCtx::for_test();
    sched.limits.glm_tool_boundary = crate::glm_tool_boundary::native_opener(
        if glm { "glm5_next" } else { "qwen3" },
        true,
        Some(903),
    );
    let mut mid = vec![false; 2048];
    mid[PREVIOUS as usize] = mid_word;
    let mut boundary = vec![false; 2048];
    boundary[PREVIOUS as usize] = forced_budget;
    let ctx = LogitsContext {
        scratch: &sched.scratch,
        dumps: &sched.dumps,
        think_end_token: Some(END),
        think_start_token: Some(900),
        tool_call_start_token: Some(903),
        tool_call_end_token: Some(904),
        glm_tool_boundary: sched.limits.glm_tool_boundary,
        limits: sched.limits,
        mid_word_mask: Some(Arc::from(mid)),
        boundary_mask: Some(Arc::from(boundary)),
        sampling: SamplingLevers::default(),
        timing: Arc::default(),
        watchdog: Default::default(),
        stats: sched.stats.clone(),
    };
    let params =
        penalty_params_for_with_floor(&seq, PositionKind::FinalDecode, 0.0, None, vec![], floor);
    // Margin five is below A4's eight: both the hard mask and the independent
    // floor bias can reverse this natural end-token winner.
    let mut logits = vec![0.0; 2048];
    logits[END as usize] = if forced_budget { -20.0 } else { 5.0 };
    let forced = process_position_logits(
        &mut logits,
        &mut seq,
        &ctx,
        &params,
        PositionKind::FinalDecode,
    );
    let token = forced.unwrap_or_else(|| spark_runtime::sampler::argmax_first_wins_f32(&logits));
    emit_token(&mut seq, token, None, &sched);
    assert_eq!(seq.output_tokens, [PREVIOUS, token]);
    assert_eq!(seq.remaining, 127);
    if token == END {
        assert!(!seq.inside_thinking && seq.think_ended);
    }
    token
}

#[test]
fn glm_natural_end_mid_word_mask_preserves_native_winner() {
    // With `</think>` masked every other logit ties at 0: the lowest id wins
    // (the engine-wide greedy tie rule; this read 2047 under the old
    // last-index host argmax).
    assert_eq!(pick(false, true, 0, false), 0, "generic hard mask stays");
    assert_eq!(
        pick(true, true, 0, false),
        END,
        "GLM may close after numeric token"
    );
}

#[test]
fn glm_natural_end_actual_model_policy_has_no_minimum_reasoning_bias() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/gb10/glm-5.3-flash");
    let policy = build_behavior::parse_behavior(&dir);
    assert_eq!(
        policy.tool_call_parser, "poolside_v1",
        "actual GLM metadata read"
    );
    // Content and stream loop guards off (they cut legitimate counting and
    // code); the thinking-loop watchdog stays armed.
    assert!(
        !policy.enable_loop_watchdog
            && !policy.enable_stream_loop_guards
            && policy.enable_think_loop_watchdog
    );
    // Disable only the mid-word input to isolate A4. No global atomic writes.
    assert_eq!(
        pick(true, false, policy.min_reasoning_floor_tokens, false),
        END
    );
    assert_eq!(policy.min_reasoning_floor_tokens, 0);
}

#[test]
fn glm_natural_end_generic_floor_and_explicit_budget_are_preserved() {
    assert_eq!(pick(false, false, 16, false), 0, "generic A4 bias stays");
    for glm in [false, true] {
        assert_eq!(
            pick(glm, true, 16, true),
            END,
            "explicit budget still forces close"
        );
    }
}
