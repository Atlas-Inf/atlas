// SPDX-License-Identifier: AGPL-3.0-only

//! One tie rule and one commit rule, whichever path picks or commits.

use super::shadow::ThinkState;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel};
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{THINK_END, THINK_START, test_seq};
use crate::scheduler::types::ActiveSeq;
use spark_runtime::gpu::DevicePtr;

const END: u32 = 900;
const LOW: u32 = 300;
const HIGH: u32 = 700;

fn model() -> TestModel {
    TestModel {
        tokens: Vec::new(),
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    }
}

fn seq(inside: bool) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![5, 6], 64, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    a.eos_tokens = vec![END];
    a.inside_thinking = inside;
    a.think_ended = !inside;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a
}

/// Rows whose top-1 is an exact tie between LOW and HIGH.
fn tied(rows: usize) {
    let mut r = vec![1.0f32; 2048];
    r[LOW as usize] = 9.0;
    r[HIGH as usize] = 9.0;
    SCRIPT.with(|s| {
        *s.borrow_mut() = Some(Script {
            rows: vec![r; rows],
            cursor: 0,
        })
    });
}

fn decode(batch: &mut Vec<ActiveSeq>, sched: &SchedCtx) {
    process_decode_logits(
        &model(),
        batch,
        DevicePtr::NULL,
        std::time::Instant::now(),
        Some(THINK_END),
        Some(THINK_START),
        None,
        None,
        false,
        sched,
    );
}

#[test]
fn an_exact_top1_tie_picks_the_lowest_id_on_every_path() {
    let sched = SchedCtx::for_test();
    let ctx = sched.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
    for inside in [true, false] {
        tied(2);
        // Alone (outside thinking: the GPU argmax) ...
        let mut alone = vec![seq(inside)];
        decode(&mut alone, &sched);
        // ... and beside a thinking row, which sends the batch to the host.
        let mut beside = vec![seq(inside), seq(true)];
        decode(&mut beside, &sched);
        // A verify span of one row, and the MTP bootstrap's one-row pick.
        let mut v = seq(inside);
        let verified = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            &model(),
            &[LOW],
            &mut v,
            &ctx,
            0,
        );
        let mut b = seq(inside);
        let boot =
            crate::scheduler::fast_greedy::pick_row(&model(), DevicePtr::NULL, &mut b, &ctx, None)
                .unwrap();
        let picks = [alone[0].last_token, beside[0].last_token, verified[0], boot];
        assert_eq!(picks, [LOW; 4], "inside_thinking={inside}");
    }
    SCRIPT.with(|s| *s.borrow_mut() = None);
}

/// An end token inside `<think>`, through decode and through emit.
fn end_in_think(honor: bool, serial: bool) -> ActiveSeq {
    let mut sched = SchedCtx::for_test();
    sched.watchdog.honor_eos_inside_thinking = honor;
    let mut a = seq(true);
    if serial {
        let mut r = vec![0.0f32; 2048];
        r[END as usize] = 9.0;
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: vec![r],
                cursor: 0,
            })
        });
        let mut batch = vec![a];
        decode(&mut batch, &sched);
        SCRIPT.with(|s| *s.borrow_mut() = None);
        a = batch.pop().unwrap();
    } else {
        emit_token(&mut a, END, None, &sched);
    }
    a
}

#[test]
fn an_end_token_inside_thinking_commits_alike_through_decode_and_emit() {
    for honor in [true, false] {
        let (s, m) = (end_in_think(honor, true), end_in_think(honor, false));
        assert_eq!(m.output_tokens, s.output_tokens, "honor={honor}");
        assert_eq!(ThinkState::of(&m), ThinkState::of(&s), "honor={honor}");
        assert!(
            !s.finished && s.output_tokens == [5, 6],
            "discarded, not a stop"
        );
        // Honored: the block closes as `</think>` would (emit used to keep it
        // open, so the next pick differed from serial decode's).
        assert_eq!(s.inside_thinking, !honor);
        assert_eq!(s.think_just_ended, honor);
    }
}

#[test]
fn a_rejected_verify_position_leaves_no_trace_in_the_think_state() {
    // F2's streak used to tick once per VERIFIED position, rejected ones too.
    let mut sched = SchedCtx::for_test();
    sched.watchdog.confidence_early_stop = true;
    sched.watchdog.confidence_run_length = 60;
    let ctx = sched.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
    let mut r = vec![0.0f32; 2048];
    r[LOW as usize] = 40.0;
    SCRIPT.with(|s| {
        *s.borrow_mut() = Some(Script {
            rows: vec![r; 4],
            cursor: 0,
        })
    });
    let mut a = seq(true);
    a.thinking_tokens = 500;
    let before = ThinkState::of(&a);
    let picks = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        &model(),
        &[LOW; 4],
        &mut a,
        &ctx,
        0,
    );
    assert_eq!(picks, [LOW; 4]);
    assert_eq!(
        ThinkState::of(&a),
        before,
        "the span is replayed, then restored"
    );
    assert_eq!(a.output_tokens, [5, 6]);
    // Committing two of the four positions advances the streak by two.
    emit_token(&mut a, LOW, None, &sched);
    emit_token(&mut a, LOW, None, &sched);
    assert_eq!(a.consecutive_confident, 2);
    SCRIPT.with(|s| *s.borrow_mut() = None);
}

/// A short post-`</think>` EOS held by a real tool obligation, through decode
/// and through emit — #175's eos_policy regression shape
/// (live_irrelevance_25-2-13). #174's `should_suppress_post_think_eos` arms on
/// a real obligation (`require_tool_call` / open incomplete call / tool body),
/// deliberately NOT on bare `tool_request`; the obligation here is an open,
/// incomplete call.
fn post_think_end(content: usize, serial: bool) -> ActiveSeq {
    let sched = SchedCtx::for_test();
    let mut a = seq(false); // inside_thinking=false, think_ended=true
    a.output_tokens = (0..content as u32).collect();
    a.thinking_tokens = 0;
    a.tool_call_opened = true;
    a.tool_call_completed = false;
    a.tool_request = true;
    if serial {
        let mut r = vec![0.0f32; 2048];
        r[END as usize] = 9.0;
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: vec![r],
                cursor: 0,
            })
        });
        let mut batch = vec![a];
        decode(&mut batch, &sched);
        SCRIPT.with(|s| *s.borrow_mut() = None);
        a = batch.pop().unwrap();
    } else {
        emit_token(&mut a, END, None, &sched);
    }
    a
}

#[test]
fn a_short_post_think_end_token_under_a_real_obligation_is_discarded_alike() {
    for content in [7usize, 12, 15] {
        let (s, m) = (
            post_think_end(content, true),
            post_think_end(content, false),
        );
        assert!(
            !s.finished,
            "content={content}: suppressed EOS must not end"
        );
        assert_eq!(s.output_tokens.len(), content, "content={content}");
        assert_eq!(m.output_tokens, s.output_tokens, "content={content}");
        assert_eq!(ThinkState::of(&m), ThinkState::of(&s), "content={content}");
        // Emit popped the token AND undid the spec-resume count, so it stays
        // at serial's zero (pre-merge it kept +1 for a token that never was).
        assert_eq!(m.post_think_emitted, 0, "content={content}");
    }
}

#[test]
fn a_post_think_end_token_with_enough_content_stops_alike() {
    let (s, m) = (post_think_end(16, true), post_think_end(16, false));
    assert!(s.finished && m.finished);
    assert_eq!(m.output_tokens, s.output_tokens);
    assert_eq!(*s.output_tokens.last().unwrap(), END);
    // The stop token IS recorded, so emit's post-think count keeps it (+1);
    // serial tracks the guard on the emit path only.
    assert_eq!(m.post_think_emitted, 1);
}

/// Bare `tool_request` (tools declared, none called) does NOT arm the guard:
/// an auto-tools answer to a tool result may end short.
#[test]
fn a_bare_tool_request_does_not_hold_a_post_think_end_token() {
    let mut a = seq(false);
    a.output_tokens = (0..7u32).collect();
    a.tool_request = true;
    emit_token(&mut a, END, None, &SchedCtx::for_test());
    assert!(a.finished);
    assert_eq!(*a.output_tokens.last().unwrap(), END);
}
