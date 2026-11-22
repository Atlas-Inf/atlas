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

#[test]
fn spontaneous_thinking_and_reentry_preserve_a_zero_server_ceiling() {
    let sched = SchedCtx::for_test();
    for fires in [0, 1, 7] {
        let mut a = seq(false);
        a.spontaneous_think_budget = 0;
        a.think_watchdog_fires = fires;
        for _ in 0..2 {
            emit_token(&mut a, THINK_START, None, &sched);
            assert!(a.inside_thinking);
            assert_eq!(a.thinking_budget, None);
            for _ in 0..12 {
                emit_token(&mut a, LOW, None, &sched);
            }
            assert!(!a.force_end_thinking, "no hidden eight-token cap");
            emit_token(&mut a, THINK_END, None, &sched);
            assert!(!a.inside_thinking);
        }
    }
}

#[test]
fn spontaneous_thinking_keeps_nonzero_ceiling_decay_and_floor() {
    let sched = SchedCtx::for_test();
    for ceiling in [4, 64] {
        for fires in [0, 1, 7] {
            let mut a = seq(false);
            a.spontaneous_think_budget = ceiling;
            a.think_watchdog_fires = fires;
            emit_token(&mut a, THINK_START, None, &sched);
            assert!(a.inside_thinking);
            assert_eq!(a.thinking_budget, Some((ceiling >> fires.min(4)).max(8)));
        }
    }
}

/// Inkling 2: every block opens with `<|message_model|>` and closes with
/// `<|end_message|>`; the reasoning is the block the header opens with
/// `<|content_thinking|>` (`SchedLimits::message_open`).
mod inkling_blocks {
    use super::*;
    use crate::scheduler::test_support::inkling::{self, END, MESSAGE, TEXT, THINKING, TOOL_JSON};

    fn commit(a: &mut ActiveSeq, toks: &[u32], sched: &SchedCtx) {
        for &t in toks {
            emit_token(a, t, None, sched);
        }
    }

    #[test]
    fn reasoning_opens_on_the_header_and_think_start_and_both_are_recorded() {
        let s = inkling::sched();
        let (mut a, _rx) = inkling::seq(true, Some(64));
        commit(&mut a, &[THINKING, 5, 6], &s);
        assert!(a.inside_thinking && !a.think_ended);
        // Nothing consumed: the chat paths' rewrite reads every marker.
        assert_eq!(a.output_tokens, [MESSAGE, THINKING, 5, 6]);
        assert_eq!(a.thinking_budget, Some(64), "the request's own budget");
        // The think start is the block's first token.
        assert_eq!(a.thinking_tokens, 3);
        commit(&mut a, &[END], &s);
        assert!(!a.inside_thinking && a.think_ended && a.think_just_ended);
        assert!(!a.think_force_closed, "the model closed it");
        assert_eq!(a.output_tokens.last(), Some(&END));
    }

    #[test]
    fn a_think_start_without_a_header_is_content() {
        let s = inkling::sched();
        let (mut a, _rx) = inkling::seq(true, Some(64));
        a.output_tokens = vec![5];
        commit(&mut a, &[THINKING, 6], &s);
        assert!(!a.inside_thinking);
        assert_eq!(a.output_tokens, [5, THINKING, 6]);
    }

    #[test]
    fn an_answer_first_turn_never_counts_as_thinking() {
        // At effort 0 the first block may be the answer; a thinking block
        // after it is not the turn's reasoning (the chat paths drop it).
        let s = inkling::sched();
        let (mut a, _rx) = inkling::seq(true, Some(4));
        let toks = [
            TEXT, 5, 6, 7, 8, 9, END, MESSAGE, THINKING, 5, 6, 7, 8, 9, END,
        ];
        commit(&mut a, &toks, &s);
        assert!(!a.inside_thinking && !a.force_end_thinking);
        assert_eq!(a.thinking_tokens, 0);
        assert!(a.think_ended, "the answer began");
        assert_eq!(a.output_tokens.len(), 1 + toks.len(), "every id recorded");
        assert!(!a.finished);
    }

    #[test]
    fn a_block_close_outside_thinking_is_not_a_stray() {
        let s = inkling::sched();
        let (mut a, _rx) = inkling::seq(true, None);
        let toks = [
            THINKING, 5, END, MESSAGE, TEXT, 6, END, MESSAGE, 7, TOOL_JSON, 8, END,
        ];
        commit(&mut a, &toks, &s);
        assert_eq!(a.think_skip_count, 0);
        assert_eq!(
            a.output_tokens.len(),
            1 + toks.len(),
            "every close recorded"
        );
        assert!(!a.finished && a.guard_stop.is_none());
    }

    #[test]
    fn the_budget_arms_the_close_and_end_message_is_it() {
        let s = inkling::sched();
        let (mut a, _rx) = inkling::seq(true, Some(4));
        commit(&mut a, &[THINKING, 5, 6], &s);
        assert!(!a.force_end_thinking);
        commit(&mut a, &[7], &s);
        assert!(
            a.force_end_thinking,
            "4 thinking tokens of a 4-token budget"
        );
        commit(&mut a, &[END], &s);
        assert!(!a.inside_thinking && a.think_ended && a.think_force_closed);
        // The model then opens its own next block, the answer.
        commit(&mut a, &[MESSAGE, TEXT, 9], &s);
        assert!(!a.inside_thinking && !a.finished);
        assert_eq!(a.thinking_tokens, 4);
    }

    #[test]
    fn thinking_off_bounds_the_reasoning_by_the_server_ceiling() {
        // Inkling's template reasons whatever the request says; with
        // thinking off the chat paths hide it, and the server ceiling (its
        // MODEL.toml `max_thinking_budget`) bounds it.
        let s = inkling::sched();
        let (mut a, _rx) = inkling::seq(false, None);
        commit(&mut a, &[THINKING], &s);
        assert!(a.inside_thinking);
        assert_eq!(a.thinking_budget, Some(8192));
    }

    #[test]
    fn the_thinking_loop_watchdog_arms_the_close() {
        let mut s = inkling::sched();
        s.watchdog.enable_think_loop_watchdog = true;
        let (mut a, _rx) = inkling::seq(true, None);
        commit(&mut a, &[THINKING], &s);
        let period: Vec<u32> = (20..28).collect();
        for _ in 0..40 {
            commit(&mut a, &period, &s);
            if a.force_end_thinking {
                break;
            }
        }
        assert!(a.force_end_thinking && a.think_watchdog_fires == 1);
        assert!(a.inside_thinking, "armed, not yet closed");
    }

    #[test]
    fn greedy_reasoning_keeps_the_gpu_argmax() {
        use crate::scheduler::fast_greedy::{
            raw_argmax_is_pick, raw_pick_masked, thinking_raw_pick,
        };
        let mut s = inkling::sched();
        s.watchdog.confidence_early_stop = false; // as Inkling's MODEL.toml
        let (mut a, _rx) = inkling::seq(true, Some(64));
        emit_token(&mut a, THINKING, None, &s);
        let raw = thinking_raw_pick(&s.limits, &s.watchdog);
        assert!(raw_argmax_is_pick(&a, raw));
        // What the pipeline masks or biases in there is redone on the host.
        a.tool_call_start_token = Some(1200);
        assert!(raw_pick_masked(&a, END) && raw_pick_masked(&a, 1200));
        assert!(!raw_pick_masked(&a, 5));
        // An armed close is the pipeline's to inject.
        a.force_end_thinking = true;
        assert!(!raw_argmax_is_pick(&a, raw));
        a.force_end_thinking = false;
        // F2 reads the logits.
        s.watchdog.confidence_early_stop = true;
        assert!(!raw_argmax_is_pick(
            &a,
            thinking_raw_pick(&s.limits, &s.watchdog)
        ));
        // Every other model's thinking rows stay on the host.
        let other = SchedCtx::for_test();
        s.watchdog.confidence_early_stop = false;
        assert!(!thinking_raw_pick(&other.limits, &s.watchdog));
        assert!(!raw_argmax_is_pick(&a, false));
    }

    #[test]
    fn serial_decode_commits_the_blocks_as_emit_does() {
        // The plain-decode commit (`process_decode_logits`) runs the same
        // gate: a header-opened thinking block, its close, then the answer.
        let s = inkling::sched();
        let toks = [THINKING, 5, END, MESSAGE, TEXT, 6];
        let (serial, _rx) = inkling::seq(true, Some(64));
        let mut batch = vec![serial];
        let mut states = Vec::new();
        for &t in &toks {
            let mut r = vec![0.0f32; 2048];
            r[t as usize] = 9.0;
            SCRIPT.with(|sc| {
                *sc.borrow_mut() = Some(Script {
                    rows: vec![r],
                    cursor: 0,
                })
            });
            process_decode_logits(
                &model(),
                &mut batch,
                DevicePtr::NULL,
                std::time::Instant::now(),
                Some(END),
                Some(THINKING),
                None,
                None,
                false,
                &s,
            );
            states.push(ThinkState::of(&batch[0]).picks_only());
        }
        SCRIPT.with(|sc| *sc.borrow_mut() = None);
        let (mut emitted, _rx2) = inkling::seq(true, Some(64));
        let mut want = Vec::new();
        for &t in &toks {
            emit_token(&mut emitted, t, None, &s);
            want.push(ThinkState::of(&emitted).picks_only());
        }
        assert_eq!(batch[0].output_tokens, emitted.output_tokens);
        assert_eq!(states, want);
        assert!(states[1] != states[3], "the span crosses the close");
    }
}
