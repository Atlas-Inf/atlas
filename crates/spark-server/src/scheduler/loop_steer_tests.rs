// SPDX-License-Identifier: AGPL-3.0-only

//! Content-loop steering (`loop_steer`): the rule, its fast-path superset,
//! and that speculation commits exactly serial decode's tokens under it
//! (single-sequence spans of every width and the batched verify), that the
//! watchdog hard-stops only once the budget is spent, and that a sequence
//! with steering off is untouched.
//!
//! The harness is `min_tokens_ban_tests.rs`'s: one deterministic logits
//! stream, a row per position, decoded serially (`process_decode_logits`)
//! and as verify spans (`verify_pick_all_with_pipeline` + the K-verdict).

use super::{banned_token, continuation, span_may_steer};
use crate::api::inference_types::RepetitionDetectionParams;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel};
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::levers::SchedLevers;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::test_seq;
use crate::scheduler::types::ActiveSeq;
use crate::scheduler::types::GUARD_STOP_CONTENT_LOOP;
use crate::scheduler::verify_pipeline_helper::{verify_pick_all_with_pipeline, verify_pick_batch};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::argmax_first_wins_f32;
use std::sync::Arc;

const V: usize = 2048;
/// The two tokens the stream alternates between (a period-2 attractor).
const A: u32 = 500;
const B: u32 = 501;
/// Tokens already out before the stream (`test_seq`).
const PRIOR: usize = 2;
const BUDGET: usize = 1000;
/// Period 2..8, four back-to-back repeats: a loop forms in 8 tokens.
const DETECT: RepetitionDetectionParams = RepetitionDetectionParams {
    min_pattern_size: 2,
    max_pattern_size: 8,
    min_count: 4,
};

fn sched(watchdog: bool) -> SchedCtx {
    let mut levers = SchedLevers::defaults();
    levers.fast_greedy_chat = true;
    let levers = Arc::new(levers);
    levers.set_loop_watchdog(watchdog);
    SchedCtx::new(
        Default::default(),
        levers,
        Arc::default(),
        crate::scheduler::limits::SchedLimits::NONE,
        Default::default(),
    )
}

fn model() -> TestModel {
    TestModel {
        tokens: Vec::new(),
        host_logits: false,
        cancel_after_sampling: None,
        cancel_after_row_commit: None,
        verify: None,
    }
}

/// A chat sequence as admission arms it, with `max` steers.
fn chat_seq(max: u32, remaining: usize) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![300, 301], remaining, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    a.repetition_detection = Some(DETECT);
    a.loop_steer_max = max;
    a
}

/// Small integer logits (exact in BF16). After a few free positions the
/// winner alternates A/B (12), so the output falls into a period-2 loop again
/// and again; each position has its own runner-up (11), the steer's landing
/// spot, and some rows tie the winner with a higher id or let a free token
/// win outright, so loops form at different offsets.
fn rows(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut x = seed;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) as u32
    };
    (0..n)
        .map(|p| {
            let mut r: Vec<f32> = (0..V).map(|_| (next() % 8) as f32).collect();
            let win = if p % 2 == 0 { A } else { B } as usize;
            r[win] = 12.0;
            r[600 + (next() % 400) as usize] = 11.0;
            match next() % 16 {
                0 | 1 if p < 6 => r[100 + (next() % 300) as usize] = 13.0,
                2 => r[1100 + (next() % 300) as usize] = 12.0,
                3 => r[100 + (next() % 300) as usize] = 13.0,
                _ => {}
            }
            r
        })
        .collect()
}

fn script(stream: Vec<Vec<f32>>) {
    SCRIPT.with(|s| {
        *s.borrow_mut() = Some(Script {
            rows: stream,
            cursor: 0,
        })
    });
}

fn at(cursor: usize) {
    SCRIPT.with(|s| s.borrow_mut().as_mut().unwrap().cursor = cursor);
}

fn stream() -> Vec<Vec<f32>> {
    SCRIPT.with(|s| s.borrow().as_ref().unwrap().rows.clone())
}

/// Serial decode: the sequence and its pick at every position.
fn serial(max: u32, n: usize, watchdog: bool) -> (ActiveSeq, Vec<u32>) {
    let s = sched(watchdog);
    let mut batch = vec![chat_seq(max, BUDGET)];
    let mut picks = Vec::new();
    for t in 0..n {
        at(t);
        let none = None;
        process_decode_logits(
            &model(),
            &mut batch,
            DevicePtr::NULL,
            std::time::Instant::now(),
            none,
            none,
            none,
            none,
            false,
            &s,
        );
        picks.push(batch[0].last_token);
        if batch[0].finished {
            break;
        }
    }
    (batch.pop().unwrap(), picks)
}

/// Verify spans of `ks` rows (cycled) over `n` positions; `draft(p)` drafts
/// position `p`.
fn speculative(
    max: u32,
    n: usize,
    ks: &[usize],
    watchdog: bool,
    draft: impl Fn(usize) -> u32,
) -> ActiveSeq {
    let s = sched(watchdog);
    let ctx = s.verify_logits_ctx(None, None, None, None);
    let stream = stream();
    let mut a = chat_seq(max, BUDGET);
    let (mut p, mut step) = (0usize, 0usize);
    while p < n && !a.finished {
        let k = ks[step % ks.len()].min(n - p);
        step += 1;
        at(p);
        let gpu: Vec<u32> = (0..k)
            .map(|i| argmax_first_wins_f32(&stream[p + i]))
            .collect();
        let drafts: Vec<u32> = (0..k - 1).map(|i| draft(p + i)).collect();
        let picks = verify_pick_all_with_pipeline(&model(), &gpu, &mut a, &ctx, 0);
        let accepted = drafts
            .iter()
            .zip(&picks)
            .take_while(|(d, v)| d == v)
            .count();
        for &d in &drafts[..accepted] {
            emit_token(&mut a, d, None, &s);
            if a.finished {
                return a;
            }
        }
        emit_token(&mut a, picks[accepted], None, &s);
        p += accepted + 1;
    }
    a
}

/// Serial vs every span shape; returns serial decode's sequence.
fn check(seed: u64, max: u32) -> ActiveSeq {
    const N: usize = 120;
    script(rows(seed, N));
    let (want, reference) = serial(max, N, false);
    // Serial decode's own picks, every fifth one wrong.
    let draft = |p: usize| {
        let d = reference.get(p).copied().unwrap_or(0);
        if p % 5 == 4 { d ^ 1 } else { d }
    };
    for ks in [&[2usize][..], &[4], &[8], &[5, 2, 3, 1], &[3, 6]] {
        let got = speculative(max, reference.len(), ks, false, draft);
        assert_eq!(
            got.output_tokens, want.output_tokens,
            "seed {seed} max {max} K={ks:?}"
        );
        assert_eq!(
            got.loop_steers, want.loop_steers,
            "seed {seed} max {max} K={ks:?}"
        );
    }
    SCRIPT.with(|s| *s.borrow_mut() = None);
    want
}

/// Does `out` run a loop past the point `DETECT` catches it: one more
/// repeat than its threshold? (A steer acts on the pick right after the
/// threshold's last token, so the threshold run itself is in the output.)
fn holds_a_loop(out: &[u32]) -> bool {
    (0..=out.len()).any(|n| continuation(n, |i| out[i], (2, 8, 5)).is_some())
}

#[test]
fn the_continuation_is_one_period_back_at_the_smallest_anchored_period() {
    let p = (2, 8, 4);
    let ab: Vec<u32> = [7, 8].iter().copied().cycle().take(8).collect();
    assert_eq!(continuation(ab.len(), |i| ab[i], p), Some(7));
    // One repeat short.
    assert_eq!(continuation(7, |i| ab[i + 1], p), None);
    // Period 3, and a run of one token (period 2 is the smallest that anchors).
    let abc: Vec<u32> = [1, 2, 3].iter().copied().cycle().take(12).collect();
    assert_eq!(continuation(abc.len(), |i| abc[i], p), Some(1));
    let run = [9u32; 8];
    assert_eq!(continuation(run.len(), |i| run[i], p), Some(9));
    // A broken tail.
    let mut broken = ab.clone();
    broken.push(5);
    assert_eq!(continuation(broken.len(), |i| broken[i], p), None);
}

#[test]
fn the_rule_acts_only_outside_think_and_tool_bodies_with_steers_left() {
    let mut a = chat_seq(2, 64);
    a.output_tokens.extend([A, B].iter().cycle().take(8));
    assert_eq!(banned_token(&a), Some(A));
    a.loop_steer_max = 0;
    assert_eq!(banned_token(&a), None);
    a.loop_steer_max = 2;
    a.loop_steers = 2;
    assert_eq!(banned_token(&a), None);
    a.loop_steers = 1;
    a.inside_thinking = true;
    assert_eq!(banned_token(&a), None);
    a.inside_thinking = false;
    a.inside_tool_body = true;
    assert_eq!(banned_token(&a), None);
    a.inside_tool_body = false;
    a.min_tokens = 64;
    assert_eq!(banned_token(&a), None);
    a.min_tokens = 0;
    assert_eq!(banned_token(&a), Some(A));
}

#[test]
fn a_span_that_completes_a_loop_takes_the_host_pipeline() {
    let mut a = chat_seq(2, 64);
    a.output_tokens.extend([A, B].iter().cycle().take(5));
    // Rows 0..2 complete the fourth repeat at row 3's history.
    assert!(span_may_steer(&a, &[B, A, B, 9]));
    assert!(!span_may_steer(&a, &[B, A]));
    assert!(!span_may_steer(&a, &[9, 9, 7, 6]));
    a.loop_steer_max = 0;
    assert!(!span_may_steer(&a, &[B, A, B, 9]));
}

#[test]
fn verify_spans_commit_exactly_what_serial_decode_commits_under_steering() {
    let mut steers = 0;
    for seed in 1..=16 {
        for max in [1, 3, 100] {
            let a = check(seed, max);
            steers += a.loop_steers;
            assert!(a.loop_steers <= max, "seed {seed} max {max}");
        }
    }
    // The stream does fall into loops, and they are steered.
    assert!(steers > 100, "{steers}");
}

#[test]
fn with_steers_left_the_output_never_holds_a_loop() {
    for seed in 1..=8 {
        let a = check(seed, 100);
        assert!(a.loop_steers > 0, "seed {seed}");
        assert!(!holds_a_loop(&a.output_tokens[PRIOR..]), "seed {seed}");
        assert!(!a.finished, "seed {seed}");
    }
}

#[test]
fn without_steering_nothing_changes() {
    for seed in 1..=8 {
        let a = check(seed, 0);
        assert_eq!(a.loop_steers, 0);
        // The loops stay, as without the feature.
        assert!(holds_a_loop(&a.output_tokens[PRIOR..]), "seed {seed}");
    }
}

#[test]
fn the_watchdog_ends_the_response_only_once_the_steers_are_spent() {
    for seed in 1..=8 {
        script(rows(seed, 400));
        let s = stream();
        let draft = |p: usize| argmax_first_wins_f32(&s[p]);
        // Steers left all the way: the armed watchdog never fires.
        let open = speculative(1000, 400, &[4], true, draft);
        assert!(!open.finished, "seed {seed}");
        assert_eq!(open.guard_stop, None, "seed {seed}");
        // Two steers, then the loops are the watchdog's again.
        let spent = speculative(2, 400, &[4], true, draft);
        assert_eq!(spent.loop_steers, 2, "seed {seed}");
        assert!(spent.finished, "seed {seed}");
        assert_eq!(
            spent.guard_stop,
            Some(GUARD_STOP_CONTENT_LOOP),
            "seed {seed}"
        );
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}

#[test]
fn the_batched_verify_picks_what_each_member_picks_under_steering() {
    for seed in 1..=12u64 {
        let ks = [4usize, 3, 8, 1, 2, 4, 5, 3];
        let mut off = vec![0usize];
        for k in ks {
            off.push(off.last().unwrap() + k);
        }
        let stream = rows(seed, off[ks.len()]);
        let gpu: Vec<u32> = stream.iter().map(|r| argmax_first_wins_f32(r)).collect();
        script(stream);
        // Members whose histories end in, near and far from a loop.
        let member = |m: usize| {
            let mut a = chat_seq([0, 1, 3][m % 3], 64);
            let tail = [A, B].iter().copied().cycle().take(m % 9);
            a.output_tokens.extend(tail);
            a
        };
        let s = sched(false);
        let ctx = s.verify_logits_ctx(None, None, None, None);
        let mut want: Vec<ActiveSeq> = (0..ks.len()).map(member).collect();
        let want_picks: Vec<Vec<u32>> = want
            .iter_mut()
            .enumerate()
            .map(|(i, a)| {
                verify_pick_all_with_pipeline(&model(), &gpu[off[i]..off[i + 1]], a, &ctx, off[i])
            })
            .collect();
        let mut got: Vec<ActiveSeq> = (0..ks.len()).map(member).collect();
        let mut refs: Vec<&mut ActiveSeq> = got.iter_mut().collect();
        let got_picks = verify_pick_batch(&model(), &gpu, &off, &mut refs, &ctx);
        assert_eq!(got_picks, want_picks, "seed {seed}");
        SCRIPT.with(|s| *s.borrow_mut() = None);
    }
}
