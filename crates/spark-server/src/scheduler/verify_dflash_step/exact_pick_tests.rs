// SPDX-License-Identifier: AGPL-3.0-only

//! Under an exact verify (`Model::verify_bit_exact`) a DFlash step commits
//! exactly the tokens plain serial decode commits from the same logits
//! (`exact_picks`).
//!
//! One deterministic logits stream (a row per position) is decoded twice:
//! one token at a time through `process_decode_logits`, and as DFlash steps
//! of γ+1 rows through `verify_dflash_tail` on the raw GPU argmax the verify
//! returns. The drafter is ideal (each draft is its row's raw argmax), which
//! is exactly the case the raw lane gets wrong: a stateful stage serial
//! decode runs per token arms mid-block and closes the reasoning where the
//! raw argmax goes on. Each stream is built to cross one stage inside a
//! block: the thinking budget, F2's confidence streak, max_tokens. With the
//! exact verify the two runs must agree token for token and state for
//! state; the raw lane (exact verify off) must still show the old gap where
//! a stage closes `<think>`, or the stream is not exercising it.

use super::*;
use crate::scheduler::cancel_test_model::{EXACT_VERIFY, SCRIPT, Script, TestModel};
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{THINK_END, THINK_START, test_seq};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::argmax_first_wins_f32;
use std::sync::Arc;

const V: usize = 2048;
const DOT: u32 = 46;
const END: u32 = 900;

fn sched() -> SchedCtx {
    let masks = crate::scheduler::vocab_masks::VocabMasks {
        boundary: Some((0..V as u32).map(|t| t == DOT || t == 10).collect()),
        ..Default::default()
    };
    let watchdog = crate::scheduler::helpers::WatchdogParams {
        confidence_early_stop: true,
        confidence_run_length: 3,
        ..Default::default()
    };
    SchedCtx::new(
        masks,
        Arc::new(crate::scheduler::levers::SchedLevers::defaults()),
        Arc::default(),
        crate::scheduler::limits::SchedLimits::NONE,
        watchdog,
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

/// How a test sequence starts.
#[derive(Clone, Copy, Debug)]
struct Start {
    inside_thinking: bool,
    thinking_tokens: u32,
    budget: Option<u32>,
    remaining: usize,
}

fn seq(s: Start) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![300, 301], s.remaining, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    a.temperature = 0.0;
    a.eos_tokens = vec![END];
    a.enable_thinking = true;
    a.inside_thinking = s.inside_thinking;
    a.think_ended = !s.inside_thinking;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.thinking_tokens = s.thinking_tokens;
    a.thinking_budget = s.budget;
    a.spontaneous_think_budget = 64;
    a
}

/// A row whose argmax is `tok` at `top` over a flat floor of small integers
/// (BF16-exact; the floor varies so no two rows are alike).
fn row(tok: u32, top: f32, salt: usize) -> Vec<f32> {
    let mut r: Vec<f32> = (0..V).map(|i| ((i * 7 + salt * 13) % 5) as f32).collect();
    r[tok as usize] = top;
    r
}

fn script(rows: Vec<Vec<f32>>) {
    SCRIPT.with(|s| *s.borrow_mut() = Some(Script { rows, cursor: 0 }));
}

fn at(cursor: usize) {
    SCRIPT.with(|s| s.borrow_mut().as_mut().unwrap().cursor = cursor);
}

/// Plain decode, one token per step, over the scripted stream.
fn serial(s: Start, n: usize) -> ActiveSeq {
    let sc = sched();
    let mut batch = vec![seq(s)];
    for t in 0..n {
        at(t);
        process_decode_logits(
            &model(),
            &mut batch,
            DevicePtr::NULL,
            std::time::Instant::now(),
            Some(THINK_END),
            Some(THINK_START),
            None,
            None,
            false,
            &sc,
        );
        if batch[0].finished {
            break;
        }
    }
    batch.pop().unwrap()
}

/// DFlash steps of `ks` rows (cycled) over the same stream, each draft its
/// row's raw argmax, every step through `verify_dflash_tail` on the raw
/// lane, `exact` answering `Model::verify_bit_exact`.
fn dflash(s: Start, n: usize, ks: &[usize], exact: bool) -> ActiveSeq {
    EXACT_VERIFY.with(|e| e.set(exact));
    let sc = sched();
    let ctx = sc.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
    let stream = SCRIPT.with(|s| s.borrow().as_ref().unwrap().rows.clone());
    let m = model();
    let mut a = seq(s);
    let (mut p, mut step) = (0usize, 0usize);
    while p < n && !a.finished {
        let k = ks[step % ks.len()].min(n - p);
        step += 1;
        at(p);
        let gpu: Vec<u32> = (0..k)
            .map(|i| argmax_first_wins_f32(&stream[p + i]))
            .collect();
        let drafts = gpu[..k - 1].to_vec();
        let mut tokens = vec![a.last_token];
        tokens.extend_from_slice(&drafts);
        let before = a.seq.seq_len;
        a.seq.tokens.extend_from_slice(&tokens);
        a.seq.seq_len += tokens.len();
        verify_dflash_tail(
            &m,
            &mut a,
            &sc,
            &drafts,
            &[],
            drafts.len(),
            &ctx,
            true,
            &tokens,
            gpu,
            false,
            0.0,
            true,
            true,
        );
        assert!(a.engine_error.is_none(), "{:?}", a.engine_error);
        p += a.seq.seq_len - before;
    }
    EXACT_VERIFY.with(|e| e.set(false));
    a
}

fn same_state(got: &ActiveSeq, want: &ActiveSeq, what: &str) {
    assert_eq!(got.output_tokens, want.output_tokens, "{what}");
    assert_eq!(got.inside_thinking, want.inside_thinking, "{what}");
    assert_eq!(got.think_ended, want.think_ended, "{what}");
    assert_eq!(got.think_force_closed, want.think_force_closed, "{what}");
    assert_eq!(got.thinking_tokens, want.thinking_tokens, "{what}");
    assert_eq!(got.force_end_thinking, want.force_end_thinking, "{what}");
    assert_eq!(
        got.sentence_defer_count, want.sentence_defer_count,
        "{what}"
    );
    assert_eq!(
        got.consecutive_confident, want.consecutive_confident,
        "{what}"
    );
    assert_eq!(got.remaining, want.remaining, "{what}");
    assert_eq!(got.finished, want.finished, "{what}");
}

/// Run `rows` both ways at the DFlash2 width (γ = 8) and at mixed widths:
/// exact equals serial; returns serial and the raw lane at γ = 8.
fn check(s: Start, rows: Vec<Vec<f32>>) -> (ActiveSeq, ActiveSeq) {
    let n = rows.len();
    script(rows);
    let want = serial(s, n);
    for ks in [&[9usize][..], &[9, 3, 5], &[16], &[2]] {
        let got = dflash(s, n, ks, true);
        same_state(&got, &want, &format!("{s:?} K={ks:?}"));
    }
    let raw = dflash(s, n, &[9], false);
    SCRIPT.with(|s| *s.borrow_mut() = None);
    (want, raw)
}

/// Ordinary reasoning tokens with a sentence end at `dots`, then `tail`.
fn stream(n: usize, top: f32, dots: &[usize]) -> Vec<Vec<f32>> {
    (0..n)
        .map(|i| {
            let tok = if dots.contains(&i) {
                DOT
            } else {
                200 + i as u32
            };
            row(tok, top, i)
        })
        .collect()
}

/// The budget is reached at the block's fourth row: serial decode arms
/// `</think>` there and closes at the next sentence boundary (the row after
/// the dot at 6), inside the first γ = 8 block. The raw lane, armed only at
/// a step's start, reasons on through the block.
#[test]
fn a_block_crossing_the_thinking_budget_closes_where_serial_decode_does() {
    let s = Start {
        inside_thinking: true,
        thinking_tokens: 10,
        budget: Some(14),
        remaining: 400,
    };
    let (want, raw) = check(s, stream(30, 12.0, &[6, 20]));
    assert!(want.think_ended && want.think_force_closed, "serial closes");
    let close = want.output_tokens.iter().position(|&t| t == THINK_END);
    assert_eq!(close, Some(2 + 7), "closed after the dot at row 6");
    assert_ne!(raw.output_tokens, want.output_tokens, "the raw lane's gap");
}

/// F2's confidence stop reads from 400 thinking tokens: two rows in, its
/// streak of three confident rows completes mid-block and arms `</think>`,
/// which closes after the next sentence end.
#[test]
fn f2_arming_mid_block_closes_where_serial_decode_does() {
    let s = Start {
        inside_thinking: true,
        thinking_tokens: 398,
        budget: None,
        remaining: 400,
    };
    let (want, raw) = check(s, stream(30, 40.0, &[7, 21]));
    assert!(want.think_ended && want.think_force_closed, "F2 closes");
    assert_ne!(raw.output_tokens, want.output_tokens, "the raw lane's gap");
}

/// max_tokens runs out four tokens into a block: the step stops there, as
/// serial decode does, whatever rows remain.
#[test]
fn max_tokens_mid_block_stops_where_serial_decode_does() {
    for inside_thinking in [false, true] {
        let s = Start {
            inside_thinking,
            thinking_tokens: 10,
            budget: None,
            remaining: 4,
        };
        let (want, _) = check(s, stream(20, 12.0, &[3]));
        assert!(want.finished, "max_tokens finishes");
        assert_eq!(want.output_tokens.len(), 2 + 4);
    }
}

/// An end token mid-block outside thinking ends the turn there.
#[test]
fn an_end_token_mid_block_stops_where_serial_decode_does() {
    let s = Start {
        inside_thinking: false,
        thinking_tokens: 0,
        budget: None,
        remaining: 400,
    };
    let mut rows = stream(20, 12.0, &[]);
    rows[5] = row(END, 12.0, 5);
    let (want, _) = check(s, rows);
    assert!(want.finished);
}

/// The gate: the raw lane unless the verify is exact.
#[test]
fn exact_picks_needs_the_raw_lane_and_an_exact_verify() {
    assert!(exact_picks(true, true));
    assert!(!exact_picks(true, false));
    assert!(!exact_picks(false, true));
    assert!(!exact_picks(false, false));
}

/// T>0 with a seed: each row is drawn from its processed logits with the
/// position's seed, so a step commits the draws serial decode makes.
#[test]
fn seeded_sampling_commits_serial_decode_draws() {
    for inside_thinking in [true, false] {
        let s = Start {
            inside_thinking,
            thinking_tokens: 10,
            budget: Some(20),
            remaining: 400,
        };
        let rows: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                let mut r = row(200 + i as u32, 6.0, i);
                r[300 + i % 7] = 5.5;
                r[DOT as usize] = if i % 9 == 4 { 6.5 } else { 3.0 };
                r
            })
            .collect();
        let n = rows.len();
        script(rows);
        let sampled = |mut a: ActiveSeq| {
            a.temperature = 0.8;
            a.seed = Some(7);
            a
        };
        let sc = sched();
        let mut batch = vec![sampled(seq(s))];
        for t in 0..n {
            at(t);
            process_decode_logits(
                &model(),
                &mut batch,
                DevicePtr::NULL,
                std::time::Instant::now(),
                Some(THINK_END),
                Some(THINK_START),
                None,
                None,
                false,
                &sc,
            );
            if batch[0].finished {
                break;
            }
        }
        let want = batch.pop().unwrap();
        EXACT_VERIFY.with(|e| e.set(true));
        let ctx = sc.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
        let stream = SCRIPT.with(|s| s.borrow().as_ref().unwrap().rows.clone());
        let m = model();
        let mut a = sampled(seq(s));
        let mut p = 0usize;
        while p < n && !a.finished {
            let k = 9.min(n - p);
            at(p);
            let gpu: Vec<u32> = (0..k)
                .map(|i| argmax_first_wins_f32(&stream[p + i]))
                .collect();
            // Drafts: serial decode's own draws (accepted), one wrong row.
            let mut drafts: Vec<u32> = (0..k - 1)
                .map(|i| want.output_tokens.get(2 + p + i).copied().unwrap_or(0))
                .collect();
            if let Some(d) = drafts.get_mut(5) {
                *d ^= 1;
            }
            let mut tokens = vec![a.last_token];
            tokens.extend_from_slice(&drafts);
            let before = a.seq.seq_len;
            a.seq.tokens.extend_from_slice(&tokens);
            a.seq.seq_len += tokens.len();
            verify_dflash_tail(
                &m,
                &mut a,
                &sc,
                &drafts,
                &[],
                drafts.len(),
                &ctx,
                true,
                &tokens,
                gpu,
                false,
                0.0,
                true,
                true,
            );
            p += a.seq.seq_len - before;
        }
        EXACT_VERIFY.with(|e| e.set(false));
        SCRIPT.with(|s| *s.borrow_mut() = None);
        same_state(
            &a,
            &want,
            &format!("sampled inside_thinking={inside_thinking}"),
        );
        // The draws are not all the argmax, or this proves nothing.
        assert!(
            (0..n.min(want.output_tokens.len() - 2))
                .any(|i| want.output_tokens[2 + i] != 200 + i as u32),
            "serial sampled the argmax everywhere"
        );
    }
}
