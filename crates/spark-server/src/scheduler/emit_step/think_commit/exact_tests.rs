// SPDX-License-Identifier: AGPL-3.0-only

//! Speculation commits exactly the tokens plain greedy decode commits.
//!
//! One deterministic logits stream (a row per position) is run twice: once
//! through `process_decode_logits` one token at a time, once as verify spans
//! (`verify_pick_all_with_pipeline` + `emit_token`, the K-verdict shape:
//! accept drafts while they equal the picks, then the first pick that
//! differs). Drafts are serial decode's own tokens with every fifth one
//! wrong, so spans accept and reject at every offset. The stream is built to
//! cross everything the commit rule touches inside and around `<think>`:
//! exact top-1 ties, `</think>` (after mid-word tokens too), end tokens and
//! `<|im_start|>` inside thinking, an armed budget deferring to a sentence
//! boundary, F2's confidence arm, fences, a stray `<think>` after the close.
//! Both runs must commit the same tokens and leave the same think state.

use super::shadow::ThinkState;
use crate::scheduler::cancel_test_model::{SCRIPT, Script, TestModel};
use crate::scheduler::decode_logits_step::process_decode_logits;
use crate::scheduler::emit_step::emit_token;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::{THINK_END, THINK_START, test_seq};
use crate::scheduler::types::ActiveSeq;
use crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::sampler::argmax_first_wins_f32;
use std::sync::Arc;

const V: usize = 2048;
const END: u32 = 900;
const IM_START: u32 = 901;
const FENCE: u32 = 960;
const DOT: u32 = 46;
/// Ids 65..=90 end mid-word.
const MID: std::ops::RangeInclusive<u32> = 65..=90;

#[derive(Clone, Copy, Debug)]
struct Case {
    honor_eos_in_think: bool,
    presence: f32,
    budget_after: Option<u32>,
    seed: u64,
}

fn sched(c: Case) -> SchedCtx {
    let masks = crate::scheduler::vocab_masks::VocabMasks {
        boundary: Some((0..V as u32).map(|t| t == DOT || t == 10).collect()),
        mid_word: Some((0..V as u32).map(|t| MID.contains(&t)).collect()),
        ..Default::default()
    };
    let watchdog = crate::scheduler::helpers::WatchdogParams {
        confidence_early_stop: true,
        confidence_run_length: 3,
        honor_eos_inside_thinking: c.honor_eos_in_think,
        ..Default::default()
    };
    let limits = crate::scheduler::limits::SchedLimits {
        im_start_hard_stop: Some(IM_START),
        code_fence_token: Some(FENCE),
        ..crate::scheduler::limits::SchedLimits::NONE
    };
    SchedCtx::new(
        masks,
        Arc::new(crate::scheduler::levers::SchedLevers::defaults()),
        Arc::default(),
        limits,
        watchdog,
    )
}

/// A sequence 398 tokens into its reasoning (F2 starts reading at 400).
fn thinking_seq(c: Case) -> ActiveSeq {
    let (mut a, rx) = test_seq(vec![300, 301], 400, None, 40);
    std::mem::forget(rx);
    a.finished = false;
    a.min_tokens = 0;
    a.eos_tokens = vec![END, IM_START];
    a.inside_thinking = true;
    a.enable_thinking = true;
    a.think_end_token = Some(THINK_END);
    a.think_start_token = Some(THINK_START);
    a.thinking_tokens = 398;
    a.thinking_budget = c.budget_after.map(|n| 398 + n);
    a.spontaneous_think_budget = 64;
    a.presence_penalty = c.presence;
    a
}

/// The stream: small integer logits (exact in BF16) with one designated
/// winner per row, drawn to hit the commit rule's cases.
fn rows(seed: u64, n: usize) -> Vec<Vec<f32>> {
    let mut x = seed;
    let mut next = move || {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (x >> 33) as u32
    };
    (0..n)
        .map(|_| {
            let mut r: Vec<f32> = (0..V).map(|_| (next() % 8) as f32).collect();
            let win = 100 + next() % 600;
            r[(100 + next() % 600) as usize] = 11.0;
            match next() % 64 {
                0..=5 => {
                    // An exact top-1 tie, lower id first.
                    r[win as usize] = 12.0;
                    r[(win + 1 + next() % 300) as usize] = 12.0;
                }
                6..=8 => r[THINK_END as usize] = 12.0,
                9 => r[END as usize] = 12.0,
                10 => r[IM_START as usize] = 12.0,
                11..=15 => r[DOT as usize] = 12.0,
                16..=21 => r[(65 + next() % 26) as usize] = 12.0,
                22 => r[FENCE as usize] = 12.0,
                23 => r[THINK_START as usize] = 12.0,
                24..=45 => r[win as usize] = 40.0,
                _ => r[win as usize] = 12.0,
            }
            r
        })
        .collect()
}

fn at(cursor: usize) {
    SCRIPT.with(|s| s.borrow_mut().as_mut().unwrap().cursor = cursor);
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

/// Plain decode, one token per step. Returns the sequence and the token
/// picked at every position (the model is fed each, recorded or not).
fn serial(c: Case, n: usize) -> (ActiveSeq, Vec<u32>) {
    let s = sched(c);
    let mut batch = vec![thinking_seq(c)];
    let mut picks = Vec::new();
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
            &s,
        );
        picks.push(batch[0].last_token);
        if batch[0].finished {
            break;
        }
    }
    (batch.pop().unwrap(), picks)
}

/// Verify spans of `ks` rows (cycled) over the same stream.
fn speculative(c: Case, n: usize, reference: &[u32], ks: &[usize]) -> ActiveSeq {
    let s = sched(c);
    let ctx = s.verify_logits_ctx(Some(THINK_END), Some(THINK_START), None, None);
    let stream = SCRIPT.with(|s| s.borrow().as_ref().unwrap().rows.clone());
    let mut a = thinking_seq(c);
    let (mut p, mut step) = (0usize, 0usize);
    while p < n && !a.finished {
        let k = ks[step % ks.len()].min(n - p);
        step += 1;
        at(p);
        let gpu: Vec<u32> = (0..k)
            .map(|i| argmax_first_wins_f32(&stream[p + i]))
            .collect();
        let drafts: Vec<u32> = (0..k - 1)
            .map(|i| {
                let d = reference.get(p + i).copied().unwrap_or(0);
                if (p + i) % 5 == 4 { d ^ 1 } else { d }
            })
            .collect();
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

fn check(c: Case) {
    const N: usize = 160;
    SCRIPT.with(|s| {
        *s.borrow_mut() = Some(Script {
            rows: rows(c.seed, N),
            cursor: 0,
        })
    });
    let (want, reference) = serial(c, N);
    for ks in [&[1usize][..], &[2], &[4], &[5, 2, 3, 1], &[3, 6]] {
        let got = speculative(c, reference.len(), &reference, ks);
        assert_eq!(got.output_tokens, want.output_tokens, "{c:?} K={ks:?}");
        assert_eq!(
            ThinkState::of(&got).picks_only(),
            ThinkState::of(&want).picks_only(),
            "{c:?} K={ks:?}"
        );
    }
    SCRIPT.with(|s| *s.borrow_mut() = None);
}

#[test]
fn verify_spans_commit_exactly_what_serial_decode_commits() {
    for seed in 1..=24 {
        for honor_eos_in_think in [true, false] {
            for (presence, budget_after) in [(0.0, None), (0.0, Some(6)), (0.5, Some(12))] {
                check(Case {
                    honor_eos_in_think,
                    presence,
                    budget_after,
                    seed,
                });
            }
        }
    }
}

#[test]
fn the_stream_reaches_the_cases_it_is_built_for() {
    // Guard against a stream that silently stops exercising the rule: over
    // the seeds, serial decode must close thinking by `</think>`, close it by
    // an honored end token, arm the budget and F2, and decode content after.
    let (mut closed, mut armed, mut content) = (0, 0, 0);
    for seed in 1..=24 {
        let c = Case {
            honor_eos_in_think: true,
            presence: 0.0,
            budget_after: Some(6),
            seed,
        };
        SCRIPT.with(|s| {
            *s.borrow_mut() = Some(Script {
                rows: rows(seed, 160),
                cursor: 0,
            })
        });
        let (a, _) = serial(c, 160);
        closed += usize::from(a.think_ended);
        armed += usize::from(a.think_force_closed);
        content += usize::from(a.content_tokens > 8);
    }
    SCRIPT.with(|s| *s.borrow_mut() = None);
    assert!(
        closed > 12 && armed > 4 && content > 8,
        "{closed} {armed} {content}"
    );
}
