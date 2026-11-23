// SPDX-License-Identifier: AGPL-3.0-only

//! The pipeline's think stages on Inkling 2's blocks
//! (`SchedLimits::message_open`): `<|end_message|>` is the think end, and it
//! closes every other block too.

use super::run_pipeline;
use crate::scheduler::ActiveSeq;
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::test_support::inkling::{self, END, MESSAGE, THINKING};
use std::sync::Arc;

/// A sentence boundary; the only id that ends one.
const DOT: u32 = 46;
/// Ends mid-word; the only id that does.
const WORD: u32 = 65;

fn sched(inkling: bool) -> SchedCtx {
    let mut s = if inkling {
        inkling::sched()
    } else {
        SchedCtx::for_test()
    };
    s.masks.boundary = Some(Arc::from(
        (0..2048u32).map(|t| t == DOT).collect::<Vec<_>>(),
    ));
    s.masks.mid_word = Some(Arc::from(
        (0..2048u32).map(|t| t == WORD).collect::<Vec<_>>(),
    ));
    s
}

/// The logits one pipeline pass leaves for `a` (all zero before it).
fn after_pipeline(a: &mut ActiveSeq, s: &SchedCtx) -> Vec<f32> {
    let ctx = s.verify_logits_ctx(Some(END), Some(THINKING), None, None);
    let mut logits = vec![0.0f32; 2048];
    assert_eq!(run_pipeline(&mut logits, a, &ctx), None);
    logits
}

/// A turn whose reasoning closed, by the model (`forced` false) or the engine.
fn closed(forced: bool) -> ActiveSeq {
    let (mut a, rx) = inkling::seq(true, Some(64));
    std::mem::forget(rx);
    a.output_tokens = vec![MESSAGE, THINKING, 5, END, MESSAGE];
    a.think_ended = true;
    a.think_force_closed = forced;
    a
}

#[test]
fn after_the_reasoning_closes_every_later_block_may_still_close() {
    let s = sched(true);
    let logits = after_pipeline(&mut closed(false), &s);
    assert_eq!(logits[END as usize], 0.0, "the answer's close stays open");
    assert_eq!(logits[THINKING as usize], 0.0, "the model's own choice");
    // Every other model: both think tokens masked after the close.
    let mut a = closed(false);
    let logits = after_pipeline(&mut a, &sched(false));
    assert_eq!(logits[END as usize], f32::NEG_INFINITY);
    assert_eq!(logits[THINKING as usize], f32::NEG_INFINITY);
}

#[test]
fn an_engine_close_bars_a_second_reasoning_block() {
    // A budget or watchdog close must lead to the answer, not to the model
    // reopening the reasoning it was taken out of.
    let logits = after_pipeline(&mut closed(true), &sched(true));
    assert_eq!(logits[THINKING as usize], f32::NEG_INFINITY);
    assert_eq!(logits[END as usize], 0.0);
}

#[test]
fn the_armed_budget_injects_end_message_at_a_sentence_boundary() {
    let s = sched(true);
    let (mut a, _rx) = inkling::seq(true, Some(4));
    a.output_tokens = vec![MESSAGE, THINKING, 5, 6, WORD];
    a.inside_thinking = true;
    a.thinking_tokens = 4;
    a.force_end_thinking = true;
    // Mid-sentence: deferred.
    let logits = after_pipeline(&mut a, &s);
    assert!(logits.iter().all(|&l| l == 0.0));
    assert_eq!(a.sentence_defer_count, 1);
    // After the sentence ends: only `<|end_message|>` is left.
    a.output_tokens.push(DOT);
    let logits = after_pipeline(&mut a, &s);
    let finite: Vec<usize> = (0..logits.len())
        .filter(|&i| logits[i].is_finite())
        .collect();
    assert_eq!(finite, [END as usize]);
}

#[test]
fn the_reasoning_may_close_after_a_word() {
    // The mid-word `</think>` defer is a Qwen guard: Inkling ends its
    // reasoning where it chooses (as GLM does).
    let (mut a, _rx) = inkling::seq(true, Some(64));
    a.output_tokens = vec![MESSAGE, THINKING, WORD];
    a.inside_thinking = true;
    assert_eq!(after_pipeline(&mut a, &sched(true))[END as usize], 0.0);
    assert_eq!(
        after_pipeline(&mut a, &sched(false))[END as usize],
        f32::NEG_INFINITY,
        "other models keep the defer"
    );
}
