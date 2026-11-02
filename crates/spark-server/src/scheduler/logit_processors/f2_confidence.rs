// SPDX-License-Identifier: AGPL-3.0-only

//! F2 confidence-based early-stop arming.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~62-87. While the model is INSIDE `<think>…</think>` and has
//! emitted ≥ 400 thinking tokens, this stage tracks how many
//! consecutive tokens land at top-1 softmax probability ≥ 0.95. Once
//! the configured streak length is reached (see
//! [`crate::scheduler::confidence::confidence_run_step`]), it arms
//! `seq.force_end_thinking` so the downstream injector (stage 5) can
//! force `</think>` at a safe boundary.
//!
//! This stage NEVER modifies `logits` — it only reads them to compute
//! the top-1 prob and updates per-sequence accumulator state.
//! The decision to actually inject `</think>` lives in stage 5; this
//! stage's job is purely to arm the flag.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;
use crate::scheduler::confidence::confidence_run_step;

pub struct F2ConfidenceEarlyStop;

/// Whether top-1 softmax probability is >= 0.95: the stage's whole read of
/// the logits, a pure function of them (a ~248k-term `exp` sum, ~0.5 ms on
/// GB10 — the bulk of a thinking row's host pick).
pub(crate) fn top1_confident(logits: &[f32]) -> bool {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    sum_exp > 0.0 && 1.0 / sum_exp >= 0.95
}

thread_local! {
    /// [`top1_confident`] of the row about to be processed, when the caller
    /// computed it ahead (`verify_pipeline_helper::prepick`). Consumed by the
    /// stage if it runs; cleared by [`with_hint`] either way.
    static HINT: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Run `f` (one row's pipeline) with `hint` as that row's
/// [`top1_confident`]. The hint MUST have been computed on the very logits
/// the stage will see — the dequantized row before any stage touched it,
/// which is what stage 1 reads.
pub(crate) fn with_hint<R>(hint: Option<bool>, f: impl FnOnce() -> R) -> R {
    HINT.with(|c| c.set(hint));
    let r = f();
    HINT.with(|c| c.set(None));
    r
}

impl LogitsProcessor for F2ConfidenceEarlyStop {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if !ctx.sampling.disable_watchdogs
            && a.inside_thinking
            && !a.force_end_thinking
            && a.thinking_tokens >= 400
            && ctx.watchdog.confidence_early_stop
        {
            // A verify span may have computed this row's answer already, on
            // the same untouched logits (`with_hint`); else compute it here.
            let confident = HINT
                .with(std::cell::Cell::take)
                .unwrap_or_else(|| top1_confident(logits));
            let (run, force_end) = confidence_run_step(
                confident,
                a.consecutive_confident,
                ctx.watchdog.confidence_run_length,
            );
            a.consecutive_confident = run;
            if force_end {
                a.force_end_thinking = true;
                a.sentence_defer_count = 0;
                tracing::info!(
                    "Confidence early stop armed: top-1 prob >= 0.95 for {} tokens (after {} thinking tokens){}",
                    ctx.watchdog.confidence_run_length,
                    a.thinking_tokens,
                    if a.in_code_fence {
                        " — deferred until ``` fence closes"
                    } else {
                        " — deferring until next sentence boundary"
                    }
                );
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "f2_confidence_early_stop"
    }

    fn is_argmax_invariant(&self) -> bool {
        // Pure state-update stage: logits are never mutated.
        true
    }
}
