// SPDX-License-Identifier: AGPL-3.0-only

//! Post-`</think>` symmetric mask for `</think>` and `<think>`.
//!
//! Ported byte-for-byte from `decode_logits_seq::process_seq_logits`
//! lines ~120-149. After thinking has ended (`seq.think_ended == true`),
//! mask BOTH the close token (`think_end_token` from ctx) and the
//! open token (`seq.think_start_token`) so the model cannot
//! degenerate into `</think></think>…` loops OR re-enter
//! `<think>` mid-response (F9, 2026-04-26).
//!
//! arXiv evidence (s1 / DeepSeek-R1 / Qwen3 / Production Repetition):
//! masking dominates penalty stacking for re-entry prevention.
//!
//! Inkling 2 (`SchedLimits::message_open`): its think end closes every
//! block, so it is never masked; its think start is masked only after an
//! engine close (budget or watchdog), so the close leads to the answer.

use super::{LogitsContext, LogitsProcessor, ProcessorOutcome};
use crate::scheduler::ActiveSeq;

pub struct PostCloseThinkMask;

impl LogitsProcessor for PostCloseThinkMask {
    fn apply(
        &self,
        logits: &mut [f32],
        a: &mut ActiveSeq,
        ctx: &LogitsContext,
    ) -> ProcessorOutcome {
        if a.think_ended && ctx.limits.message_open.is_some() {
            // The think end closes every block, so it stays open; only after
            // the engine closed the reasoning is a second one barred, so the
            // close leads to the answer, not back into the reasoning.
            if a.think_force_closed
                && let Some(start_idx) = a.think_start_token.map(|t| t as usize)
                && start_idx < logits.len()
            {
                logits[start_idx] = f32::NEG_INFINITY;
            }
        } else if a.think_ended {
            if let Some(end_tok) = ctx.think_end_token {
                let end_idx = end_tok as usize;
                if end_idx < logits.len() {
                    logits[end_idx] = f32::NEG_INFINITY;
                }
            }
            if let Some(start_tok) = a.think_start_token {
                let start_idx = start_tok as usize;
                if start_idx < logits.len() {
                    logits[start_idx] = f32::NEG_INFINITY;
                }
            }
        }
        ProcessorOutcome::Continue
    }

    fn name(&self) -> &'static str {
        "post_close_think_mask"
    }
}
