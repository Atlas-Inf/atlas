// SPDX-License-Identifier: AGPL-3.0-only

//! The thinking-budget `</think>` for DFlash verify, which never runs the
//! logits pipeline.

use crate::scheduler::ActiveSeq;
use crate::scheduler::confidence::{
    MAX_SENTENCE_DEFER_TOKENS, THINK_DEFER_ABS_CEILING, THINK_DEFER_BUDGET_FACTOR,
    should_inject_think_end,
};

/// Accept-prefix length of one verified row (`drafts[i] == verified[i]`, the
/// first mismatch is the bonus), with the thinking budget's `</think>` applied.
///
/// A DFlash verify picks the raw argmax, so the pipeline's
/// `ForcedThinkEndInjector` never sees these positions and an exhausted budget
/// was never enforced: a model that kept answering inside `<think>` (a JSON
/// array has no sentence boundary) left its whole answer in the reasoning.
/// This applies the injector's rule to each position the step would emit.
/// Once `force_end_thinking` is armed, the first position whose previous token
/// is a sentence boundary (outside a code fence), or at which the deferral
/// ceilings are reached, becomes `</think>` and is the step's bonus. Positions
/// that defer tick `sentence_defer_count`, one per emitted token.
pub(in crate::scheduler) fn accept_with_forced_think_end(
    a: &mut ActiveSeq,
    think_end: Option<u32>,
    boundary_mask: Option<&[bool]>,
    drafts: &[u32],
    verified: &mut [u32],
) -> usize {
    let mut n = 0usize;
    while n < drafts.len() && n + 1 < verified.len() && drafts[n] == verified[n] {
        n += 1;
    }
    let Some(end) = think_end.filter(|_| a.inside_thinking && a.force_end_thinking) else {
        return n;
    };
    let boundary = |t: u32| boundary_mask.and_then(|m| m.get(t as usize).copied());
    let ceiling = a.thinking_budget.map_or(THINK_DEFER_ABS_CEILING, |b| {
        b.saturating_mul(THINK_DEFER_BUDGET_FACTOR)
    });
    for p in 0..=n {
        let prev = match p {
            0 => a.output_tokens.last().copied(),
            _ => Some(verified[p - 1]),
        };
        if prev == Some(end) {
            // The model closed thinking itself inside this step.
            return n;
        }
        let ahead = p as u32;
        let hard = a.thinking_tokens.saturating_add(ahead) >= ceiling
            || a.sentence_defer_count.saturating_add(ahead) >= MAX_SENTENCE_DEFER_TOKENS;
        let at_boundary = prev.and_then(boundary).unwrap_or(false);
        if should_inject_think_end(true, a.in_code_fence, at_boundary, hard) {
            verified[p] = end;
            return p;
        }
    }
    a.sentence_defer_count = a.sentence_defer_count.saturating_add(n as u32 + 1);
    n
}

#[cfg(test)]
#[path = "think_end_tests.rs"]
mod tests;
