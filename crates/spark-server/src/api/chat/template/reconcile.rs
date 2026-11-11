// SPDX-License-Identifier: AGPL-3.0-only
//! `<think>` span-vs-flag reconciliation (extracted; identical content).

use super::*;

pub(super) fn reconcile_prompt_thinking(
    prompt_tokens: &[u32],
    think_start_token_id: Option<u32>,
    think_end_token_id: Option<u32>,
    enable_thinking: bool,
    thinking_budget: Option<u32>,
    max_thinking_budget: u32,
) -> (bool, Option<u32>) {
    if let Some(think_start) = think_start_token_id {
        // An explicit empty reasoning suffix has already closed thinking before
        // generation (GLM tools). Do not mistake a historical pair followed by
        // another assistant header or prompt text for this terminal boundary.
        if think_end_token_id.is_some_and(|end| prompt_tokens.ends_with(&[think_start, end])) {
            return (false, None);
        }
        let tail = &prompt_tokens[prompt_tokens.len().saturating_sub(8)..];
        let last_start = tail.iter().rposition(|t| *t == think_start);
        let has_unclosed_think = match (last_start, think_end_token_id) {
            (Some(si), Some(end_tok)) => !tail[si + 1..].contains(&end_tok),
            (Some(_), None) => true,
            (None, _) => false,
        };
        if has_unclosed_think && !enable_thinking {
            tracing::info!(
                "Template-forced thinking detected (unclosed \\<think\\> in prompt tail) — \
                 overriding enable_thinking=true with budget={}",
                max_thinking_budget,
            );
            (true, atlas_kernels::thinking_ceiling(max_thinking_budget))
        } else {
            (enable_thinking, thinking_budget)
        }
    } else {
        (enable_thinking, thinking_budget)
    }
}
