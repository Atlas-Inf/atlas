// SPDX-License-Identifier: AGPL-3.0-only

//! The one post-pick `<think>` commit rule.
//!
//! Plain decode (`process_decode_logits`), the commit path (`emit_token`) and
//! the verify pick loop (`verify_pipeline_helper::selection`) all advance the
//! same `<think>` state once a token is picked. They used to carry separate
//! copies that disagreed — fence parity, the thinking-loop watchdog, an end
//! token inside `<think>` (closed by decode under `honor_eos_inside_thinking`,
//! swallowed by emit), `<|im_start|>`, the spontaneous-`<think>` budget — and
//! the verify loop advanced nothing between positions. A token committed
//! through a verify span therefore left different state than the same token
//! decoded alone, and a later pick could differ. Every path now runs these
//! functions, so committing a token advances the state the same way whichever
//! path commits it; `shadow` replays them between verify positions.

use super::*;

mod post_think_eos;
mod shadow;
pub(in crate::scheduler) use post_think_eos::should_suppress_post_think_eos;
pub(in crate::scheduler) use shadow::{PickEffects, SpanShadow};

#[cfg(test)]
mod exact_tests;
#[cfg(test)]
mod span_tests;
#[cfg(test)]
mod tests;

/// The run-scoped inputs of the commit rule, read off either carrier.
#[derive(Debug, Clone, Copy)]
pub(in crate::scheduler) struct CommitEnv {
    pub limits: crate::scheduler::limits::SchedLimits,
    pub watchdog: WatchdogParams,
    pub disable_watchdogs: bool,
    /// The verify loop's replay: no logging (a rejected position must not
    /// report a budget or watchdog event the stream never sees).
    pub quiet: bool,
}

impl CommitEnv {
    pub(in crate::scheduler) fn of(sched: &crate::scheduler::sched_ctx::SchedCtx) -> Self {
        Self {
            limits: sched.limits,
            watchdog: sched.watchdog,
            disable_watchdogs: sched.levers.disable_watchdogs,
            quiet: false,
        }
    }

    pub(in crate::scheduler) fn of_ctx(
        ctx: &crate::scheduler::logit_processors::LogitsContext,
    ) -> Self {
        Self {
            limits: ctx.limits,
            watchdog: ctx.watchdog,
            disable_watchdogs: ctx.sampling.disable_watchdogs,
            quiet: false,
        }
    }
}

/// Control tokens that end the turn before anything else looks at them: the
/// token is recorded and the sequence finishes. `<tool_response>` always (kill
/// switch `ATLAS_TOOL_RESPONSE_STOP`): the model must never write it, and as it
/// is not an end token the cut is named or it would wire "stop". Then
/// `<|im_start|>` ([`im_start_stop`]). Returns true when the turn ended.
pub(in crate::scheduler) fn hard_stop(a: &mut ActiveSeq, tok: u32, env: &CommitEnv) -> bool {
    if tool_response_stop_enabled() && env.limits.tool_response_hard_stop == Some(tok) {
        a.output_tokens.push(tok);
        a.finished = true;
        a.guard_stop = Some(GUARD_STOP_TOOL_RESPONSE);
        if !env.quiet {
            tracing::debug!("<tool_response> hard-stop fired (id={tok}); ending turn");
        }
        return true;
    }
    im_start_stop(a, tok, env)
}

/// `<think>` / `</think>` met OUTSIDE a thinking block consume the token: it
/// was fed to the model but is never recorded. A spontaneous `<think>` opens a
/// block whose budget decays with each thinking-loop watchdog fire, so a model
/// that keeps re-entering is cut sooner each time; a stray `</think>` is
/// skipped, and 50 in a row end the turn (`GUARD_STOP_THINK_SKIP`: it is not an
/// end token, so an unnamed cut would wire "stop"). Returns true when the
/// token was consumed.
pub(in crate::scheduler) fn think_gate(a: &mut ActiveSeq, tok: u32, env: &CommitEnv) -> bool {
    if !a.inside_thinking && a.think_start_token == Some(tok) {
        let decayed = a.spontaneous_think_budget >> a.think_watchdog_fires.min(4);
        a.inside_thinking = true;
        a.think_ended = false;
        a.think_skip_count = 0;
        // Re-entering thinking re-arms the spec-resume guard for the next exit.
        a.post_think_emitted = 0;
        // Floored so the watchdog stays functional.
        a.thinking_budget = Some(decayed.max(8));
        if !env.quiet {
            tracing::debug!(
                fires = a.think_watchdog_fires,
                budget = decayed.max(8),
                "spontaneous <think>: entering thinking mode"
            );
        }
        return true;
    }
    if !a.inside_thinking && a.think_end_token == Some(tok) {
        a.think_skip_count += 1;
        if a.think_skip_count >= 50 {
            a.finished = true;
            a.guard_stop = Some(GUARD_STOP_THINK_SKIP);
            if !env.quiet {
                tracing::debug!(
                    "</think> think-skip watchdog hard-stop fired (50 consecutive strays); \
                     ending turn"
                );
            }
        }
        return true;
    }
    // A real content token ends a run of strays: the guard counts CONSECUTIVE
    // `</think>` (the long-context degeneration), not scattered ones.
    if a.think_ended {
        a.think_skip_count = 0;
    }
    false
}

/// The tool-call bookkeeping of a token past [`think_gate`], outside `<think>`
/// only (a tool tag inside reasoning is spurious): the opener satisfies a
/// required call (the end-token hold and the post-`</think>` pin read it), the
/// tool/parameter body state machine advances (DRY's tool-body zeroing, B1's
/// margin stage, the opener-bias strip read it), and `</tool_call>` completes
/// the call (the EOS-escape gate). Before the token is recorded.
pub(in crate::scheduler) fn tool_state(a: &mut ActiveSeq, tok: u32, env: &CommitEnv) {
    if a.inside_thinking {
        return;
    }
    if a.require_tool_call && a.tool_call_start_token == Some(tok) {
        a.require_tool_call = false;
        a.tool_call_opened = true;
    }
    update_tool_param_state(a, tok, env.quiet);
    if a.tool_call_end_token == Some(tok) {
        a.tool_call_completed = true;
    }
}

/// The KV position serial decode checks row `row` of a verify span at: the
/// span's `rows` rows end at `span_end` (`seq_len` with all of them written),
/// and row `row` is picked with its own input, the rows before it, cached.
pub(in crate::scheduler) fn span_row_position(span_end: usize, rows: usize, row: usize) -> usize {
    span_end.saturating_sub(rows) + row + 1
}

/// Close the thinking block: the model's `</think>` (`forced` = whether a
/// budget/watchdog arm forced it) or an honored end token (`forced` = true).
fn close_thinking(a: &mut ActiveSeq, forced: bool) {
    a.inside_thinking = false;
    a.think_force_closed = forced;
    a.force_end_thinking = false;
    a.sentence_defer_count = 0;
    a.consecutive_confident = 0;
    a.in_code_fence = false;
    a.think_ended = true;
    // One-shot for the next pick: pin to the tool-call opener when a call
    // is required (`PinToToolCallStart`).
    a.think_just_ended = true;
}

/// Advance an open thinking block past `tok` (the caller checked
/// `a.inside_thinking`). `prior` is the number of output tokens before `tok`:
/// the thinking-loop watchdog scans only those, as plain decode always did.
/// Thinking tokens draw down the completion budget like content tokens
/// (`remaining`); `thinking_budget` is the separate per-block cap.
pub(in crate::scheduler) fn advance_thinking(
    a: &mut ActiveSeq,
    tok: u32,
    prior: usize,
    native_glm_eos: bool,
    env: &CommitEnv,
) {
    a.consume_generation_budget();
    if a.think_end_token == Some(tok) {
        let forced = a.force_end_thinking;
        close_thinking(a, forced);
        if !env.quiet {
            tracing::info!(
                "Thinking ended after {} tokens (budget={:?})",
                a.thinking_tokens,
                a.thinking_budget,
            );
        }
        return;
    }
    if native_glm_eos {
        return;
    }
    a.thinking_tokens += 1;
    // ``` parity: F2's confidence stop holds off inside a fenced span (code is
    // near-deterministic, which is not a "done reasoning" signal).
    a.in_code_fence = toggle_code_fence(a.in_code_fence, tok, env.limits.code_fence_token);
    if let Some(budget) = a.thinking_budget
        && a.thinking_tokens >= budget
        && !a.force_end_thinking
    {
        a.force_end_thinking = true;
        a.sentence_defer_count = 0;
        if !env.quiet {
            // Name the budget's SOURCE: a client budget/effort rung is fixed
            // in the request, not on the server.
            tracing::info!(
                source = if a.enable_thinking {
                    "request (client budget/effort; scaled by --max-thinking-budget)"
                } else {
                    "spontaneous <think> (--max-thinking-budget / MODEL.toml)"
                },
                "Thinking budget exhausted ({budget} tokens), arming </think>; \
                 deferring up to {MAX_SENTENCE_DEFER_TOKENS} tokens for sentence boundary"
            );
        }
    }
    // Token-level loop detection inside thinking (the Qwen3.5 fence-narration
    // attractor), well before the budget would cut it.
    if !env.disable_watchdogs
        && env.watchdog.enable_think_loop_watchdog
        && !a.force_end_thinking
        && watchdog_floor_reached(prior.saturating_add(1), a.min_tokens)
        && a.thinking_tokens >= THINK_LOOP_MIN_TOKENS
        && a.thinking_tokens.is_multiple_of(THINK_LOOP_CHECK_STRIDE)
        && detect_thinking_token_loop_with(
            &a.output_tokens[..prior],
            a.repetition_detection,
            env.watchdog,
        )
    {
        a.force_end_thinking = true;
        a.sentence_defer_count = 0;
        a.think_watchdog_fires = a.think_watchdog_fires.saturating_add(1);
        if !env.quiet {
            tracing::warn!(
                thinking_tokens = a.thinking_tokens,
                watchdog_fires = a.think_watchdog_fires,
                "Thinking-loop watchdog fired (period-{THINK_LOOP_PERIOD_MIN}…\
                 {THINK_LOOP_PERIOD_MAX} repeat in tail); forcing </think> early",
            );
        }
    }
}

/// What a picked end token does, decided after the token's think/content step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::scheduler) enum EndToken {
    /// Not one of this request's end tokens.
    No,
    /// The turn ends here; the token is recorded (never streamed).
    Stop,
    /// Held back: discarded, never recorded, decoding goes on. When `<think>`
    /// was the ONLY reason and the model honors that (`[behavior]
    /// honor_eos_inside_thinking`), the block was closed as `</think>` would.
    Suppressed,
}

/// Decide a picked end token. `prior` = output tokens before it; `position`
/// = the KV position the context ceiling is checked at. A hard ceiling (budget
/// spent or context full) always stops, whatever would otherwise hold the end
/// token back.
pub(in crate::scheduler) fn end_token(
    a: &mut ActiveSeq,
    tok: u32,
    prior: usize,
    position: usize,
    native_glm_eos: bool,
    env: &CommitEnv,
) -> EndToken {
    if !a.eos_tokens.contains(&tok) {
        return EndToken::No;
    }
    // ATLAS_TOOL_EOS_ESCAPE: after a completed call (outside a tool body and
    // thinking) the model's own end token may end an auto-mode turn.
    let eos_escape = tool_eos_escape_enabled()
        && a.tool_call_completed
        && !a.inside_tool_body
        && !a.inside_thinking;
    // #192: stop LEGALITY, not `is_terminated()` (an auto trigger grammar
    // never terminates).
    let by_grammar =
        !eos_escape && crate::grammar::grammar_blocks_stop(a.grammar_state.as_mut(), &a.eos_tokens);
    let by_legacy_tool = a.require_tool_call;
    let by_min_tokens = prior < a.min_tokens;
    let hard_ceiling = hard_ceiling_hit(a.remaining, position, env.limits.max_seq_len);
    // GLM may end its turn without closing reasoning.
    let by_thinking =
        eos_suppressed_by_thinking(a.inside_thinking, hard_ceiling) && !native_glm_eos;
    let by_post_think =
        should_suppress_post_think_eos(a, (prior as u32).saturating_sub(a.thinking_tokens));
    let suppress = by_grammar || by_legacy_tool || by_min_tokens || by_thinking || by_post_think;
    if hard_ceiling || !suppress {
        return EndToken::Stop;
    }
    // THE MODEL IS TRYING TO STOP. Discarding the end token while `<think>`
    // is the only thing holding it strands the model with no continuation
    // (Laguna-S-2.1: 'I\nI\nI...' to length), so a model that opts in closes
    // the block instead and gets one clean shot at its answer.
    let thinking_sole =
        by_thinking && !by_grammar && !by_legacy_tool && !by_post_think && !by_min_tokens;
    let close = thinking_sole && env.watchdog.honor_eos_inside_thinking;
    if close {
        close_thinking(a, true);
    }
    if !env.quiet {
        tracing::debug!(
            target: "atlas::eos",
            tok,
            implicit_think_close = close,
            thinking_sole_suppressor = thinking_sole,
            thinking_tokens = a.thinking_tokens,
            by_thinking,
            by_grammar,
            by_legacy_tool,
            by_post_think,
            by_min_tokens,
            "EOS suppressed; model forced to continue"
        );
    }
    EndToken::Suppressed
}

/// A model that opens a new ChatML turn on its own (`<|im_start|>`) ends the
/// turn at the role boundary, whatever grammar or tool obligation would hold
/// an end token back — else the role literal after it streams to the client.
/// Only outside `<think>` (inside it is an end token like the others:
/// discarded or, under `honor_eos_inside_thinking`, an implicit close), only
/// at or past an explicit `min_tokens` floor, and only while it is one of this
/// request's end tokens (`ignore_eos` makes it ordinary). End-token
/// registered, so "stop" is the true finish reason: unnamed by design.
fn im_start_stop(a: &mut ActiveSeq, tok: u32, env: &CommitEnv) -> bool {
    if env.limits.im_start_hard_stop != Some(tok)
        || a.inside_thinking
        || !a.eos_tokens.contains(&tok)
        || a.output_tokens.len() < a.min_tokens
    {
        return false;
    }
    a.output_tokens.push(tok);
    a.finished = true;
    if !env.quiet {
        tracing::debug!("<|im_start|> hard-stop fired (id={tok}); ending turn");
    }
    true
}
