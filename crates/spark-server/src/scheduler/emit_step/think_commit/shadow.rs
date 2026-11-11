// SPDX-License-Identifier: AGPL-3.0-only

//! The verify pick loop's view of a span.
//!
//! Position `i` of a verify span must be picked against the state serial
//! decode would hold after committing positions `0..i`. The loop used to pick
//! every position against the state at the span's start: a `</think>` at
//! position 2 left positions 3.. masked and floored as if still thinking, the
//! mid-word and sentence-boundary lookbacks read a stale last token, and the
//! pipeline's own counters (F2's confidence streak, the `</think>` deferral)
//! ticked once per VERIFIED position, rejected ones included, and were never
//! restored.
//!
//! [`SpanShadow`] commits each pick to the live sequence through the same
//! functions the commit path runs (on accept the pick IS the draft the next
//! position was computed on; past the first mismatch nothing is committed),
//! then restores the sequence exactly. What each position's pipeline wrote is
//! kept in [`PickEffects`], keyed by the pick, and `emit_token` applies it as
//! that token is actually committed — so the state advances once per committed
//! token, as in serial decode, and never for a rejected position.

use super::*;
use std::collections::VecDeque;

/// Everything the commit rule and the pick pipeline write, captured whole:
/// the `<think>` state, and the tool-call state ([`tool_state`]) the pick
/// stages read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::scheduler) struct ThinkState {
    inside_thinking: bool,
    think_ended: bool,
    think_just_ended: bool,
    thinking_tokens: u32,
    thinking_budget: Option<u32>,
    force_end_thinking: bool,
    think_force_closed: bool,
    sentence_defer_count: u32,
    consecutive_confident: u32,
    in_code_fence: bool,
    think_skip_count: u32,
    think_watchdog_fires: u32,
    loop_steers: u32,
    post_think_emitted: u32,
    remaining: usize,
    finished: bool,
    guard_stop: Option<&'static str>,
    require_tool_call: bool,
    tool_call_opened: bool,
    tool_call_completed: bool,
    inside_tool_body: bool,
    tool_body_streak_tokens: u32,
    inside_parameter_body: bool,
    param_body_chars_emitted: u32,
    param_close_pending: u8,
}

impl ThinkState {
    pub(in crate::scheduler) fn of(a: &ActiveSeq) -> Self {
        Self {
            inside_thinking: a.inside_thinking,
            think_ended: a.think_ended,
            think_just_ended: a.think_just_ended,
            thinking_tokens: a.thinking_tokens,
            thinking_budget: a.thinking_budget,
            force_end_thinking: a.force_end_thinking,
            think_force_closed: a.think_force_closed,
            sentence_defer_count: a.sentence_defer_count,
            consecutive_confident: a.consecutive_confident,
            in_code_fence: a.in_code_fence,
            think_skip_count: a.think_skip_count,
            think_watchdog_fires: a.think_watchdog_fires,
            loop_steers: a.loop_steers,
            post_think_emitted: a.post_think_emitted,
            remaining: a.remaining,
            finished: a.finished,
            guard_stop: a.guard_stop,
            require_tool_call: a.require_tool_call,
            tool_call_opened: a.tool_call_opened,
            tool_call_completed: a.tool_call_completed,
            inside_tool_body: a.inside_tool_body,
            tool_body_streak_tokens: a.tool_body_streak_tokens,
            inside_parameter_body: a.inside_parameter_body,
            param_body_chars_emitted: a.param_body_chars_emitted,
            param_close_pending: a.param_close_pending,
        }
    }

    /// Without `post_think_emitted`: the spec-resume gate's count of tokens
    /// emitted since `</think>`, kept by the commit path only. It decides when
    /// speculation resumes, never a pick.
    #[cfg(test)]
    pub(in crate::scheduler) fn picks_only(self) -> Self {
        Self {
            post_think_emitted: 0,
            ..self
        }
    }

    fn restore(self, a: &mut ActiveSeq) {
        a.inside_thinking = self.inside_thinking;
        a.think_ended = self.think_ended;
        a.think_just_ended = self.think_just_ended;
        a.thinking_tokens = self.thinking_tokens;
        a.thinking_budget = self.thinking_budget;
        a.force_end_thinking = self.force_end_thinking;
        a.think_force_closed = self.think_force_closed;
        a.sentence_defer_count = self.sentence_defer_count;
        a.consecutive_confident = self.consecutive_confident;
        a.in_code_fence = self.in_code_fence;
        a.think_skip_count = self.think_skip_count;
        a.think_watchdog_fires = self.think_watchdog_fires;
        a.loop_steers = self.loop_steers;
        a.post_think_emitted = self.post_think_emitted;
        a.remaining = self.remaining;
        a.finished = self.finished;
        a.guard_stop = self.guard_stop;
        a.require_tool_call = self.require_tool_call;
        a.tool_call_opened = self.tool_call_opened;
        a.tool_call_completed = self.tool_call_completed;
        a.inside_tool_body = self.inside_tool_body;
        a.tool_body_streak_tokens = self.tool_body_streak_tokens;
        a.inside_parameter_body = self.inside_parameter_body;
        a.param_body_chars_emitted = self.param_body_chars_emitted;
        a.param_close_pending = self.param_close_pending;
    }
}

/// What one position's pipeline left in the fields it writes (F2's streak and
/// arm, the `</think>` deferral counter), for the pick it produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PickEffect {
    tok: u32,
    consecutive_confident: u32,
    force_end_thinking: bool,
    sentence_defer_count: u32,
}

/// The current verify span's [`PickEffect`]s, in position order.
#[derive(Debug, Default)]
pub(in crate::scheduler) struct PickEffects(VecDeque<PickEffect>);

impl PickEffects {
    /// Drop whatever a previous span left (picks never committed).
    pub(in crate::scheduler) fn clear(a: &mut ActiveSeq) {
        a.spec_adapt.pick_effects.0.clear();
    }

    /// Record what the pipeline just did for `tok` at the next position.
    fn record(a: &mut ActiveSeq, tok: u32) {
        let effect = PickEffect {
            tok,
            consecutive_confident: a.consecutive_confident,
            force_end_thinking: a.force_end_thinking,
            sentence_defer_count: a.sentence_defer_count,
        };
        a.spec_adapt.pick_effects.0.push_back(effect);
    }

    /// Apply the pipeline effect of the position `tok` is committed from: the
    /// front entry when it was picked as `tok`. A token no verify picked ends
    /// the span (its leftovers are never committed) and changes nothing.
    pub(in crate::scheduler) fn apply(a: &mut ActiveSeq, tok: u32) {
        let q = &mut a.spec_adapt.pick_effects.0;
        let Some(e) = q.pop_front().filter(|e| e.tok == tok) else {
            q.clear();
            return;
        };
        a.consecutive_confident = e.consecutive_confident;
        a.force_end_thinking = e.force_end_thinking;
        a.sentence_defer_count = e.sentence_defer_count;
    }
}

/// One verify span's replay of the commit rule on the live sequence.
pub(in crate::scheduler) struct SpanShadow {
    state: ThinkState,
    output_len: usize,
    env: CommitEnv,
    /// `seq_len` with every row of the span written.
    span_end: usize,
    rows: usize,
}

impl SpanShadow {
    /// Start a span of `rows` rows (all in the cache): capture the sequence
    /// and forget any earlier span's picks.
    pub(in crate::scheduler) fn begin(a: &mut ActiveSeq, env: CommitEnv, rows: usize) -> Self {
        PickEffects::clear(a);
        Self {
            state: ThinkState::of(a),
            output_len: a.output_tokens.len(),
            env: CommitEnv { quiet: true, ..env },
            span_end: a.seq.seq_len,
            rows,
        }
    }

    /// Record the pipeline effects of row `row`'s pick, then commit it so the
    /// next row sees the state serial decode would, the ceilings checked at
    /// the row's own position ([`span_row_position`]). Returns false when the
    /// grammar refused the pick (speculation stops there; the commit path
    /// decides what the refusal means). The grammar matcher's advances are
    /// the caller's to roll back.
    pub(in crate::scheduler) fn pick(&self, a: &mut ActiveSeq, tok: u32, row: usize) -> bool {
        PickEffects::record(a, tok);
        if row + 1 >= self.rows || a.finished {
            return true;
        }
        // The commit path counts this row's steer first (`emit_token`).
        crate::scheduler::loop_steer::note_commit(a);
        let env = &self.env;
        let position = span_row_position(self.span_end, self.rows, row);
        if hard_stop(a, tok, env) {
            return true;
        }
        crate::scheduler::first_token_thinking::apply_native_tool_boundary(a, tok, env);
        if think_gate(a, tok, env) {
            return true;
        }
        let thinking = a.inside_thinking;
        tool_state(a, tok, env);
        if !thinking
            && let Some(gs) = a.grammar_state.as_mut()
            && !gs.accept_token(tok)
        {
            return false;
        }
        let prior = a.output_tokens.len();
        a.output_tokens.push(tok);
        let native_glm_eos = crate::glm_tool_boundary::native_eos_while_thinking(
            env.limits.glm_tool_boundary,
            thinking,
            tok,
            &a.eos_tokens,
        );
        if thinking {
            advance_thinking(a, tok, prior, native_glm_eos, env);
        } else {
            a.consume_generation_budget();
            a.think_just_ended = false;
        }
        match end_token(a, tok, prior, position, native_glm_eos, env) {
            EndToken::Stop => a.finished = true,
            // Emission returns here, and never at a ceiling (it always stops).
            EndToken::Suppressed => {
                a.output_tokens.pop();
            }
            EndToken::No => {
                if hard_ceiling_hit(a.remaining, position, env.limits.max_seq_len) {
                    a.finished = true;
                }
            }
        }
        true
    }

    /// Restore the sequence to the span's start; the recorded effects stay
    /// for the commit path.
    pub(in crate::scheduler) fn end(self, a: &mut ActiveSeq) {
        a.output_tokens.truncate(self.output_len);
        self.state.restore(a);
    }
}
