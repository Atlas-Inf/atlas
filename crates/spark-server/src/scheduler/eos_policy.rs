// SPDX-License-Identifier: AGPL-3.0-only

//! Single source of truth for the "may this sampled EOS end the turn"
//! decision. `process_decode_logits` (serial) and `emit_token` (verify/MTP)
//! both call `evaluate` and apply the identical suppressed-side-effects;
//! before this module the verify path lacked the thinking/post-think terms
//! and treated EOS inside a think block differently.

use crate::scheduler::ActiveSeq;
use crate::scheduler::helpers::{
    eos_suppressed_by_thinking, hard_ceiling_hit, tool_eos_escape_enabled,
};
use crate::scheduler::sched_ctx::SchedCtx;

/// Every input the EOS decision reads. Callers fill this BEFORE the sampled
/// token is committed to `output_tokens` (serial decides pre-push; the emit
/// path passes `output_len_before` = `output_tokens.len() - 1` because it
/// pushes earlier in the function — the derived counts then match serial).
pub struct EosInputs {
    /// Is the sampled token an EOS?
    pub is_eos: bool,
    /// `grammar_blocks_stop(...)` result, evaluated by the caller (it mutates
    /// matcher state, so the policy module can't own the call). The caller
    /// evaluates it lazily — only when `is_eos && !eos_escape(&inputs)`.
    pub grammar_blocks_stop: bool,
    pub require_tool_call: bool,
    pub tool_request: bool,
    pub min_tokens: usize,
    /// output_tokens.len() BEFORE this token is committed.
    pub output_len_before: usize,
    pub thinking_tokens: u32,
    pub inside_thinking: bool,
    pub think_ended: bool,
    pub tool_call_completed: bool,
    pub inside_tool_body: bool,
    pub remaining: usize,
    pub seq_len: usize,
    pub max_seq_len: usize,
}

/// The five EOS suppressors plus the Fix-A escape. `suppress()` = "don't end
/// the turn on this EOS".
#[derive(Debug, Default)]
pub struct EosDecision {
    pub grammar: bool,
    pub legacy_tool: bool,
    pub min_tokens: bool,
    pub thinking: bool,
    pub post_think: bool,
    pub eos_escape: bool,
}

pub const POST_THINK_MIN_CONTENT: u32 = 16;

/// Fix-A tool-EOS escape (default off): a completed tool call outside a tool
/// body / thinking lifts grammar suppression so a natural EOS can end the
/// turn. Shared by both paths.
pub fn eos_escape(i: &EosInputs) -> bool {
    tool_eos_escape_enabled() && i.tool_call_completed && !i.inside_tool_body && !i.inside_thinking
}

impl EosDecision {
    pub fn suppress(&self) -> bool {
        self.grammar || self.legacy_tool || self.min_tokens || self.thinking || self.post_think
    }
    /// Thinking is the SOLE suppressor → the watchdog may apply the
    /// implicit-think-close mutation (honor_eos_inside_thinking lane).
    pub fn thinking_sole_suppressor(&self) -> bool {
        self.thinking && !self.grammar && !self.legacy_tool && !self.post_think && !self.min_tokens
    }
}

/// Evaluate every suppressor for a sampled token. Mirrors the serial
/// `process_decode_logits` terms exactly:
///   - grammar: stop-legality, lifted by `eos_escape`. The emit path's
///     historical `(inside_thinking && grammar.is_some())` sub-case is
///     carried by the thinking term instead — identical outcomes, and now
///     ceiling-aware like serial.
///   - legacy: `require_tool_call`.
///   - min_tokens: `output_len_before < min_tokens`.
///   - thinking: `inside_thinking && !hard_ceiling` (§C-2).
///   - post_think: `tools_armed && think_ended && content < 16`
///     (tools_armed = require_tool_call || tool_request).
pub fn evaluate(i: &EosInputs) -> EosDecision {
    let eos_escape = eos_escape(i);
    let grammar = i.is_eos && !eos_escape && i.grammar_blocks_stop;
    let legacy_tool = i.require_tool_call;
    let min_tokens = i.output_len_before < i.min_tokens;
    let hard_ceiling = hard_ceiling_hit(i.remaining, i.seq_len, i.max_seq_len);
    let thinking = eos_suppressed_by_thinking(i.inside_thinking, hard_ceiling);
    let post_think_content = (i.output_len_before as u32).saturating_sub(i.thinking_tokens);
    let tools_armed = i.require_tool_call || i.tool_request;
    let post_think = tools_armed && i.think_ended && post_think_content < POST_THINK_MIN_CONTENT;
    EosDecision {
        grammar,
        legacy_tool,
        min_tokens,
        thinking,
        post_think,
        eos_escape,
    }
}

/// The serial path's implicit-think-close mutation
/// (`honor_eos_inside_thinking`, gated per model): applied identically in
/// both paths when `decision.thinking_sole_suppressor()`.
pub fn apply_implicit_think_close(a: &mut ActiveSeq, sched: &SchedCtx, d: &EosDecision) {
    if d.thinking_sole_suppressor() && sched.watchdog.honor_eos_inside_thinking {
        a.inside_thinking = false;
        a.think_force_closed = true;
        a.force_end_thinking = false;
        a.sentence_defer_count = 0;
        a.consecutive_confident = 0;
        a.in_code_fence = false;
        a.think_ended = true;
        a.think_just_ended = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> EosInputs {
        EosInputs {
            is_eos: true,
            grammar_blocks_stop: false,
            require_tool_call: false,
            tool_request: false,
            min_tokens: 0,
            output_len_before: 20,
            thinking_tokens: 0,
            inside_thinking: false,
            think_ended: true,
            tool_call_completed: false,
            inside_tool_body: false,
            remaining: 100,
            seq_len: 500,
            max_seq_len: 8192,
        }
    }

    /// The serial decision table — `evaluate` is the shared SSOT, so the
    /// same matrix is what verify produces.
    #[test]
    fn eos_decision_table() {
        struct Case {
            name: &'static str,
            f: fn(&mut EosInputs),
            suppress: bool,
            sole_thinking: bool,
        }
        let cases = [
            Case {
                name: "plain EOS",
                f: |_| {},
                suppress: false,
                sole_thinking: false,
            },
            Case {
                name: "grammar blocks",
                f: |i| i.grammar_blocks_stop = true,
                suppress: true,
                sole_thinking: false,
            },
            Case {
                name: "require_tool_call",
                f: |i| i.require_tool_call = true,
                suppress: true,
                sole_thinking: false,
            },
            Case {
                name: "min_tokens",
                f: |i| {
                    i.min_tokens = 25;
                    i.output_len_before = 10;
                },
                suppress: true,
                sole_thinking: false,
            },
            Case {
                name: "inside thinking",
                f: |i| {
                    i.inside_thinking = true;
                    i.think_ended = false;
                },
                suppress: true,
                sole_thinking: true,
            },
            Case {
                name: "inside thinking at hard ceiling",
                f: |i| {
                    i.inside_thinking = true;
                    i.think_ended = false;
                    i.remaining = 0;
                },
                suppress: false,
                sole_thinking: false,
            },
            Case {
                name: "post-think 7 content",
                f: |i| {
                    i.tool_request = true;
                    i.thinking_tokens = 5;
                    i.output_len_before = 12;
                },
                suppress: true,
                sole_thinking: false,
            },
            Case {
                name: "post-think 15 content",
                f: |i| {
                    i.tool_request = true;
                    i.thinking_tokens = 5;
                    i.output_len_before = 20;
                },
                suppress: true,
                sole_thinking: false,
            },
            Case {
                name: "post-think 16 content allowed",
                f: |i| {
                    i.tool_request = true;
                    i.thinking_tokens = 5;
                    i.output_len_before = 21;
                },
                suppress: false,
                sole_thinking: false,
            },
            Case {
                name: "post-think no tools -> allowed",
                f: |i| {
                    i.thinking_tokens = 5;
                    i.output_len_before = 10;
                },
                suppress: false,
                sole_thinking: false,
            },
            Case {
                name: "post-think require_tool_call armed",
                f: |i| {
                    i.require_tool_call = true;
                    i.thinking_tokens = 5;
                    i.output_len_before = 10;
                },
                suppress: true,
                sole_thinking: false,
            },
        ];
        for c in cases {
            let mut i = base();
            (c.f)(&mut i);
            let d = evaluate(&i);
            assert_eq!(d.suppress(), c.suppress, "{}", c.name);
            assert_eq!(d.thinking_sole_suppressor(), c.sole_thinking, "{}", c.name);
        }
    }

    /// Regression: the live_irrelevance_25-2-13 shape — tools armed,
    /// thinking disabled (think_ended holds from the start), 7 content
    /// tokens, sampled EOS. Serial suppressed via post_think; verify did
    /// not before this module.
    #[test]
    fn eos_regression_post_think_suppresses_on_both_paths() {
        let mut i = base();
        i.tool_request = true;
        i.thinking_tokens = 0;
        i.output_len_before = 7;
        let d = evaluate(&i);
        assert!(d.post_think);
        assert!(d.suppress());
    }
}
