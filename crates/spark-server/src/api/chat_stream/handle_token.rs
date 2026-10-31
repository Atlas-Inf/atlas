// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Token` / `StreamEvent::TokenWithLogprobs` arm of the
// streaming `flat_map` closure (originally ~672 LoC at the top of the
// `chat_stream::chat_completions_stream` body).
//
// Returns the neutral stream deltas produced for this single token.
// The caller encodes them into wire SSE events at the surface seam
// (`openai::delta_to_chunk_events`).

use crate::ir::StreamDelta;
use crate::literal_xml::{first_marker_outside_literal, state_after};
use crate::tool_parser;

use super::super::sanitizer::sanitize_content_chunk_with_literal_state;
use super::super::stream_guards::{bump_f12_tool_call_count, check_loop_watchdog};
use super::ctx::StreamCtx;
use super::state::StreamState;
use super::strip::{
    maybe_log_decode_trace, strip_all_preserving_boundary, strip_preserving_boundary,
};
use super::tool_handlers::{
    handle_complete_tool_call, handle_tool_call_args_fragment, handle_tool_call_delta,
    handle_tool_call_end, handle_tool_call_start,
};

type DeltaVec = Vec<StreamDelta>;

#[path = "terminal_token.rs"]
mod terminal_token;

/// Maximum consecutive tokens the stream may spend with
/// `state.suppressing_param_leak == true` (sanitizer holding content
/// because of an orphan `<parameter=` / `<tool_call>` opener without
/// a matching close). When the model degenerates into a doom-loop of
/// partial-envelope leakage — observed 2026-05-24 on
/// opencode-hotfix.jsonl seq=10: 8192 tokens emitted after Atlas
/// rejected a `write({})` call, all suppressed by the sanitizer, no
/// content-loop watchdog fire (the period exceeded 64) — this
/// threshold ends the stream cleanly instead of burning to
/// `max_tokens=8192`. 256 tokens is enough headroom for legitimately
/// long tool-call bodies that take many tokens to close (long
/// `content` field on a `write` call) while bounding worst-case
/// wasted decode at ~10s @ 30 tok/s.
const MAX_SUPPRESS_STREAK_TOKENS: u32 = 256;

/// Drop a delta that is nothing but a bare role literal (`user` /
/// `assistant` / `tool`) — a Qwen3.5/3.6 hallucination leak, companion to
/// the scheduler-side `<|im_start|>` hard-stop.
///
/// `inside_tool_call` MUST be true whenever the streaming tool-call
/// detector is mid-body (between `<tool_call>` and its close). Issue #222:
/// for a `tool_*`-prefixed tool name (`tool_search`, `tool_call`,
/// `tool_describe`) the byte-level BPE tokenizer emits a standalone `tool`
/// token as the leading fragment of the NAME. Without the guard this strip
/// clears that fragment, and the detector reassembles the name from the
/// remainder (`_search`), truncating the streamed tool-call name by exactly
/// `len("tool") == 4` chars. Non-streaming was unaffected because it parses
/// the whole buffer at once. The guard confines the strip to genuine
/// content leaks (no tool call in flight).
pub(super) fn strip_bare_role_literal(delta: &mut String, inside_tool_call: bool) {
    if inside_tool_call {
        return;
    }
    let trimmed = delta.trim();
    if delta.len() < 20 && matches!(trimmed, "user" | "assistant" | "tool") {
        tracing::debug!("role-literal strip: dropped bare '{trimmed}' delta");
        delta.clear();
    }
}

/// Process one token. Returns the stream deltas to forward to the
/// client (empty `Vec` is valid).
///
/// Thin wrapper around [`handle_token_inner`] that runs the
/// orphan-suppression streak watchdog after every token regardless
/// of which early-return branch fired in the body. The watchdog
/// can't live inside `handle_token_inner` because that function has
/// many early returns (one per emission path) — putting the check
/// at the end of the body would only fire when the natural fall-
/// through is taken, leaving the doom-loop case (long suppressed
/// stream of orphan `<tool_call>` openers) uncaught.
pub(super) fn handle_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    // Count generated tokens without reopening a terminal parser/stream.
    crate::metrics::DECODED_TOKENS_TOTAL.inc();
    terminal_token::while_open(state, |state| handle_open_token(state, ctx, tok))
}

fn handle_open_token(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    let result = handle_token_inner(state, ctx, tok);

    // Orphan-suppression streak watchdog. The sanitizer flips
    // `suppressing_param_leak=true` when it sees an orphan
    // `<tool_call>` / `<parameter=` opener without a matching close.
    // Suppressing forever (until max_tokens) burns the user's
    // patience and decode budget — observed live as an 8192-token
    // doom loop. If the streak exceeds the bound, end the stream.
    // `--reference-mode` keeps every stream-side cut below off: a reference
    // server ends a response only on EOS / stop / max_tokens.
    if state.suppressing_param_leak && !state.stop_string_triggered && !ctx.state.reference_mode {
        state.suppress_streak_tokens = state.suppress_streak_tokens.saturating_add(1);
        if state.suppress_streak_tokens > MAX_SUPPRESS_STREAK_TOKENS {
            tracing::warn!(
                streak = state.suppress_streak_tokens,
                "orphan tool-call suppression streak exceeded {MAX_SUPPRESS_STREAK_TOKENS} tokens; ending stream",
            );
            state.loop_watchdog_triggered = true;
            state.stop_string_triggered = true;
            // ★ Name the guard so the wire reason comes out "length".
            // Without this the scheduler sees a bare `cancel_flag` with
            // budget left, falls to its early-finalize rule and reports
            // "stop" — telling the client the model finished, when in
            // fact we truncated it mid-doom-loop.
            //
            // This was the THIRD such path. The scheduler guards and the
            // simhash/token-loop watchdogs were fixed first; this one
            // survived both and cost the agentic gate runs 0 and 7 on
            // three consecutive shas, because the harness's
            // `was_cut_off()` only nudges on "length".
            //
            // ★ Note the harness's OTHER recovery route cannot cover it:
            // `emitted_unparsed_call` scans the text for `<tool_call>` /
            // `<function=`, and the sanitizer has already suppressed
            // exactly those bytes — which is why this streak fired at all.
            state.guard_stop = Some("suppress_streak");
            state
                .cancel_flag
                .store(true, std::sync::atomic::Ordering::Release);
        }
    } else if !state.suppressing_param_leak {
        state.suppress_streak_tokens = 0;
    }

    result
}

#[path = "handle_token/detector_outputs.rs"]
mod detector_outputs;
#[path = "handle_token/inner.rs"]
mod inner;
pub(crate) use inner::*;

/// Common processing for a sanitized content chunk: SimHash semantic
/// guard, token-level loop watchdog, salvage on trip, otherwise
/// emit a `Content` delta. Returns `Some(deltas)` when the watchdog
/// fired (caller must short-circuit), else `None` (caller continues).
///
/// Note: when called from the detector-active branch, `sanitized`
/// has already been routed through `sanitize_content_chunk`. When
/// called from the no-detector branch, the caller must pre-sanitize
/// (the no-detector path uses the same sanitizer state).
#[path = "handle_token/tail2.rs"]
mod tail2;
pub(crate) use tail2::*;

#[path = "handle_token/tail.rs"]
mod tail;
pub(crate) use tail::*;
