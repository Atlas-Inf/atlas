// SPDX-License-Identifier: AGPL-3.0-only

//! emit_token + compile_grammar_state + StartPrefillResult enum.

use super::*;

mod cancel;
mod think_commit;
pub(super) use cancel::retire_if_cancelled;
#[cfg(test)]
pub(in crate::scheduler) use think_commit::should_suppress_post_think_eos;
pub(in crate::scheduler) use think_commit::{
    CommitEnv, EndToken, PickEffects, SpanShadow, advance_thinking, end_token, hard_stop,
    think_gate, tool_state,
};

#[cfg(test)]
mod tests;

/// Emit a token for an active sequence (stream + bookkeeping).
///
/// Per OpenAI spec, stop/EOS tokens are NOT streamed to the client —
/// the returned text must not contain the stop sequence. The token is
/// still recorded in output_tokens for accurate token counting.
///
/// When `logprobs` is Some, the logprobs data is accumulated for blocking
/// responses and sent via `StreamEvent::TokenWithLogprobs` for streaming.
pub fn emit_token(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    emit_token_at_position(a, tok, logprobs, sched, a.seq.seq_len);
}

/// [`emit_token`] for row `row` of a verify span's committed prefix of `rows`
/// rows, which ends at the current (post-rewind) `seq_len`: the end-token and
/// context ceilings are checked at the row's own position
/// ([`think_commit::span_row_position`], as serial decode and the verify
/// replay check them), not at the prefix's end for every row.
pub(super) fn emit_span_row(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    row: usize,
    rows: usize,
) {
    let position = think_commit::span_row_position(a.seq.seq_len, rows, row);
    emit_token_at_position(a, tok, logprobs, sched, position);
}

/// Emit one already-verified row after its whole target prefix was committed.
/// The caller supplies the validated logical position for ceiling checks only;
/// canonical model state and all other emission/accounting semantics stay intact.
pub(super) fn emit_token_at_position(
    a: &mut ActiveSeq,
    tok: u32,
    logprobs: Option<crate::api::TokenLogprobs>,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    position: usize,
) {
    // Cooperative cancellation from the streaming pipeline. The
    // stream-side guards (Bug-2 name-run cap, F11 within-dedup, F44
    // perm-fail, loop-watchdog, client stop-sequence match) flip this
    // flag when they decide the response should end. Treat it like an
    // EOS: finalise now (lifecycle derives "stop" — budget not hit —
    // and `handle_done`'s overrides refine it) instead of letting the
    // model keep emitting tokens that just get suppressed.
    if retire_if_cancelled(a) {
        return;
    }
    // A content-loop steer of this pick is counted against the state it was
    // picked in (`loop_steer`), before the commit changes any of it.
    if crate::scheduler::loop_steer::note_commit(a) {
        tracing::warn!(
            tok,
            output_len = a.output_tokens.len(),
            steers = a.loop_steers,
            max = a.loop_steer_max,
            "Content-loop steer: the pick skipped the loop's continuation token"
        );
    }

    let env = CommitEnv::of(sched);
    // The verify pipeline's effects at the position this token was picked
    // from (none for a token no verify picked).
    PickEffects::apply(a, tok);
    if hard_stop(a, tok, &env) {
        return;
    }
    first_token_thinking::apply_native_tool_boundary(a, tok, &env);
    if think_gate(a, tok, &env) {
        return;
    }

    // The required-call opener, the tool-body / parameter-body state machine
    // (SM1, also driven by `process_decode_logits`) and `</tool_call>`'s
    // completion (Fix A): the commit rule's, which the verify replay runs too.
    tool_state(a, tok, &env);

    // F2 mirror (Iter 46, 2026-06-02): reset the inter-tool prose budget when
    // a tool call opens on the MTP/emit path — parity with the non-MTP reset
    // in `decode_logits_step.rs` (the `tool_call_start_token` branch). Without
    // this the budget would accrue across the whole response and the MTP-path
    // budget watchdog (added below) would false-fire after the first
    // `max_inter_tool_prose` content tokens even across legitimate multi-tool
    // turns. Keyed identically: tool-call open, not inside `<think>`.
    if a.tool_call_start_token == Some(tok) && !a.inside_thinking {
        a.prose_tokens_since_last_tool = 0;
    }

    // Advance grammar state with the emitted token — skip while the
    // sequence is inside `<think>`…`</think>` so the matcher only
    // sees the final-output token stream.
    let mut disengage_grammar = false;
    let mut strict_violation = false;
    if !a.inside_thinking
        && let Some(ref mut gs) = a.grammar_state
    {
        let advanced = gs.accept_token(tok);
        if !advanced && gs.is_strict() {
            strict_violation = true;
        } else if !advanced {
            // Grammar/model disagreement (BUG#2 class: e.g. a merged BPE token
            // like `><` or a `</X` content run the qwen3_coder value rule
            // forbids, often surfaced via an under-masked MTP draft). The token
            // is already a legitimate model emission; the matcher is now
            // desynced. Previously we set `a.finished = true` here — a
            // CATASTROPHIC cliff that lost the ENTIRE agentic turn on a single
            // refused token (root cause of the opencode webserver_ok gap).
            // Instead, DISENGAGE the grammar for the remainder of this response
            // and continue decoding UNCONSTRAINED — exactly what vLLM (the 10/10
            // reference) does by parsing tools post-hoc. Atlas's server-side
            // tool parser still extracts tool calls from the emitted text, so
            // the structural guarantee is gracefully traded for turn survival.
            tracing::warn!(
                tok,
                output_len = a.output_tokens.len(),
                "gs.accept_token returned false — grammar/model disagreement; disengaging grammar for the remainder of this response (free decode + post-hoc tool parse) instead of aborting the turn."
            );
            disengage_grammar = true;
        }
    }
    if strict_violation {
        fail_strict_grammar(a, tok);
        return;
    }
    if disengage_grammar {
        // Drop the matcher: subsequent decode steps see `grammar_state == None`
        // and decode unconstrained. Set after the `ref mut gs` borrow ends.
        a.grammar_state = None;
    }

    // Accumulate logprobs data for blocking responses.
    if let Some(lp) = logprobs {
        a.logprobs_data.push(lp);
    }

    let prior = a.output_tokens.len();
    a.output_tokens.push(tok);

    // Spec-resume guard bookkeeping: count tokens emitted after `</think>`.
    // The `</think>` token itself is not counted (think_ended is still false
    // when it arrives; the transition below sets it). For requests that never
    // think, think_ended starts true, so the guard delays spec by the same N
    // from the response start.
    if a.think_ended && !a.inside_thinking {
        a.post_think_emitted += 1;
    }

    let native_glm_eos = crate::glm_tool_boundary::native_eos_while_thinking(
        sched.limits.glm_tool_boundary,
        a.inside_thinking,
        tok,
        &a.eos_tokens,
    );
    if a.inside_thinking {
        advance_thinking(a, tok, prior, native_glm_eos, &env);
    } else {
        a.consume_generation_budget();
        // Clear think_just_ended one-shot now that we've consumed the
        // token after </think>.
        a.think_just_ended = false;
        // Content-phase loop watchdog. Mirrored from
        // `handle_content_token` (decode_logits_content.rs) because
        // that handler is only invoked on the non-MTP decode path
        // (`process_decode_logits`). MTP speculative decode
        // (`verify_k2_step`) reaches every token through this
        // `emit_token` instead — without this mirror, the
        // content-loop watchdog never fires while MTP is enabled, and
        // the model can burn the full `max_tokens` budget on a
        // period-N attractor. Observed live 2026-05-24 on
        // opencode-hotfix2b.jsonl seq=13: 8193 content tokens of
        // `[29, 198, 510, 15704, …]` period-4 loop (the
        // `parameter>\n` attractor) with no watchdog fire,
        // finish=length.
        //
        // 2026-05-24 sweep #3: Re-introduced the `!a.inside_tool_body`
        // gate (mirrors the handle_content_token policy). The previous
        // inside-body false-positives turned out to be triggered by a
        // separate MTP-pipeline gap (see bench/hotfix3-debug/
        // SYNTHESIS.md). With the pipeline correctly applied to MTP
        // verify, JSON structural repetition is bounded by the
        // grammar's terminal state. The `parameter>\n` real-loop case
        // is still caught one tick AFTER the model exits the tool
        // body — its emission outside the body forms a tight period-N
        // tail.
        //
        // Skip rollback here — `emit_token` doesn't take `&dyn Model`
        // (the SSM rewind requires it) and plumbing it through every
        // call site would balloon the diff. Instead set `a.finished`
        // and let the lifecycle close the response. The non-MTP path
        // retains rollback via `handle_content_token`.
        use crate::scheduler::helpers::{
            CONTENT_LOOP_CHECK_STRIDE, CONTENT_LOOP_MIN_TOKENS, CONTENT_LOOP_PERIOD_MAX,
            CONTENT_LOOP_PERIOD_MIN, detect_content_token_loop_normalized_with,
            detect_content_token_loop_with,
        };
        a.content_tokens = a.content_tokens.saturating_add(1);
        // F1 (2026-06-02): unconditional per-generation post-think content
        // cap. Fires regardless of `inside_tool_body` so it bounds the
        // runaway no matter which heuristic state machine desynced (RC1/
        // RC2/RC3). Gated on `grammar_state.is_some()` ⇒ only tool-active
        // requests are ever capped (plain chat attaches no grammar and is
        // never truncated). Default 100_000 (`MAX_POST_THINK_CONTENT_TOKENS`)
        // = no-op; Qwen3.6-35B-A3B-FP8 sets 1536 in MODEL.toml.
        if !sched.levers.disable_watchdogs
            && a.grammar_state.is_some()
            && !a.strict_grammar()
            && a.content_tokens > sched.watchdog.max_post_think_content_tokens
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                max = sched.watchdog.max_post_think_content_tokens,
                "post-think content cap exceeded in MTP/emit path; ending response (tool-active request would otherwise burn to max_tokens)"
            );
            a.guard_stop = Some(GUARD_STOP_POST_THINK_CAP);
            a.finished = true;
        }
        // Same precedence as the non-MTP twin (decode_logits_content.rs):
        // request `repetition_detection` → operator min-repeats override →
        // built-in constants.
        let loop_params = sched.watchdog.content_loop_params(a.repetition_detection);
        if !sched.levers.disable_watchdogs
            && sched.levers.loop_watchdog()
            && !a.inside_tool_body
            && !a.strict_grammar()
            && watchdog_floor_reached(a.output_tokens.len(), a.min_tokens)
            && a.content_tokens >= CONTENT_LOOP_MIN_TOKENS
            && a.content_tokens.is_multiple_of(CONTENT_LOOP_CHECK_STRIDE)
            // The next pick steers this tail off its loop (`loop_steer`).
            && !crate::scheduler::loop_steer::will_steer(a)
            && (detect_content_token_loop_with(&a.output_tokens, loop_params)
                || sched.masks.numeric.as_deref().is_some_and(|m| {
                    detect_content_token_loop_normalized_with(
                        &a.output_tokens,
                        m,
                        sched.masks.punctuation.as_deref(),
                        loop_params,
                    )
                }))
        {
            tracing::warn!(
                content_tokens = a.content_tokens,
                output_len = a.output_tokens.len(),
                "Content-loop watchdog fired in MTP/emit path (period-{}…{} repeat); ending response. \
                 Tune via --content-loop-min-repeats / ATLAS_CONTENT_LOOP_MIN_REPEATS, per-request \
                 repetition_detection, or disarm via --content-loop-watchdog false / \
                 ATLAS_CONTENT_LOOP_WATCHDOG=0",
                CONTENT_LOOP_PERIOD_MIN,
                CONTENT_LOOP_PERIOD_MAX,
            );
            // #328 class: name the cut or it wires "stop" (see types.rs).
            a.guard_stop = Some(GUARD_STOP_CONTENT_LOOP);
            a.finished = true;
        }

        // F2 mirror (Iter 46, 2026-06-02): inter-tool PROSE-BUDGET watchdog on
        // the MTP/emit path. The 2026-05-24 mirror above copied only the
        // content-LOOP guard; the prose-budget guard (decode_logits_content.rs)
        // stayed non-MTP-only — so with `--num-drafts ≥ 1` (MTP/verify path),
        // a turn that wanders WITHOUT producing a parseable tool call had NO
        // bound and burned the whole `max_tokens` budget (~270s at 30 tok/s),
        // starving the agent of turns. This was the dominant opencode
        // `webserver_ok` 360s-timeout cause: at deep context the model flips
        // its tool opener to Anthropic-XML `<invoke name=…>`, which never
        // matches the qwen3_coder trigger, so the trigger-gated grammar stays
        // dormant and the wander is not a tight period-≤64 loop the content
        // watchdog catches. Same gates as the non-MTP block: free-text only
        // (`!inside_tool_body`) and grammar attached (`grammar_state.is_some()`
        // ⇒ a tool request, never plain chat — so a long chat answer is not
        // truncated). No rollback here: `emit_token` has no `&dyn Model` (the
        // SSM rewind needs it), so we hard-stop exactly like the content-loop
        // mirror; the sanitizer + post-hoc tool parser salvage what was emitted.
        // F4 (2026-06-02): gate on the sticky `tool_request` flag (set at
        // prefill, survives a graceful grammar disengage) instead of
        // `grammar_state.is_some()` — otherwise a disengaged tool turn on
        // the MTP path wanders to `max_tokens` with the budget inert.
        if !sched.levers.disable_watchdogs && !a.inside_tool_body && a.tool_request {
            a.prose_tokens_since_last_tool = a.prose_tokens_since_last_tool.saturating_add(1);
            let max_prose = sched.watchdog.max_inter_tool_prose;
            if a.prose_tokens_since_last_tool > max_prose {
                tracing::warn!(
                    prose_tokens = a.prose_tokens_since_last_tool,
                    max = max_prose,
                    output_len = a.output_tokens.len(),
                    "Inter-tool prose budget exhausted in MTP/emit path; ending response \
                     (no tool call after budget — would otherwise burn to max_tokens); \
                     raise via --max-inter-tool-prose / ATLAS_MAX_INTER_TOOL_PROSE / \
                     MODEL.toml [behavior].max_inter_tool_prose (0 disables)"
                );
                a.guard_stop = Some(GUARD_STOP_INTER_TOOL_PROSE);
                a.finished = true;
            }
        }
    }

    // The end-token decision is the commit rule's (`think_commit::end_token`),
    // made the same way by plain decode: grammar stop-legality, an unsatisfied
    // tool call, the `min_tokens` floor, `<think>` (an honored end token closes
    // the block instead) and the post-`</think>` tool guard hold it back; a
    // hard ceiling (budget spent, context full) always stops. A stop is
    // recorded for the token count but never streamed (OpenAI: the returned
    // text excludes the stop sequence).
    match end_token(a, tok, prior, position, native_glm_eos, &env) {
        EndToken::Stop => {
            a.finished = true;
            return;
        }
        // Discarded exactly as serial decode discards it: never in the output,
        // so it never counts toward the min_tokens floor (`min_tokens_eos_tests`).
        EndToken::Suppressed => {
            a.output_tokens.pop();
            return;
        }
        EndToken::No => {}
    }
    // OPENCODE FIX: see process_decode_logits — same gate. Suppress streaming
    // of spontaneous-thinking content so it doesn't pollute opencode's history.
    let suppress_stream = a.inside_thinking && !a.enable_thinking;
    if !suppress_stream {
        let event = if let Some(lp) = a.logprobs_data.last().cloned() {
            StreamEvent::TokenWithLogprobs(tok, lp)
        } else {
            StreamEvent::Token(tok)
        };
        if !send_stream_event(a, event) {
            a.finished = true;
            return;
        }
    }
    // §C-3 (DS4F hard-limit lane, 2026-07-21): the `remaining == 0` completion
    // stop is now joined by the per-step served-`max_seq_len` ceiling stop
    // (twin of the non-MTP guard in `decode_logits_step`), so the MTP/emit path
    // also cannot run KV past the context ceiling. No-op when `max_seq_len` is
    // unset (0) or not yet reached.
    if a.remaining == 0 || seqlen_force_stop(position, sched.limits.max_seq_len) {
        // #144: before the hard length-stop, if a grammar is active and the
        // stop token is not legal at the current position (e.g. mid JSON
        // string), emit the shortest grammar-legal close so the truncated
        // `finish_reason="length"` output is still parseable.
        emit_grammar_close(a);
        tracing::info!(
            "emit_token: remaining={}, seq_len={}, max_seq_len={}, output_tokens={}, thinking_tokens={}",
            a.remaining,
            a.seq.seq_len,
            sched.limits.max_seq_len,
            a.output_tokens.len(),
            a.thinking_tokens
        );
        a.finished = true;
    }
}

/// Cap on the grammar-close byte length explored at budget end (#144). A
/// structural close (`"`, `}`, `]`, …) is short; if no close is reachable
/// within this many bytes the response finishes as before (plain length-stop).
const MAX_GRAMMAR_CLOSE_BYTES: usize = 32;

/// Send one stream event to the response sink, handling backpressure.
/// Returns `false` if the receiver has dropped (caller should finish the
/// sequence). A non-streaming sink is a no-op that returns `true`.
///
/// Extracted from `emit_token`'s inline send so the budget-aware close
/// streams its tokens through the identical path — SSOT for the
/// try_send / blocking_send backpressure discrimination (transient channel-full
/// vs. real consumer-drop; collapsing them once truncated seqs mid-stream).
fn send_stream_event(a: &ActiveSeq, event: StreamEvent) -> bool {
    let ResponseSink::Streaming(ref tx) = a.sink else {
        return true;
    };
    super::mod_helpers::bounded_stream_send(tx, event, "token stream")
}

/// #144 budget-aware graceful close. At budget exhaustion, if a grammar is
/// active, not terminated, and the stop token is NOT legal at the current
/// position, emit the shortest grammar-legal close so a length-truncated
/// structured-output response is still parseable instead of ending with an
/// open string / unbalanced JSON. The close tokens are pushed to
/// `output_tokens` and streamed through [`send_stream_event`] (so blocking and
/// streaming responses agree); they intentionally exceed `max_tokens` by the
/// bounded close length, mirroring a graceful EOS. No-op when disabled
/// (`ATLAS_GRAMMAR_BUDGET_CLOSE=0`), inside `<think>`, or when no bounded close
/// is found — all of which fall back to the prior plain length-stop.
pub(crate) fn emit_grammar_close(a: &mut ActiveSeq) {
    if a.inside_thinking || !grammar_budget_close_enabled() {
        return;
    }
    let close = {
        let Some(gs) = a.grammar_state.as_mut() else {
            return;
        };
        if gs.is_terminated() || gs.stop_legal(&a.eos_tokens) {
            return;
        }
        match gs.completion_token_ids(MAX_GRAMMAR_CLOSE_BYTES) {
            Some(tokens) if !tokens.is_empty() => tokens,
            _ => return,
        }
    };
    tracing::info!(
        close_len = close.len(),
        output_len = a.output_tokens.len(),
        "grammar budget-close: emitting graceful close so length-stop yields parseable output"
    );
    for tok in close {
        let tok = tok as u32;
        a.output_tokens.push(tok);
        if !send_stream_event(a, StreamEvent::Token(tok)) {
            break;
        }
    }
}

// F72 (byte-level partial-trigger anchor) was removed in F73 / fix42.
// The sampler-side anchor hung the server in production; the broken
// envelope is now recovered at the streaming-sanitizer + parser
// layer. xgrammar's non-anchored TagDispatch limitation is pinned
// for documentation by
// `grammar.rs::tests::test_minimax_xml_grammar_masks_trigger_breaking_multibyte_token`.

/// A `response_format` grammar refused an emitted token. Its output must
/// conform, so end the response with an error (wire finish reason "error",
/// blocking HTTP 500) rather than disengage and return unconstrained text as
/// if it were valid. Tool grammars disengage instead (see `emit_token`). The
/// refused token is neither emitted nor recorded.
pub(super) fn fail_strict_grammar(a: &mut ActiveSeq, tok: u32) {
    let msg = strict_refusal(tok, a.output_tokens.len());
    a.abort_on_engine_error(msg);
}

/// Logged and returned for every strict-grammar refusal (first token included).
pub(super) fn strict_refusal(tok: u32, at: usize) -> String {
    tracing::error!(
        tok,
        output_len = at,
        "response_format grammar refused an emitted token; ending the response with an error"
    );
    format!(
        "structured output: token {tok} at output position {at} violates the response_format grammar"
    )
}

/// Compile a grammar state from a grammar specification + engine.
///
/// Returns `Ok(Some(GrammarState))` if compilation succeeds, `Ok(None)` when
/// no grammar was requested or a tool grammar failed (logging a warning so the
/// request falls back to legacy tool_call suppression), except a tool grammar
/// too large to compile, which is refused like a strict one. Called once per
/// request during prefill; a panic inside compilation is caught and reported.
///
/// A `response_format` grammar is marked strict (`opens_in_thinking`: the
/// prompt leaves the model inside `<think>`), and failing to arm one is
/// reported to the client on `sink` as an invalid request (HTTP 400 when
/// blocking) and returned as `Err`: its output must never come back
/// unconstrained as if it were valid.
pub fn compile_grammar_state(
    engine: &mut Option<GrammarEngine>,
    grammar_spec: &Option<GrammarSpec>,
    eos_tokens: &[u32],
    opens_in_thinking: bool,
    sink: &mut ResponseSink,
) -> Result<Option<GrammarState>> {
    let Some(spec) = grammar_spec.as_ref() else {
        return Ok(None);
    };
    let strict = spec.is_response_format();
    let compiled = match engine.as_mut() {
        // Backstop: nothing a client sends may unwind the scheduler thread.
        Some(engine) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            compile_spec(engine, spec, eos_tokens)
        }))
        .unwrap_or_else(|p| Err(format!("grammar compilation panicked: {}", panic_text(&p)))),
        None if strict => Err("this model has no grammar engine".to_string()),
        None => Ok(None),
    };
    match compiled {
        Ok(state) if strict => Ok(state.map(|s| s.strict_output(opens_in_thinking))),
        Ok(state) => Ok(state),
        // A request the server cannot serve as asked: HTTP 400 for a blocking
        // client, not a retryable 500. A tool grammar too large to compile is
        // one too: serving it unconstrained would hide the cause.
        Err(e) if strict || e.contains(GRAMMAR_TOO_LARGE) => {
            let what = if strict {
                "response_format"
            } else {
                "the tool grammar"
            };
            let msg = format!("{what} cannot be enforced: {e}");
            tracing::warn!("{msg}");
            send_invalid_request_to_sink(sink, &msg);
            anyhow::bail!(msg)
        }
        Err(e) => {
            tracing::warn!("{e}");
            Ok(None)
        }
    }
}

/// How `xgrammar::CompileError::TooLarge` reads once stringified.
const GRAMMAR_TOO_LARGE: &str = "grammar too large";

fn panic_text(p: &(dyn std::any::Any + Send)) -> &str {
    p.downcast_ref::<&str>()
        .copied()
        .or_else(|| p.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("non-string payload")
}

/// `Ok(None)` only when a tool parser opts out of constrained decoding.
fn compile_spec(
    engine: &mut GrammarEngine,
    spec: &GrammarSpec,
    eos_tokens: &[u32],
) -> Result<Option<GrammarState>, String> {
    // F69 (2026-04-29): symmetric dispatch via the trait. The parser
    // is the single source of truth for both runtime parsing and
    // grammar compilation; no string match keyed on `parser_name`.
    // Mistral's default trait impl returns `None`, which we treat as
    // "no constraint, fall through to unconstrained decoding."
    let compiled = match spec {
        GrammarSpec::ToolCall {
            tools,
            parser,
            use_triggers,
        } => match parser.compile_tool_grammar(engine, tools, *use_triggers) {
            Some(result) => result,
            None => {
                tracing::debug!(
                    "Grammar: parser '{}' opted out of constrained decoding for this request",
                    parser.name(),
                );
                return Ok(None);
            }
        },
        GrammarSpec::JsonObject => engine.compile_json_grammar(),
        GrammarSpec::JsonSchema { schema } => engine.compile_json_schema(schema),
    };

    let label = match spec {
        GrammarSpec::ToolCall { parser, tools, .. } => {
            format!("parser={}, tools={}", parser.name(), tools.len())
        }
        GrammarSpec::JsonObject => "response_format=json_object".to_string(),
        GrammarSpec::JsonSchema { .. } => "response_format=json_schema".to_string(),
    };

    let grammar = compiled.map_err(|e| format!("Grammar compilation failed: {e}"))?;
    let state = GrammarState::new(&grammar, engine.vocab_size())
        .map_err(|e| format!("Grammar state creation failed: {e}"))?;
    tracing::info!("Grammar constrained decoding active: {label}");
    // Exempt the model's stop/EOS tokens from grammar refusal so a legitimate
    // end-of-turn token cannot desync the NPDA and truncate the response
    // (see GrammarState::accept_token).
    Ok(Some(state.with_stop_tokens(eos_tokens)))
}

/// Result of starting a chunked prefill.
pub enum StartPrefillResult {
    /// Prompt fit in one chunk → ready for decode.
    Active(ActiveSeq),
    /// Prompt needs more chunks → add to prefilling[].
    InProgress(PrefillInProgress),
    /// Completed during first chunk (EOS on first token).
    Finished,
}

// Tool-body / parameter-body state machine, hoisted out of
// `emit_token` (SM1, 2026-05-26).
//
// Both speculative-decoding paths (`verify_k2_step`, `verify_k4_step`,
// `verify_dflash_step`, `spec_step`) and the regular non-spec decode
// path (`decode_logits_step::process_decode_logits`) call this on
// every emitted token so the state machine stays in sync with
// `a.output_tokens`. The previous inline version was unreachable
// from the non-spec path, leaving the close-tag mask, AM1 attractor
// suppression, B1 margin detector, and A1 penalty toggle all silently
// dead.
//
// **Slice semantics**: this function does NOT assume `tok` has been
// pushed onto `a.output_tokens` or that it has not. It auto-detects
// from `a.output_tokens.last()` and slices accordingly:
//  - `emit_token` calls this BEFORE pushing → `last()` is the prior
//    token, lookback uses the full slice.
//  - `decode_logits_step::process_decode_logits` calls this AFTER
//    pushing → `last()` is `tok`, lookback excludes the trailing
//    entry so the search for `<parameter=KEY>` ending at the current
//    `>` is correct in both cases.
//
// State mutations:
//  - `a.inside_tool_body`         set on `<tool_call>`, cleared on `</tool_call>`
//  - `a.tool_body_streak_tokens`  ++ per body token, reset on enter/exit
//  - `a.inside_parameter_body`    set on `<parameter=KEY>` close `>`, cleared on `</`
//  - `a.param_body_chars_emitted` ++ per non-close body token
//  - `a.finished`                 forced when stuck >MAX_TOOL_BODY_TOKENS
//
// Token IDs are Qwen3.6 byte-level BPE (verified via /tokenize 2026-05-25):
//   27 = `<`, 28 = `=`, 29 = `>`, 510 = `</`, 15704 = `parameter`.

/// Cap on tool-call ENVELOPE tokens (everything inside `<tool_call>…</tool_call>`
/// that is NOT a parameter-value body). Catches a model that opens `<tool_call>`
/// and never reaches `</tool_call>` — it would otherwise burn to max_tokens.
const MAX_TOOL_BODY_TOKENS: u32 = 1024;

/// Pure decision core for the envelope-stuck guard (CC6, 2026-06-07).
/// Tokens of a parameter VALUE (`inside_parameter_body`) are exempt — a
/// legitimately large single-file Write must stream without tripping the cap.
/// Only envelope tokens (`<parameter=KEY>` openers, inter-parameter junk, any
/// non-value token) advance the streak. Pure over scalars so it is unit-tested
/// directly, mirroring `rollback_tests.rs`'s pure-core approach (no `ActiveSeq`
/// fixture needed). Returns `(new_streak, exceeded_cap)`.
fn advance_envelope_streak(inside_parameter_body: bool, streak: u32) -> (u32, bool) {
    if inside_parameter_body {
        (streak, false)
    } else {
        let s = streak.saturating_add(1);
        (s, s > MAX_TOOL_BODY_TOKENS)
    }
}

/// `quiet`: the verify replay's, which must not log a cut the stream never sees.
pub fn update_tool_param_state(a: &mut ActiveSeq, tok: u32, quiet: bool) {
    if a.inside_thinking {
        return;
    }
    if a.tool_call_start_token == Some(tok) {
        a.inside_tool_body = true;
        a.tool_body_streak_tokens = 0;
        return;
    }
    if a.tool_call_end_token == Some(tok) {
        a.inside_tool_body = false;
        a.tool_body_streak_tokens = 0;
        a.inside_parameter_body = false;
        a.param_body_chars_emitted = 0;
        return;
    }
    if !a.inside_tool_body {
        return;
    }
    // CC6 (2026-06-07): count ONLY envelope tokens — tokens of a
    // `<parameter=…>` VALUE (the file content of a `write` call) are exempt,
    // so a legitimately large single-file Write streams without tripping the
    // cap. The never-closing-envelope runaway still accumulates here (openers,
    // inter-parameter junk, any token emitted while `inside_parameter_body ==
    // false` still counts). Resets on tool open/close (above) and `</parameter>`
    // exit (below) are unchanged.
    let (streak, exceeded) =
        advance_envelope_streak(a.inside_parameter_body, a.tool_body_streak_tokens);
    a.tool_body_streak_tokens = streak;
    if exceeded {
        if !quiet {
            tracing::warn!(
                streak = a.tool_body_streak_tokens,
                "Stuck in tool-call ENVELOPE for {MAX_TOOL_BODY_TOKENS}+ tokens with no </tool_call> (excludes parameter-value content); ending response (model never closed the envelope — would otherwise burn to max_tokens). Sanitizer will salvage what it can."
            );
        }
        a.guard_stop = Some("tool_envelope_stuck");
        a.finished = true;
    }

    const TOK_LT: u32 = 27;
    const TOK_PARAMETER: u32 = 15704;
    const TOK_EQ: u32 = 28;
    const TOK_GT: u32 = 29;
    const TOK_LT_SLASH: u32 = 510;

    if a.inside_parameter_body {
        // P0-1 (2026-07-09): PROVISIONAL close detection. The old code
        // exited the body on ANY `</` token (510) — but since the
        // `<`-initial-value fix, parameter VALUES legitimately contain
        // HTML/Svelte close tags (`</script>`, `</div>`, …), every one of
        // which starts with token 510. Each false exit reclassified the
        // rest of the file content as ENVELOPE tokens, walked the streak
        // to MAX_TOOL_BODY_TOKENS, and force-killed legitimate writes
        // mid-file (8 kills in the 2026-07-09 45k session). Now the exit
        // COMMITS only on the full confirmed `</` `parameter` `>` token
        // sequence; any other continuation re-enters the value body and
        // back-counts the provisionally-held tokens as body chars. A
        // merged/nonstandard tokenization of the close falls back to
        // "stay inside" — safe: value tokens are streak-exempt, and the
        // next exact close still exits.
        match a.param_close_pending {
            0 => {
                if tok == TOK_LT_SLASH {
                    a.param_close_pending = 1;
                } else {
                    // Any non-close body token advances the counter. The
                    // position-0 mask in `decode_logits_seq.rs` (close-tag +
                    // AM1 attractor) fires only while this counter is 0, so it
                    // deactivates after the first emitted body token.
                    a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(1);
                }
            }
            1 => {
                if tok == TOK_PARAMETER {
                    a.param_close_pending = 2;
                } else {
                    // `</` was value content (e.g. `</div>`), not a close.
                    a.param_close_pending = 0;
                    a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(2);
                }
            }
            _ => {
                a.param_close_pending = 0;
                if tok == TOK_GT {
                    // Confirmed `</parameter>` — exit body. Also reset the
                    // envelope streak: a confirmed close IS forward progress
                    // (the doc comment above always claimed this reset; the
                    // code never performed it).
                    a.inside_parameter_body = false;
                    a.param_body_chars_emitted = 0;
                    a.tool_body_streak_tokens = 0;
                } else {
                    // `</parameter` NOT followed by `>` (e.g. the garbled
                    // `</parameter<parameter=` reopen, or `</parameters`) —
                    // grammar-legal value content; stay inside the body.
                    a.param_body_chars_emitted = a.param_body_chars_emitted.saturating_add(3);
                }
            }
        }
        return;
    }

    // Not yet inside_parameter_body: scan for `<parameter=KEY>` opener
    // ending at this `>` (29). Lookback 8 tokens for `[27, 15704, 28]`
    // signature without an intervening close.
    if tok != TOK_GT {
        return;
    }
    // Auto-detect whether `tok` is already in output_tokens (caller
    // pushed) or not (caller has not yet pushed). The signature search
    // must NOT include `tok` itself — the lookback is "what came
    // BEFORE this `>`".
    let n = a.output_tokens.len();
    let n_for_lookback = if n > 0 && a.output_tokens[n - 1] == tok {
        n - 1
    } else {
        n
    };
    if n_for_lookback < 3 {
        return;
    }
    let start = n_for_lookback.saturating_sub(8);
    let window = &a.output_tokens[start..n_for_lookback];
    let mut sig_idx: Option<usize> = None;
    for i in 0..window.len().saturating_sub(2) {
        if window[i] == TOK_LT && window[i + 1] == TOK_PARAMETER && window[i + 2] == TOK_EQ {
            sig_idx = Some(i + 3);
        }
    }
    let Some(after_eq) = sig_idx else { return };
    // Check no intervening `</` or `>` in the KEY span between
    // `<parameter=` and the current `>`.
    let body_segment = &window[after_eq..];
    let intervening_close = body_segment
        .iter()
        .any(|&t| t == TOK_LT_SLASH || t == TOK_GT);
    if !intervening_close {
        a.inside_parameter_body = true;
        a.param_body_chars_emitted = 0;
    }
}

// SM1 unit tests deferred: ActiveSeq has 60+ fields and no public
// constructor; building a test instance requires more boilerplate
// than the state machine itself. Live-verification post-deploy is via
// the A1 rep-penalty toggle / B1 margin-detector behaviour.

#[cfg(test)]
mod cc6_envelope_streak_tests {
    //! CC6 (2026-06-07): the envelope-stuck guard must NOT truncate a large
    //! legitimate file write (parameter-value content), while STILL catching a
    //! `<tool_call>` that never closes. Tested on the pure `advance_envelope_streak`
    //! core (mirrors `rollback_tests.rs` — no `ActiveSeq` fixture required).
    use super::{MAX_TOOL_BODY_TOKENS, advance_envelope_streak};

    #[test]
    fn parameter_value_content_is_exempt_at_any_size() {
        // Simulate a ~6000-token file content streaming inside <parameter=content>.
        let mut streak = 0u32;
        for _ in 0..6000 {
            let (s, exceeded) = advance_envelope_streak(true, streak);
            streak = s;
            assert!(
                !exceeded,
                "parameter-value content must never trip the envelope cap"
            );
        }
        assert_eq!(
            streak, 0,
            "value content must not advance the envelope streak"
        );
    }

    #[test]
    fn never_closing_envelope_still_trips_cap() {
        // True runaway: envelope tokens (NOT inside a parameter value) past the cap.
        let mut streak = 0u32;
        let mut tripped = false;
        for _ in 0..(MAX_TOOL_BODY_TOKENS + 5) {
            let (s, exceeded) = advance_envelope_streak(false, streak);
            streak = s;
            if exceeded {
                tripped = true;
                break;
            }
        }
        assert!(
            tripped,
            "a never-closing envelope emitting >cap non-value tokens must trip"
        );
        assert_eq!(
            streak,
            MAX_TOOL_BODY_TOKENS + 1,
            "fires exactly one token past the cap"
        );
    }

    #[test]
    fn exact_cap_boundary() {
        assert_eq!(
            advance_envelope_streak(false, MAX_TOOL_BODY_TOKENS - 1),
            (MAX_TOOL_BODY_TOKENS, false)
        );
        assert_eq!(
            advance_envelope_streak(false, MAX_TOOL_BODY_TOKENS),
            (MAX_TOOL_BODY_TOKENS + 1, true)
        );
    }

    #[test]
    fn saturates_without_panic() {
        // Envelope streak at u32::MAX must not panic (saturating_add) and stays tripped.
        let (s, exceeded) = advance_envelope_streak(false, u32::MAX);
        assert_eq!(s, u32::MAX);
        assert!(exceeded);
    }
}
