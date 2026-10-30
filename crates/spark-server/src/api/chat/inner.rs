// SPDX-License-Identifier: AGPL-3.0-only

//! `chat_completions_inner` — the chat-completions handler (extracted from
//! `chat/mod.rs` under the 500-LoC cap; no logic edits).

use super::*;

/// Internal entry for the IR-request path. Called by
/// [`chat_completions`] after body capture and wire→IR lowering, and
/// by the Responses / Anthropic adapters (which lower their own wire
/// formats and skip HTTP body bytes). `dump_seq` is `Some` only on
/// the public handler path.
pub(crate) async fn chat_completions_inner(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    mut req: crate::ir::ChatRequest,
    dump_seq: Option<u64>,
) -> ChatOutcome {
    crate::metrics::REQUESTS_TOTAL.inc();
    // RAII: decrements on EVERY exit path, including this future being dropped
    // when the client disconnects (see ActiveRequestGuard / atlas#368). Moved
    // into the SSE stream for streaming requests so it outlives this function.
    let active_guard = crate::metrics::ActiveRequestGuard::new();

    // ── Input validation + cross-turn F-feature guards ──
    if let Err(resp) = super::super::chat_phases::validate_input(&req) {
        return ChatOutcome::Http(resp);
    }

    // Tool-parser behavioral system prompt REMOVED again (2026-05-25 PM).
    //
    // Re-injecting the qwen3_coder `system_prompt` (with its
    // `<parameter=content>[package]\nname = "x"</parameter>` example
    // and `For 'Write'/'Edit' tools specifically: ...` guidance) was a
    // mid-day attempt to give the model better multi-line content
    // hints. Live opencode v39 session showed the opposite effect:
    // the model emitted LITERAL `<tool_call><bash><command>` XML as
    // CONTENT (with HTML-entity escaping like `&amp;`) because TWO
    // tool-format guidances were competing — the chat template's
    // `tools` argument AND my injected prompt — combined with PR 73's
    // `qwen3_xml` parser. The model got confused which format to use
    // and emitted free-form XML that the parser couldn't recognise.
    //
    // Per user's recall: the "MUCH better" state had `thinking_in_tools=true`
    // and the chat template alone (no injection). Reverting matches
    // that state. PR 73's qwen3_xml + native FP8 SSM + streaming
    // byte-exact + gate-BF16 + thinking_in_tools=true is the live
    // combination.

    // M2 per-request LoRA routing: resolve the optional `adapter` name to a
    // pool slot ONCE here (both dispatch paths inherit it). Unset defers to the
    // installed active adapter (`-1`, byte-identical to today); an unknown name
    // is a hard 400, a STAGEABLE name triggers the #27 on-miss RDMA promotion.
    let adapter_slot = match super::super::lora_control::resolve_request_adapter_slot(
        &state,
        req.adapter.as_deref(),
        &req.model,
    )
    .await
    {
        Ok(slot) => slot,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    // Resolve optional per-request source/target language token NAMES to token
    // ids via the server tokenizer. Absent = deployment default (0); an unknown
    // token is a hard 400 (mirrors the adapter-name resolution convention).
    let resolve_lang = |name: &Option<String>| -> Result<u32, Response> {
        match name {
            None => Ok(0),
            Some(s) => state.tokenizer.inner().token_to_id(s).ok_or_else(|| {
                openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("unknown language token '{s}'"),
                )
            }),
        }
    };
    let src_lang_id = match resolve_lang(&req.src_lang) {
        Ok(v) => v,
        Err(resp) => return ChatOutcome::Http(resp),
    };
    let tgt_lang_id = match resolve_lang(&req.tgt_lang) {
        Ok(v) => v,
        Err(resp) => return ChatOutcome::Http(resp),
    };

    // NLLB beam search params (mirrors src/tgt lang resolution). Streaming +
    // beam is unsupported (the beam path emits a single completed hypothesis,
    // not an incremental token stream) — reject up front like n>1.
    let num_beams = req.num_beams.unwrap_or(1);
    let length_penalty = req.length_penalty.unwrap_or(1.0);
    let early_stopping = req.early_stopping.unwrap_or(false);
    if num_beams > 1 && req.stream {
        return ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            "num_beams > 1 is not supported in streaming mode".to_string(),
        ));
    }

    // ── Phases 1-5 (prompt-affecting): shared with count_tokens ──
    //
    // Runs on the BLOCKING pool, not the async worker. Phase timing measured
    // this chain at 1.4-3.2 ms on a short prompt and 13.0 ms at 12931 prompt
    // tokens — nearly all of it the Jinja render + tokenize — and it used to run
    // inline in this async fn, so a worker thread was held for the duration and
    // could poll no other request.
    //
    // `spawn_blocking` rather than `block_in_place`: `block_in_place` evicts the
    // other tasks from the current worker and may spawn a replacement thread, so
    // it suits OCCASIONAL blocking. Here every request would do it, which turns
    // into continuous worker cannibalisation as concurrency rises. The blocking
    // pool exists for exactly this and keeps all async workers pollable.
    //
    // `'static` is satisfied by ownership, not by a scoped-spawn crate: `state`
    // is already an `Arc` (clone = refcount bump) and `req` is MOVED in and
    // handed back, which also preserves the tool-system-prompt mutation
    // `prepare_chat_prompt` applies to it via `&mut`.
    //
    // NOTE: this is a CONCURRENCY fix. At --max-batch-size 1 single-stream there
    // is no second request to unblock, so it buys nothing there and costs one
    // task handoff; it is worth it for concurrent serving.
    let _t_seg = std::time::Instant::now();
    // Keyed on the client's messages, before prepare's tool-prompt injection.
    let session_key = crate::session_key::resolve_from_request(&req);
    let state_for_prepare = state.clone();
    let (prepared, moved_req) = match tokio::task::spawn_blocking(move || {
        let out = prepare::prepare_chat_prompt(&state_for_prepare, &mut req);
        (out, req)
    })
    .await
    {
        Ok(pair) => pair,
        Err(join_err) => {
            // The blocking task panicked. Surface a 500 rather than letting the
            // JoinError unwind through the handler.
            tracing::error!("prepare_chat_prompt panicked: {join_err}");
            return ChatOutcome::Http(openai_error_response(
                StatusCode::INTERNAL_SERVER_ERROR,
                "Internal error preparing the chat prompt".to_string(),
            ));
        }
    };
    req = moved_req;
    let prepare::PreparedChat {
        tools_active,
        cwd_hint,
        image_pixels,
        prompt_tokens,
        enable_thinking,
        thinking_budget,
        force_nonempty_content,
        repeat_run,
    } = match prepared {
        Ok(p) => p,
        Err(resp) => {
            // A prepare failure (template render, image decode, context
            // bounds) must not be a silent 4xx: 2022's ST-995 leg lost ~775
            // samples to an invisible early exit because nothing logged the
            // rejection. WARN with the request's shape is the minimum a
            // triager needs.
            tracing::warn!(
                "Request rejected before prefill: status={} messages={} tools={} stream={}",
                resp.status(),
                req.messages.len(),
                req.tools.len(),
                req.stream,
            );
            return ChatOutcome::Http(resp);
        }
    };

    let us_prepare = _t_seg.elapsed().as_micros();

    // ── Phase 4: generic loop / spinning detection + task pin ───
    // Its verdict only steers `<tool_call>`: none without tool-call steering.
    let loop_detect::LoopDetectOut {
        suppress_tool_call,
        tool_call_repeat_count,
    } = loop_detect::check_loops(
        &req.messages,
        tools_active && state.behavior.tool_call_steering,
        repeat_run,
    );
    let us_loop_detect = _t_seg.elapsed().as_micros() - us_prepare;

    let session_hash = crate::session_key::chat_session_hash(session_key, &prompt_tokens);
    let us_session_hash = _t_seg.elapsed().as_micros() - us_prepare - us_loop_detect;
    let tools_count = req.tools.len();
    tracing::info!(
        "Session {session_hash:#x}: {prompt_tokens} prompt tokens, tools={tools_active} ({tools_count} defined)",
        prompt_tokens = prompt_tokens.len()
    );
    let prompt_len = prompt_tokens.len();
    if prompt_len >= state.max_seq_len {
        return ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt too long: {prompt_len} tokens exceeds max_seq_len {} (leave room for output tokens)",
                state.max_seq_len
            ),
        ));
    }

    // ── Phase 6: sampling preset / stop / grammar / timeout ─────
    let sampling_setup::SamplingSetup {
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        max_tokens,
        stop_tokens,
        tool_choice_required,
        grammar_spec,
        timeout_at,
        top_logprobs,
    } = match sampling_setup::build_sampling(
        &state,
        &req,
        enable_thinking,
        tools_active,
        suppress_tool_call,
        tool_call_repeat_count,
    ) {
        Ok(s) => s,
        Err(resp) => return ChatOutcome::Http(resp),
    };
    if prompt_len.saturating_add(max_tokens) > state.max_seq_len {
        return ChatOutcome::Http(openai_error_response(
            StatusCode::BAD_REQUEST,
            format!(
                "Prompt plus max_tokens exceeds max_seq_len {} (prompt={}, max_tokens={})",
                state.max_seq_len, prompt_len, max_tokens
            ),
        ));
    }
    if state.chat.phase_timing {
        let us_sampling =
            _t_seg.elapsed().as_micros() - us_prepare - us_loop_detect - us_session_hash;
        tracing::info!(
            "CHAT_PHASE handler: prepare={us_prepare}us loop_detect={us_loop_detect}us \
             session_hash={us_session_hash}us sampling_and_grammar={us_sampling}us \
             total_pre_dispatch={}us",
            _t_seg.elapsed().as_micros()
        );
    }

    // ── Phase 7: dispatch streaming or blocking ─────────────────
    if req.stream {
        return super::super::chat_stream_dispatch::dispatch_streaming(
            state,
            &req,
            req_ctx,
            dump_seq,
            prompt_tokens,
            session_hash,
            adapter_slot,
            src_lang_id,
            tgt_lang_id,
            num_beams,
            length_penalty,
            early_stopping,
            image_pixels,
            max_tokens,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            lz_penalty,
            logit_bias.clone(),
            enable_thinking,
            thinking_budget,
            force_nonempty_content,
            tools_active,
            tool_choice_required,
            suppress_tool_call,
            cwd_hint.clone(),
            stop_tokens,
            grammar_spec.clone(),
            top_logprobs,
            timeout_at,
            active_guard,
        )
        .await;
    }

    super::super::chat_blocking::run_blocking_path(super::super::chat_blocking::BlockingPathArgs {
        state,
        req,
        req_ctx,
        prompt_tokens,
        session_hash,
        adapter_slot,
        src_lang_id,
        tgt_lang_id,
        num_beams,
        length_penalty,
        early_stopping,
        image_pixels,
        max_tokens,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        lz_penalty,
        logit_bias,
        stop_tokens,
        enable_thinking,
        thinking_budget,
        force_nonempty_content,
        tools_active,
        tool_choice_required,
        suppress_tool_call,
        grammar_spec,
        top_logprobs,
        timeout_at,
        cwd_hint,
        prompt_len,
    })
    .await
}
