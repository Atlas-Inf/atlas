// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(crate) fn handle_token_inner(state: &mut StreamState, ctx: &StreamCtx, tok: u32) -> DeltaVec {
    let mut deltas: DeltaVec = Vec::new();
    state.generated_tokens = state.generated_tokens.saturating_add(1);
    state.all_toks.push(tok);
    // One push per call == one sampled token == one increment of
    // `usage.completion_tokens`. Drained onto the next client-visible
    // chunk (return_token_ids); a no-op clone-free take when opted out.
    if ctx.req_return_token_ids {
        state.pending_token_ids.push(tok);
    }

    // GLM's native opener ends reasoning and must also reach the existing
    // content/tool parser. The startup carrier is model-scoped, not inferred
    // from the display name or from tool-shaped reasoning text.
    let native_marker_is_literal = ctx.state.glm_tool_boundary == Some(tok)
        && state.detector.is_some()
        && state.all_toks.len() > 1
        && ctx
            .state
            .tokenizer
            .decode(&state.all_toks[..state.all_toks.len() - 1])
            .ok()
            .is_some_and(|prefix| state_after(&prefix).is_literal());
    let implicit_tool = !state.thinking_done
        && state.detector.is_some()
        && ctx.state.glm_tool_boundary == Some(tok)
        && !native_marker_is_literal;
    // ── Thinking-phase: token-ID based boundary detection ──────────
    if !state.thinking_done && (ctx.state.think_end_token_id == Some(tok) || implicit_tool) {
        // Keep the opener out of a residual reasoning delta's token IDs;
        // it will be restored once, with the retained content token below.
        if implicit_tool && ctx.req_return_token_ids {
            state.pending_token_ids.pop();
        }
        state.thinking_done = true;
        // Emit only the residual reasoning delta not yet sent
        // by incremental streaming (e.g. trailing bytes held
        // back due to incomplete UTF-8 at prior token boundary).
        // The full reasoning has already been streamed
        // incrementally via reasoning_chunk deltas above —
        // re-emitting the full text here would double it.
        if ctx.enable_thinking && state.all_toks.len() > 1 {
            let full = ctx
                .state
                .tokenizer
                .decode(&state.all_toks[..state.all_toks.len() - 1])
                .unwrap_or_default();
            let stable = full.trim_end_matches('\u{FFFD}');
            if stable.len() > state.emitted {
                let residual = &stable[state.emitted..];
                // Same fix as the in-loop emit: whitespace-only residuals
                // are legitimate `\n   ` indents that the model emitted;
                // dropping them would lose chars permanently.
                if !residual.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: residual.to_string(),
                        token_ids: state.take_ids_if(ctx.req_return_token_ids),
                    });
                }
            }
        }
        // Flush the reasoning sanitizer's tail buffer. Without this, up to
        // ~18 trailing bytes of the final thinking block (or anything held
        // back for partial-tag fusion) are silently dropped. Skip when
        // suppression is active (no close arrived during thinking) — those
        // bytes are intentionally not surfaced.
        if !state.reasoning_suppressing_leak && !state.reasoning_tag_scan_buf.is_empty() {
            let tail = std::mem::take(&mut state.reasoning_tag_scan_buf);
            // Whitespace-only tail can be a real trailing `\n   ` indent
            // — emit anything non-empty so byte boundaries align.
            if !tail.is_empty() {
                deltas.push(StreamDelta::Reasoning {
                    text: tail,
                    token_ids: Vec::new(),
                });
            }
        }
        // Reset tool detector to clear any thinking-era tag fragments.
        if let Some(ref mut det) = state.detector {
            det.reset();
        }
        state.emitted = 0; // Reset — next decode will be content-only
        state.all_toks.clear(); // Clear thinking tokens from accumulator
        state.content_decoded.clear();
        state.detok_prefix_offset = 0;
        state.detok_read_offset = 0;
        if !implicit_tool {
            return deltas;
        }
        state.all_toks.push(tok);
        if ctx.req_return_token_ids {
            state.pending_token_ids.push(tok);
        }
    }
    if !state.thinking_done {
        // Still in thinking — accumulate but don't emit as content
        if ctx.enable_thinking {
            // Layer-A one-shot guard: after the in-think tool-call leak
            // scanner has fired, suppress all subsequent reasoning
            // deltas for this stream. The scheduler's `cancel_flag`
            // (set when the scanner fired) finalises the sequence
            // within one token via `emit_step::emit_token`; this
            // guard catches the in-flight token race so the next
            // opener never reaches the client.
            if state.reasoning_xml_leak_detected {
                return deltas;
            }
            // Open thinking: emit as reasoning_content. Incrementally extend
            // the stable decoded text instead of re-decoding all_toks (O(n²)).
            let delta_stable = ctx.state.tokenizer.incremental_decode(
                &state.all_toks,
                &mut state.detok_prefix_offset,
                &mut state.detok_read_offset,
            );
            state.content_decoded.push_str(&delta_stable);
            let stable_end = state.content_decoded.len();
            if stable_end > state.emitted {
                let raw = state.content_decoded[state.emitted..stable_end].to_string();
                let mut cleaned = raw.clone();
                let reasoning_literal_before = state.reasoning_literal_xml_state;
                state.emitted = stable_end;
                // Strip format tokens that shouldn't appear in thinking.
                // `<think>` only fires at the literal opener (always
                // whitespace-adjacent in the prompt), so a plain replace
                // is safe here.
                cleaned = cleaned.replace("<think>", "");
                if let Some(rest) = cleaned.strip_prefix("assistant\n") {
                    cleaned = rest.to_string();
                } else if let Some(rest) = cleaned.strip_prefix("assistant") {
                    cleaned = rest.to_string();
                }
                // Qwen4 treats tool markup as ordinary text until the
                // explicit think-end token (Enntity/atlas 69d32d3e, AGPL).
                // Enntity/atlas: literal-XML carriers preserve textual
                // spellings in reasoning; the native-opener token was
                // already handled by token ID above (69d32d3e/1c1a8fee).
                if ctx.state.tokenizer.glm_tool_markup_as_text {
                    if !state.reasoning_suppressing_leak && !cleaned.is_empty() {
                        deltas.push(StreamDelta::Reasoning {
                            text: cleaned,
                            token_ids: state.take_ids_if(ctx.req_return_token_ids),
                        });
                    }
                    return deltas;
                }
                // Boundary-preserving strip: see `strip_preserving_boundary`
                // doc — prevents `the<tool_call>...</tool_call>project`
                // from collapsing to `theproject`.
                while let Some((start, _)) = first_marker_outside_literal(
                    &cleaned,
                    reasoning_literal_before,
                    &["<tool_call>"],
                ) {
                    if let Some((end_rel, end_len)) = first_marker_outside_literal(
                        &cleaned[start..],
                        reasoning_literal_before,
                        &["</tool_call>"],
                    ) {
                        let end = start + end_rel + end_len;
                        cleaned = strip_preserving_boundary(&cleaned, start, end);
                    } else {
                        cleaned = cleaned[..start].to_string();
                        break;
                    }
                }
                if let Some((start, _)) = first_marker_outside_literal(
                    &cleaned,
                    reasoning_literal_before,
                    &["<function="],
                ) {
                    cleaned = cleaned[..start].to_string();
                }
                // Strip leaked tool-call closing tags from reasoning
                // (observed pattern: `</parameter></function>` right
                // before a role-word repetition loop). Route through
                // `strip_all_preserving_boundary` (2026-05-23 sweep)
                // to avoid gluing words when a closing tag straddles
                // two reasoning sentences.
                for tag in &["</parameter>", "</function>", "</tool_call>"] {
                    while let Some((start, len)) =
                        first_marker_outside_literal(&cleaned, reasoning_literal_before, &[*tag])
                    {
                        cleaned =
                            strip_all_preserving_boundary(&cleaned, &cleaned[start..start + len]);
                    }
                }
                // Collapse role-word repetition loops (Qwen3.5/3.6
                // post-tool-call hallucination): `userX...userX` →
                // "" until no adjacent pairs remain, then strip
                // line-bounded standalones (`\nuser\n` → `\n`).
                for word in &["user", "assistant", "tool"] {
                    let pair = format!("{word}{word}");
                    cleaned = strip_all_preserving_boundary(&cleaned, &pair);
                    let nl_form = format!("\n{word}\n");
                    while cleaned.contains(&nl_form) {
                        cleaned = cleaned.replace(&nl_form, "\n");
                    }
                }
                maybe_log_decode_trace(&raw, &cleaned, stable_end, stable_end - raw.len());
                // Layer-A in-think tool-call leak scanner. The per-
                // delta strippers above can miss boundary splits
                // (e.g. `<too` in delta N + `l_call>` in delta N+1)
                // and even when they strip, the model keeps emitting
                // the next repetition because its own KV already
                // contains the literal opener. This sliding-window
                // detector across deltas catches the opener on
                // arrival, drops the delta, sets the loop-cap flag
                // (→ finish_reason="length" via the PR #87 override)
                // and flips the scheduler cancel_flag so generation
                // terminates within one token via PR #89.
                let tools_active_request =
                    !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
                if tools_active_request {
                    state.reasoning_xml_scan_buf.push_str(&cleaned);
                    if state.reasoning_xml_scan_buf.len() > 256 {
                        let drop_to = state.reasoning_xml_scan_buf.len() - 256;
                        let cut = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .find(|&(i, _)| i >= drop_to)
                            .map(|(i, _)| i)
                            .unwrap_or(state.reasoning_xml_scan_buf.len());
                        let dropped = state.reasoning_xml_scan_buf[..cut].to_string();
                        state.reasoning_xml_scan_state.feed(&dropped);
                        state.reasoning_xml_scan_buf.drain(..cut);
                    }
                    let opener = first_marker_outside_literal(
                        &state.reasoning_xml_scan_buf,
                        state.reasoning_xml_scan_state,
                        &["<tool_call>", "<function=", "<parameter=", "<invoke "],
                    )
                    .and_then(|(at, len)| {
                        let marker = state.reasoning_xml_scan_buf[at..at + len].to_string();
                        Some((at, marker))
                    });
                    if let Some((at, op)) = opener {
                        // Snapshot the log tail BEFORE consuming the match,
                        // so the WARN still shows the opener in context.
                        let tail_start = state
                            .reasoning_xml_scan_buf
                            .char_indices()
                            .rev()
                            .nth(63)
                            .map(|(i, _)| i)
                            .unwrap_or(0);
                        let tail = state.reasoning_xml_scan_buf[tail_start..].to_string();
                        // Consume the buffer through this opener so ONE
                        // occurrence is counted once, not on every
                        // subsequent delta the rolling tail still contains
                        // it in.
                        let consumed = state.reasoning_xml_scan_buf[..at + op.len()].to_string();
                        state.reasoning_xml_scan_state.feed(&consumed);
                        state.reasoning_xml_scan_buf.drain(..at + op.len());
                        state.reasoning_xml_opener_hits =
                            state.reasoning_xml_opener_hits.saturating_add(1);
                        let threshold = ctx.state.chat.in_think_leak_openers;
                        if threshold != 0 && state.reasoning_xml_opener_hits >= threshold {
                            state.reasoning_xml_leak_detected = true;
                            // Name the cut for the --dump body / Done event;
                            // `tool_loop_capped` wires finish_reason
                            // "length" (the shipped doom-loop contract).
                            state.guard_stop = Some("in_think_tool_leak");
                            state.tool_loop_capped = true;
                            state.stop_string_triggered = true;
                            state
                                .cancel_flag
                                .store(true, std::sync::atomic::Ordering::Release);
                            tracing::warn!(
                                model = %ctx.model,
                                request_id = %ctx.id,
                                opener = op,
                                hits = state.reasoning_xml_opener_hits,
                                threshold,
                                tail = %tail,
                                "in-think tool-call leak: opener threshold reached; cancelling \
                                 sequence (finish_reason \"length\", guard in_think_tool_leak). \
                                 Raise/disable via ATLAS_INTHINK_TOOL_LEAK_OPENERS (0 = strip-only)"
                            );
                            return deltas;
                        }
                        // Below threshold: keep streaming. A model whose
                        // native tool syntax surfaces in legitimate
                        // reasoning (qwen3_coder family) lands here instead
                        // of losing the whole turn. Same-delta openers were
                        // stripped by the cleanups above; an opener SPLIT
                        // across deltas may have partially reached the
                        // client's reasoning channel — cosmetic, and true
                        // of the pre-knob code's one free delta too.
                        tracing::warn!(
                            model = %ctx.model,
                            request_id = %ctx.id,
                            opener = op,
                            hits = state.reasoning_xml_opener_hits,
                            threshold,
                            "in-think tool-call opener observed in reasoning; below \
                             ATLAS_INTHINK_TOOL_LEAK_OPENERS threshold, not cancelling"
                        );
                    }
                }
                // F19: final structured sanitisation pass catches
                // any leak markers the hand-rolled cleanups missed.
                let cleaned = sanitize_content_chunk_with_literal_state(
                    &cleaned,
                    &mut state.reasoning_tag_scan_buf,
                    &mut state.reasoning_suppressing_leak,
                    &mut state.reasoning_inside_envelope,
                    &mut state.reasoning_literal_xml_state,
                    &ctx.leak_markers,
                );
                // Emit whitespace-only chunks too. The `sanitize_content_chunk`
                // holdback can roll out runs of `\n   ` (newline + indent) as
                // a single committed chunk when the suffix exceeds tag_max
                // chars; dropping those via `trim().is_empty()` permanently
                // loses byte boundaries because `state.emitted` already
                // advanced past them. Symptom: streamed reasoning has
                // `**\n -Calculate` where the model actually emitted
                // `**\n   - Calculate` — verified byte-for-byte against the
                // non-streaming response on temp=0 seed=42 (live A/B
                // 2026-05-25). Drop only TRULY empty chunks.
                if !cleaned.is_empty() {
                    deltas.push(StreamDelta::Reasoning {
                        text: cleaned,
                        token_ids: state.take_ids_if(ctx.req_return_token_ids),
                    });
                }
            }
        }
        return deltas;
    }

    // ── Content phase: full-decode + slice (matches reasoning path) ──
    //
    // Previously this path used the HF `tokenizers` crate's
    // `DecodeStream` (`decoder.step(tok)`). That incremental decoder
    // drops the leading metaspace byte at certain BPE-token boundaries
    // for byte-level tokenizers like Qwen's GPT-2-style BPE — verified
    // live 2026-05-25 against the FP8 Qwen3.6 model, opencode session
    // `ses_1a0e59bc7ffeFKSvtvWqoswsll`: tool-call `<parameter=content>`
    // for a Cargo.toml emitted `name = test-rust-axum-v32version =
    // 0.1.0edition = 2021` (no newlines between fields, no quotes
    // around values). Non-streaming `tokenizer.decode(&all_toks)`
    // for the same tokens produces the correct multi-line TOML.
    //
    // The fix: mirror the reasoning path — keep `state.all_toks`
    // populated with content tokens (already done at line 86), decode
    // the cumulative list, and emit the byte slice that's stable past
    // `state.emitted`. `trim_end_matches('\u{FFFD}')` defers any
    // incomplete UTF-8 multi-byte sequence at the tail until the next
    // token completes it. `state.all_toks` and `state.emitted` are
    // reset at `</think>` (line 147), so this slice references the
    // post-thinking content only.
    // Incrementally extend the stable decoded text instead of re-decoding the
    // whole `all_toks` list every token (O(n²)). `content_decoded` stays
    // byte-identical to the previous `decode(&all_toks)` trimmed of any
    // trailing incomplete-multibyte token.
    let delta_stable = ctx.state.tokenizer.incremental_decode(
        &state.all_toks,
        &mut state.detok_prefix_offset,
        &mut state.detok_read_offset,
    );
    state.content_decoded.push_str(&delta_stable);
    let stable_end = state.content_decoded.len();
    let _ = tok; // tok already in state.all_toks via line 86
    let mut delta = if stable_end > state.emitted {
        let raw = state.content_decoded[state.emitted..stable_end].to_string();
        state.emitted = stable_end;
        raw
    } else {
        return deltas;
    };
    // Retire the lazy `content_decoder` field — kept in StreamState
    // only to avoid a wider state-struct migration. The HF decoder is
    // no longer the source of truth.
    let _ = &state.content_decoder;

    // Strip residual think tags from content after thinking is done.
    if state.thinking_done {
        // SSOT with the blocking path (api/chat_blocking.rs), which had no
        // equivalent scrub and therefore leaked '</think>' into content.
        delta = crate::api::strip::scrub_think_markers(&delta);
        // If model re-opens <think>, suppress content from <think> onward.
        if let Some(pos) = delta.find("<think>") {
            delta = delta[..pos].to_string();
            state.thinking_done = false;
            state.all_toks.clear();
            state.emitted = 0;
            state.content_decoded.clear();
            state.detok_prefix_offset = 0;
            state.detok_read_offset = 0;
        }
        // Orphan tool-call markup in content when NO tools are defined: the
        // model emits a spurious `<tool_call>…` block after a normal answer and
        // the tool detector (which would strip it) is not running. Cut content
        // at the opener — SSOT with the blocking path's strip_orphan_tool_markup.
        // When tools ARE active the detector owns tool markup, so leave it.
        let tools_active_request =
            !ctx.tool_defs_for_backfill.is_empty() || state.detector.is_some();
        if !tools_active_request {
            delta = crate::api::strip::strip_orphan_tool_markup_with_literal_state(
                &delta,
                &state.literal_xml_state,
            );
        }
    }

    // Bare role-literal leak (Qwen3.5/3.6) — companion to the
    // scheduler-side <|im_start|> hard-stop. Suppressed mid tool-call
    // body: there a standalone `tool` token is the leading BPE fragment
    // of a `tool_*` NAME (issue #222) being reassembled, not a role leak.
    // ChatML formats only: elsewhere `user`, `assistant` and `tool` are
    // ordinary words, and GLM streams " tool" as one token, so the strip
    // deleted it from prose ("I don't have a that can ...").
    if ctx.role_literal_leaks {
        let inside_tool_call = state
            .detector
            .as_ref()
            .is_some_and(|d| d.inside_tool_call());
        strip_bare_role_literal(&mut delta, inside_tool_call);
    }

    if delta.is_empty() {
        return deltas;
    }

    // Multi-token stop sequences via string matching, with a vLLM-style
    // hold-back buffer (see `vllm/v1/engine/detokenizer.py`
    // `IncrementalDetokenizer.update`). All the state mutation lives in
    // `apply_stop_string_holdback` so the algorithm can be unit-tested
    // without spinning up a full `StreamCtx`.
    if !ctx.stop_strings.is_empty() && !state.stop_string_triggered {
        delta = apply_stop_string_holdback(
            &delta,
            &ctx.stop_strings,
            ctx.stop_string_buffer_len,
            &mut state.accumulated_content,
            &mut state.stop_string_emitted_len,
            &mut state.stop_string_triggered,
        );
        // Entry was gated on `!triggered`, so a flip here is a GENUINE
        // client stop-sequence match (the watchdog paths set the flag
        // elsewhere): record it for the wire finish_reason ("stop", per
        // OpenAI) and cancel generation.
        if state.stop_string_triggered {
            state.note_stop_string_match();
        }
        if delta.is_empty() {
            // Either everything is sitting in the hold-back window
            // (waiting for the next chunk / stream close) or a match
            // already truncated the emittable bytes to nothing.
            return deltas;
        }
    }

    if state.stop_string_triggered {
        // Only the current token's genuine client stop match can reach this
        // branch. Emit its pre-stop prefix once, preserving markup sanitization.
        // Tokens queued AFTER a guard or stop match are rejected at handle_token
        // entry, before detokenization, content accumulation or token-ID tracking.
        if !delta.is_empty() {
            let cleaned = sanitize_content_chunk_with_literal_state(
                &delta,
                &mut state.tag_scan_buf,
                &mut state.suppressing_param_leak,
                &mut state.inside_envelope,
                &mut state.literal_xml_state,
                &ctx.leak_markers,
            );
            if !cleaned.is_empty() {
                deltas.push(StreamDelta::Content {
                    text: cleaned,
                    token_ids: state.take_ids_if(ctx.req_return_token_ids),
                });
            }
        }
        return deltas;
    }

    // Fork: detector-active vs pure-content path.
    if state.detector.is_some() {
        // Drain the detector outputs into a local Vec so we can drop
        // the &mut borrow on `state.detector` before the helpers below
        // (which take other &mut state fields) run.
        let outputs = {
            let det = state.detector.as_mut().expect("detector is Some");
            det.process(&delta)
        };
        for output in outputs {
            match output {
                tool_parser::DetectorOutput::Content(text) => {
                    if let Some(events_out) = detector_content_arm(state, ctx, &text) {
                        deltas.extend(events_out);
                        // One token can yield text before, inside and after a
                        // literal marker (Enntity/atlas fbda04d1); drain every
                        // event unless a guard actually cancelled this response.
                        if state.stop_string_triggered {
                            return deltas;
                        }
                    }
                }
                tool_parser::DetectorOutput::ToolCall(mut tc, tc_idx) => {
                    handle_complete_tool_call(state, ctx, &mut tc, tc_idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallStart {
                    id: tc_id,
                    name,
                    idx,
                } => {
                    handle_tool_call_start(state, ctx, tc_id, name, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                    handle_tool_call_delta(state, ctx, args, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                    handle_tool_call_args_fragment(state, ctx, fragment, idx, &mut deltas);
                }
                tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                    handle_tool_call_end(state, ctx, idx, &mut deltas);
                }
            }
            if state.guard_stop == Some(atlas_tool_guard::GUARD_STOP) {
                deltas.clear();
                return deltas;
            }
        }
    } else {
        let sanitized = sanitize_content_chunk_with_literal_state(
            &delta,
            &mut state.tag_scan_buf,
            &mut state.suppressing_param_leak,
            &mut state.inside_envelope,
            &mut state.literal_xml_state,
            &ctx.leak_markers,
        );
        if let Some(events_out) = process_detector_content(state, ctx, &sanitized) {
            deltas.extend(events_out);
            return deltas;
        }
        // process_detector_content does NOT pre-sanitize when called
        // from the no-detector branch — but the sanitizer was already
        // run above, so the helper's branch handling matches.
    }

    deltas
}
