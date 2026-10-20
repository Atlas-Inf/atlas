// SPDX-License-Identifier: AGPL-3.0-only

//! process_decode_logits: post-decode logits processing.

use super::*;

// The reusable host staging buffer for the D2H logits copy is now
// `SchedCtx::scratch.host_bytes` — same zero-contention single-thread
// access, with a lifetime that ends when the run does.

thread_local! {
    /// Dequant scratch for ONE rayon worker on the parallel host-sampling arm.
    ///
    /// The run-owned `SchedCtx::scratch` is a `RefCell`, so it is neither
    /// `Sync` nor shareable across the pool; a fan-out needs one buffer per
    /// worker or it double-borrows. The serial arm — and every other caller —
    /// still uses the run's scratch, so this exists only for the `n > 1`
    /// fan-out and holds nothing model-derived: `copy_logits_to_host`
    /// overwrites every byte it reads, and `seq_f32` is resized per call.
    static PAR_SAMPLE_SCRATCH: crate::scheduler::sched_ctx::DecodeScratch =
        crate::scheduler::sched_ctx::DecodeScratch::default();
}

/// Build the pre-sample pipeline's context around a chosen scratch buffer.
///
/// SSOT: the serial arm and each parallel worker differ ONLY in which
/// `DecodeScratch` they borrow, so the other nine fields are assembled once
/// here rather than spelled twice.
fn logits_ctx<'a>(
    sched: &'a crate::scheduler::sched_ctx::SchedCtx,
    scratch: &'a crate::scheduler::sched_ctx::DecodeScratch,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
) -> crate::scheduler::logit_processors::LogitsContext<'a> {
    crate::scheduler::logit_processors::LogitsContext {
        glm_tool_boundary: sched.limits.glm_tool_boundary,
        limits: sched.limits,
        think_end_token,
        think_start_token,
        tool_call_start_token,
        tool_call_end_token,
        watchdog: sched.watchdog,
        scratch,
        dumps: &sched.dumps,
        stats: sched.stats.clone(),
        boundary_mask: sched.masks.boundary.clone(),
        mid_word_mask: sched.masks.mid_word.clone(),
        sampling: sched.levers.sampling(),
        timing: sched.timing.clone(),
    }
}

/// Admit `think_ended` rows (which need only a 2-token mask) to the GPU argmax
/// fast path. Kill switch: `ATLAS_NO_THINKENDED_GPU_ARGMAX=1`.
fn think_ended_gpu_argmax_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_NO_THINKENDED_GPU_ARGMAX")
            .ok()
            .as_deref()
            != Some("1")
    })
}

/// Steps that fell back to the host path because a GPU argmax landed on a
/// masked think token. Expected to be a small fraction; a large count means the
/// fast path is not paying for itself.
static THINK_MASK_FALLBACKS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Parallel host sampling toggle (ATLAS_PARALLEL_SAMPLE, default ON). Set to
/// "0" to force the serial per-seq sampling path — an escape hatch for the
/// telemetry-ordering caveat above, or for A/B measurement of the rayon win.
fn parallel_sample_enabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_PARALLEL_SAMPLE").as_deref() != Ok("0"))
}

/// DIAG (ATLAS_DECODE_TIMING=1): localize the host-path decode cost. Splits the
/// per-token wall into `copy` (D2H of the full 248k-vocab logits + the GPU
/// forward-wait absorbed by that sync) vs `sample` (the host scalar loops over
/// 248k: BF16→FP32 expand + penalties + masks + argmax). Emits a 100-token
/// running summary. Zero-cost when the env var is unset (OnceLock-gated).
fn decode_timing_record(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    copy_us: u64,
    sample_us: u64,
) {
    use std::sync::atomic::Ordering;
    if !sched.levers.decode_timing {
        return;
    }
    let stats = &sched.stats;
    stats.decode_copy_us.fetch_add(copy_us, Ordering::Relaxed);
    stats
        .decode_sample_us
        .fetch_add(sample_us, Ordering::Relaxed);
    let n = stats.decode_count.fetch_add(1, Ordering::Relaxed) + 1;
    if n.is_multiple_of(100) {
        let c = stats.decode_copy_us.swap(0, Ordering::Relaxed);
        let s = stats.decode_sample_us.swap(0, Ordering::Relaxed);
        stats.decode_count.store(0, Ordering::Relaxed);
        tracing::info!(
            "DECODE_TIMING (last 100 host-path tokens): copy+fwd-wait={:.2}ms/tok sample(248k host)={:.2}ms/tok",
            c as f64 / 100_000.0,
            s as f64 / 100_000.0,
        );
    }
}

/// Sample and process decode logits for all active sequences.
///
/// Factored out of `step_decode_only` so that `mixed_forward` can reuse
/// the same sampling + token-processing logic without duplication (SSOT).
/// `logits` must point to `[n, vocab_size]` BF16 on device where n = active.len().
pub fn process_decode_logits(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    logits: DevicePtr,
    t0: std::time::Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    let n = active.len();

    // Grammar bitmask is CPU-side, so any sequence with active grammar forces
    // the host-side sampling path for its logits slice.
    let any_grammar = active.iter().any(|a| a.grammar_state.is_some());
    let any_logprobs = active.iter().any(|a| a.top_logprobs.is_some());
    // FP32 lm_head models (Gemma-4 dense) MUST use the host-side path —
    // `argmax_batch` assumes BF16 layout and would interpret 4-byte FP32
    // values as 2-byte BF16 pairs, returning garbage tokens.
    let model_logits_fp32 = model.decode_logits_fp32();
    // A row may take the GPU argmax only where the host pipeline provably
    // keeps it (`fast_greedy::raw_argmax_is_pick`: outside `<think>`, neutral
    // penalties, no bias, no tool-call pin); an argmax on an id the pipeline
    // masks there is redone on the host below. One rule for every row, so a
    // row's pick does not depend on what else is in the batch: never-thinking
    // rows used to take the GPU argmax with their penalties and bias ignored,
    // but had them applied whenever another row forced the host path.
    //
    // `--disable-thinking` sets `think_ended` for EVERY sequence at birth, so
    // admitting `think_ended` rows keeps that config (the MLPerf-edge one) off
    // the 7.95 MB D2H + n full-vocab host passes. Kill switch:
    // ATLAS_NO_THINKENDED_GPU_ARGMAX=1 sends them to the host.
    let admit_think_ended = think_ended_gpu_argmax_enabled();
    let needs_host_logits = active.iter().any(|a| {
        !crate::scheduler::fast_greedy::raw_argmax_is_pick(a)
            || (a.think_ended && !admit_think_ended)
    }) || any_logprobs
        || model_logits_fp32;

    // Try the GPU argmax first. `None` here means "not eligible, or the result
    // needs the host pipeline after all" and falls through to the host branch —
    // it must never mean "emit nothing".
    let fast_tokens: Option<Vec<(u32, Option<crate::api::TokenLogprobs>)>> =
        if active.iter().all(|a| a.temperature == 0.0) && !any_grammar && !needs_host_logits {
            match model.argmax_batch(logits, n, 0) {
                Ok(mut t) => {
                    // The min_tokens end-token ban: each row's masked argmax
                    // (a failed row read sends the batch to the host).
                    let vocab = model.vocab_size();
                    let served = active.iter().enumerate().all(|(i, a)| {
                        let row = logits.offset(i * vocab * 2);
                        crate::scheduler::min_tokens_ban::fix_raw_picks(
                            model,
                            a,
                            &mut t[i..=i],
                            row,
                            false,
                        )
                        .unwrap_or(false)
                    });
                    // An argmax on an id the pipeline masks (rare: the model
                    // seldom re-opens <think> mid-response) is redone on the
                    // host, so the emitted token is exactly the pipeline's.
                    let hit_mask = !served
                        || t.iter().zip(active.iter()).any(|(&tok, a)| {
                            crate::scheduler::fast_greedy::raw_pick_masked(a, tok)
                        });
                    if hit_mask {
                        THINK_MASK_FALLBACKS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        None
                    } else {
                        Some(t.into_iter().map(|tok| (tok, None)).collect())
                    }
                }
                Err(e) => {
                    tracing::error!("argmax_batch error: {e:#}");
                    for mut a in active.drain(..) {
                        send_error(model, &mut a, &format!("{e:#}"));
                    }
                    return;
                }
            }
        } else {
            None
        };

    let new_tokens: Vec<(u32, Option<crate::api::TokenLogprobs>)> = if let Some(t) = fast_tokens {
        t
    } else {
        // Host-side path: copy all batch logits to host, sample per-sequence.
        // Required when any sequence has temperature > 0 or grammar constraints.
        let vocab_size = model.vocab_size();
        // FP32 lm_head dispatch (Gemma-4 dense). When `use_fp32_logits` is
        // on, the per-token decode lm_head writes 4 bytes/element. The
        // passed `logits` pointer is whatever the most-recent forward
        // returned — that's already the correct buffer (prefill or decode).
        // We just need to read it with the matching width.
        let logits_fp32 = model.decode_logits_fp32();
        let elem_bytes = if logits_fp32 { 4 } else { 2 };
        let t_copy = std::time::Instant::now();
        // Reuse the run's staging buffer (restored at the end of this block).
        // `resize` only grows it; `copy_logits_to_host` overwrites every byte
        // so the residual/zero-fill is irrelevant.
        let mut buf = sched.scratch.host_bytes.borrow_mut().split_off(0);
        buf.resize(n * vocab_size * elem_bytes, 0);
        if let Err(e) = model.copy_logits_to_host(logits, &mut buf) {
            tracing::error!("copy_logits_to_host error: {e:#}");
            for mut a in active.drain(..) {
                send_error(model, &mut a, &format!("{e:#}"));
            }
            return;
        }
        let copy_us = t_copy.elapsed().as_micros() as u64;
        let t_sample = std::time::Instant::now();
        // Per-sequence host sampling is independent: each call reads a
        // disjoint `buf` slice, mutates its own `ActiveSeq`, uses its own
        // dequant scratch, and advances a per-seq seed. The collect is
        // order-preserving, so the emitted tokens are identical to the serial
        // path. (Process-global sampler telemetry — `sampler::LAST_ENTROPY`,
        // the AdaDec/B1 diagnostics — is synchronized but last-write-wins
        // across workers, so those best-effort gauges may report an arbitrary
        // in-step sequence's value under n>1; token output is unaffected.)
        // Each call scans the full ~250k vocab (BF16->FP32 expand + penalties
        // + argmax), the dominant host-path cost at n>=2. Fan out across the
        // rayon pool ONLY for n>1: at n=1 (the common single-stream /
        // opencode host path) the serial path avoids rayon's dispatch
        // overhead. `process_seq_logits` touches no GPU state (its `_model`
        // arg is unused), so no CUDA calls cross threads. The parallel path
        // is also gated off when the opt-in `ATLAS_LOGIT_DUMP` diagnostic is
        // active, since its shared per-step record would interleave across
        // workers.
        let parallel_sample = n > 1 && sched.dumps.logits.is_none() && parallel_sample_enabled();
        let sampled: Vec<(u32, Option<crate::api::TokenLogprobs>)> = if parallel_sample {
            use rayon::prelude::*;
            // `SchedCtx` itself is NOT `Sync` (its `DecodeScratch` is a
            // `RefCell`), so the fan-out closure must not capture it. Hoist
            // the pieces the pipeline reads — all of them `Copy`, `Arc` or
            // `&`-to-`Sync` — and leave the scratch to each worker.
            let dumps = &sched.dumps;
            let watchdog = sched.watchdog;
            let glm_tool_boundary = sched.limits.glm_tool_boundary;
            let limits = sched.limits;
            let stats = sched.stats.clone();
            let boundary_mask = sched.masks.boundary.clone();
            let mid_word_mask = sched.masks.mid_word.clone();
            let sampling = sched.levers.sampling();
            let timing = sched.timing.clone();
            active
                .par_iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    // Preserve batch row mapping but skip mutable host sampling
                    // for an already-cancelled request. The retained last token
                    // is only a placeholder: the commit gate below discards it.
                    if retire_if_cancelled(a) {
                        return (a.last_token, None);
                    }
                    // The run's `DecodeScratch` is a `RefCell` — neither
                    // `Sync` nor shareable across the pool — so each worker
                    // borrows its own and builds the context around it. Same
                    // reuse, one buffer per worker instead of one per run.
                    PAR_SAMPLE_SCRATCH.with(|scratch| {
                        let ctx = crate::scheduler::logit_processors::LogitsContext {
                            glm_tool_boundary,
                            limits,
                            think_end_token,
                            think_start_token,
                            tool_call_start_token,
                            tool_call_end_token,
                            watchdog,
                            scratch,
                            dumps,
                            stats: stats.clone(),
                            boundary_mask: boundary_mask.clone(),
                            mid_word_mask: mid_word_mask.clone(),
                            sampling,
                            timing: timing.clone(),
                        };
                        process_seq_logits(
                            model,
                            a,
                            &buf,
                            i,
                            vocab_size,
                            elem_bytes,
                            logits_fp32,
                            &ctx,
                            adaptive_sampling,
                        )
                    })
                })
                .collect()
        } else {
            let ctx = logits_ctx(
                sched,
                &sched.scratch,
                think_end_token,
                think_start_token,
                tool_call_start_token,
                tool_call_end_token,
            );
            active
                .iter_mut()
                .enumerate()
                .map(|(i, a)| {
                    if retire_if_cancelled(a) {
                        return (a.last_token, None);
                    }
                    process_seq_logits(
                        model,
                        a,
                        &buf,
                        i,
                        vocab_size,
                        elem_bytes,
                        logits_fp32,
                        &ctx,
                        adaptive_sampling,
                    )
                })
                .collect()
        };
        decode_timing_record(sched, copy_us, t_sample.elapsed().as_micros() as u64);
        // Return the staging buffer for reuse next token (its capacity is
        // preserved). The error path above intentionally drops it — that is
        // rare and only forfeits the cached capacity.
        *sched.scratch.host_bytes.borrow_mut() = buf;
        sampled
    };
    let step_ms = t0.elapsed().as_secs_f64() * 1000.0;
    if tracing::enabled!(tracing::Level::DEBUG) {
        let token_ids: Vec<u32> = new_tokens.iter().map(|(t, _)| *t).collect();
        tracing::debug!(
            "DECODE: n={n} step={step_ms:.1}ms ({:.1} tok/s) tokens={:?}",
            1000.0 * n as f64 / step_ms,
            token_ids,
        );
    }

    let now = Instant::now();
    for (i, (tok, logprobs)) in new_tokens.into_iter().enumerate() {
        let a = &mut active[i];
        // Recheck after sampling: stream cancellation can arrive during this
        // step or while a preceding independent row is being delivered. Do not
        // compact rows or undo the completed forward; lifecycle retires state.
        if retire_if_cancelled(a) {
            continue;
        }
        // A content-loop steer of this pick, counted as `emit_token` counts it.
        let steered = crate::scheduler::loop_steer::note_commit(a);
        a.last_token = tok;
        a.last_token_time = now;

        let env = CommitEnv::of(sched);
        if hard_stop(a, tok, &env) {
            continue;
        }
        first_token_thinking::apply_native_tool_boundary(a, tok, &env);
        if think_gate(a, tok, &env) {
            continue;
        }

        // Advance grammar state with the sampled token — but only
        // once thinking is finished, because thinking tokens are
        // stripped from the API output and should not consume grammar
        // slots (matches the bitmask-skip in the sampler above).
        // A strict (response_format) grammar fails loud on a refusal.
        if !a.inside_thinking
            && let Some(ref mut gs) = a.grammar_state
            && !gs.accept_token(tok)
            && gs.is_strict()
        {
            crate::scheduler::emit_step::fail_strict_grammar(a, tok);
            continue;
        }

        let prior = a.output_tokens.len();
        let native_glm_eos = crate::glm_tool_boundary::native_eos_while_thinking(
            sched.limits.glm_tool_boundary,
            a.inside_thinking,
            tok,
            &a.eos_tokens,
        );
        if a.inside_thinking {
            advance_thinking(a, tok, prior, native_glm_eos, &env);
        } else {
            // Content-phase token: budget bookkeeping + the content-loop
            // and inter-tool-prose watchdogs. Extracted to
            // `decode_logits_content.rs` to keep this file ≤500 LoC.
            // `model` is threaded through so a watchdog rollback can
            // restore SSM recurrent state on hybrid models (Phase-C).
            if handle_content_token(a, model, sched, steered) {
                // A watchdog rolled back: `tok` came from the discarded
                // context and `last_token` is the boundary token. Drop `tok`
                // (no push / emit / EOS bookkeeping for it).
                continue;
            }
        }

        // Track <tool_call> token: once seen, legacy tool call requirement is satisfied.
        // Guard with !inside_thinking — a <tool_call> inside thinking is spurious
        // and must not clear require_tool_call (which would allow premature EOS).
        if a.require_tool_call && tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.require_tool_call = false;
            a.tool_call_opened = true;
        }
        // F2 (2026-04-26): reset the inter-tool prose budget on
        // every `<tool_call>` open. This keeps the budget scoped to
        // "free-text since the last tool call started" rather than
        // accumulating across the whole response.
        if tool_call_start_token == Some(tok) && !a.inside_thinking {
            a.prose_tokens_since_last_tool = 0;
            // Tool-call-repetition runaway guard. On a `tool_choice="auto"`
            // grammar turn the grammar never terminates after a tool call
            // (stop_after_first=false), so EOS stays grammar-suppressed and the
            // only stop path is the ATLAS_TOOL_EOS_ESCAPE hatch — which a
            // re-opened tool body defeats (its `!inside_tool_body` guard flips
            // false the moment the model emits another `<tool_call>`). A
            // degenerating FP8/long-context model loops emitting whole
            // `<tool_call>…</tool_call>` blocks as content; each closes cleanly
            // so the envelope-streak guard never fires, and the turn burns to
            // max_tokens. Count opens that happen AFTER a real call already
            // completed; once past threshold the turn is provably degenerating
            // and we force-finish it (below).
            if a.tool_call_completed {
                a.post_completion_tool_opens = a.post_completion_tool_opens.saturating_add(1);
                // Threshold = how many EXTRA `<tool_call>` openers (after the
                // first completed) mark the turn as a degenerate content-leak
                // loop with no legitimate continuation. Measured: real
                // degenerate runaways emit 50-58 blocks in a single decode
                // burning to the 8192 cap; a model making genuine back-to-back
                // calls in one decode tops out far lower and then STOPS. 8 sits
                // safely above any plausible legit single-decode multi-call
                // (catches the runaway at ~8 blocks ≈ ~1.2k tokens, an order of
                // magnitude below the 8k-token cap it used to hit) while leaving
                // generous headroom so a legitimate multi-call turn is never
                // truncated. This is the only path that reliably halts the
                // runaway — lifting the post-sample EOS suppression alone is not
                // enough if the grammar bitmask never surfaces an EOS token
                // during the auto-mode alternation. Mirrors the existing
                // MAX_TOOL_BODY_TOKENS envelope guard (emit_step.rs), which
                // force-finishes the never-closing variant; this handles the
                // closing-but-repeating variant.
                const MAX_POST_COMPLETION_TOOL_OPENS: u32 = 8;
                if a.post_completion_tool_opens >= MAX_POST_COMPLETION_TOOL_OPENS {
                    tracing::warn!(
                        opens = a.post_completion_tool_opens,
                        "tool-call repetition runaway: model re-opened {MAX_POST_COMPLETION_TOOL_OPENS}+ tool-call blocks after a completed call on a tool_choice=auto turn; ending response (was burning to max_tokens). Sanitizer keeps the first valid call(s)."
                    );
                    a.output_tokens.push(tok);
                    a.tool_call_opened = true;
                    if let Some(ref mut gs) = a.grammar_state {
                        gs.accept_token(tok);
                    }
                    a.finished = true;
                    continue;
                }
            }
        }
        // Safety: if require_tool_call is still set after 512 tokens, the model
        // isn't generating a tool call (grammar may have failed to compile).
        // Clear the flag so EOS is no longer suppressed — prevents infinite gen.
        if a.require_tool_call && a.output_tokens.len() > 512 {
            tracing::warn!(
                "require_tool_call safety: no <tool_call> after 512 tokens, clearing EOS suppression"
            );
            a.require_tool_call = false;
        }

        // Accumulate logprobs data for blocking responses.
        if let Some(lp) = logprobs {
            a.logprobs_data.push(lp);
        }

        // </tool_call> handling. Tool-armed requests (grammar active OR
        // `tools_present`) continue generating past a closed call so the model
        // can emit multiple/parallel calls (#192); only a NON-tool request
        // that spuriously emits `</tool_call>` hard-stops here.
        if tool_call_end_token == Some(tok) && !a.inside_thinking {
            a.output_tokens.push(tok);
            // Fix A (2026-06-05): mark the tool call complete so the EOS-escape
            // gate (below) can lift suppression. Inert unless
            // `tool_eos_escape_enabled()` (default OFF).
            a.tool_call_completed = true;
            if let ResponseSink::Streaming(ref tx) = a.sink {
                let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                    StreamEvent::TokenWithLogprobs(tok, lp)
                } else {
                    StreamEvent::Token(tok)
                };
                if !super::mod_helpers::bounded_stream_send(tx, event, "tool_call_end") {
                    tracing::warn!(
                        "Streaming receiver dropped during tool_call_end, finishing sequence"
                    );
                    a.finished = true;
                    continue;
                }
            }
            if a.grammar_state.is_none() && !a.tools_present {
                // Plain-chat hard stop: a request with NO tools declared has no
                // business emitting `<tool_call>` blocks — end the turn (the
                // historical "legacy mode" behavior, now scoped to non-tool
                // requests only).
                //
                // #192: when tools ARE declared (`tools_present`), a closed
                // tool call no longer finishes the sequence even without an
                // active grammar (grammar disengaged mid-response on a
                // model/matcher disagreement, opted out, or disabled). vLLM
                // parity: keep decoding so the model can emit PARALLEL calls;
                // the turn ends at natural EOS (require_tool_call was cleared
                // at `<tool_call>`, so EOS is no longer suppressed) or via the
                // tool watchdogs (post-completion open cap above, prose
                // budget, loop detectors) if it runs on.
                a.finished = true;
            }
            // Mirror finish_sequence (lines ~3445-3448): keep
            // `inside_tool_body` and the grammar FSM in sync with the
            // emitted token stream. The `continue;` below skips the
            // `emit_token()` path that would normally do this, so
            // without these two lines the flag stays `true` for all
            // subsequent prose tokens — sampler penalties stay
            // disabled for the rest of the response, and the grammar
            // bitmask drifts out of sync with the actual emission.
            // Root-caused 2026-04-26 (8-agent sweep, F1).
            a.inside_tool_body = false;
            if let Some(ref mut gs) = a.grammar_state {
                gs.accept_token(tok);
            }
            // F9 companion (2026-04-26): clear `think_ended` at every
            // </tool_call> boundary so legitimate post-tool
            // re-thinking is allowed. F9 masks <think> when
            // `think_ended=true`, but between tool calls the model
            // SHOULD be allowed to re-think (MiniMax-M2 / Qwen3.6
            // pattern per project_minimax_m27_final.md). F10's
            // watchdog-fire counter still applies — repeated
            // re-thinking that loops will decay its budget.
            a.think_ended = false;
            continue;
        }

        match end_token(a, tok, prior, a.seq.seq_len, native_glm_eos, &env) {
            // Recorded for the token count, never streamed (OpenAI: the text
            // excludes the stop sequence).
            EndToken::Stop => {
                a.output_tokens.push(tok);
                crate::scheduler::emit_step::update_tool_param_state(a, tok, false);
                a.finished = true;
                continue;
            }
            // Discarded: never recorded or streamed, the model goes on.
            EndToken::Suppressed => continue,
            EndToken::No => {}
        }
        a.output_tokens.push(tok);
        // SM1 (2026-05-26): drive the tool-body / parameter-body
        // state machine from the non-spec decode path. Previously
        // only spec/verify paths called this (via emit_token),
        // leaving every dependent gate (close-tag mask, AM1, B1,
        // A1) silently dead under `mtp=false`.
        crate::scheduler::emit_step::update_tool_param_state(a, tok, false);
        // Phase-C: if this committed token is a content-phase
        // boundary token (sentence end / newline) and the model is
        // hybrid (attention + SSM), snapshot the recurrent SSM
        // state now so a later watchdog rollback to this boundary
        // can also rewind h_state/conv_state — not just the KV
        // cache. Gated to content tokens because the watchdogs that
        // roll back all fire post-`</think>`, and `apply_rollback`
        // requires every dropped token to be a content token. No-op
        // for pure-attention models / disabled rings (see
        // `rollback::snapshot_boundary_if_ssm`).
        if !a.inside_thinking {
            rollback::snapshot_boundary_if_ssm(a, model, sched);
            // #155 iter3: block-aligned Marconi checkpoint on the
            // non-MTP decode path (live SSM state is canonical here).
            model.decode_marconi_checkpoint(&mut a.seq);
        }
        // OPENCODE FIX: when the model spontaneously emits `<think>` even
        // though the request didn't ask for thinking (`enable_thinking=false`),
        // the `<think>` open token itself is suppressed (line ~1356), but
        // the thinking-content tokens that follow MUST also be kept off the
        // wire — otherwise opencode persists them as `assistant.content` and
        // on the next turn the model sees its own past garbage (fake
        // `<function=…>`, fake `<tool_response>`) as a "format example" and
        // continues the pattern. Tokens stay in `output_tokens` for the
        // blocking response path's reasoning_content extraction.
        let suppress_stream = a.inside_thinking && !a.enable_thinking;
        if let ResponseSink::Streaming(ref tx) = a.sink
            && !suppress_stream
        {
            let event = if let Some(lp) = a.logprobs_data.last().cloned() {
                StreamEvent::TokenWithLogprobs(tok, lp)
            } else {
                StreamEvent::Token(tok)
            };
            if !super::mod_helpers::bounded_stream_send(tx, event, "decode_logits token") {
                tracing::debug!("Streaming receiver dropped (decode_logits), finishing seq");
                a.finished = true;
            }
        }
        if a.remaining == 0 {
            // #144: non-MTP twin of the budget-aware close in
            // `emit_step::emit_token`. The grammar already accepted `tok`
            // above (line ~230), so it is at the current position; if it
            // is active and cannot legally stop here (open JSON string),
            // emit the shortest grammar-legal close so the length-stopped
            // output is still parseable.
            crate::scheduler::emit_step::emit_grammar_close(a);
            tracing::info!(
                "process_decode_logits: remaining=0, output_tokens={}, thinking_tokens={}",
                a.output_tokens.len(),
                a.thinking_tokens
            );
            a.finished = true;
        }
        // §C-3 (DS4F hard-limit lane, 2026-07-21): per-step context-ceiling
        // stop. Independent of thinking state and of `max_tokens` — enforces
        // the served `max_seq_len` DURING decode instead of relying on the
        // on-completion true-up (`middleware.rs`), which let a long `<think>`
        // block run KV past the ceiling (R1X overrun past max_seq_len=8192).
        // Finishes with no EOS pushed → lifecycle reports finish=length.
        // No-op when `max_seq_len` is unset (0) or not yet reached, so
        // direct-mode short answers are unaffected.
        if !a.finished && seqlen_force_stop(a.seq.seq_len, sched.limits.max_seq_len) {
            tracing::info!(
                seq_len = a.seq.seq_len,
                max_seq_len = sched.limits.max_seq_len,
                output_tokens = a.output_tokens.len(),
                "process_decode_logits: max_seq_len ceiling reached; force-stop (finish=length)"
            );
            a.finished = true;
        }
        // Grammar termination = end of sequence. With `stop_after_first=true`
        // (tool_choice="required"), the structural-tag matcher transitions
        // to its terminal state right after the single tool call closes.
        // The model's free distribution past that point can be degenerate
        // (Nemotron-Super-120B emits a `</parameter>` loop and never
        // samples EOS naturally). Finish here instead of letting it run.
        if a.grammar_state
            .as_ref()
            .is_some_and(|gs| gs.is_terminated())
        {
            a.finished = true;
        }

        // Intra-response fuzzy repetition detection: if the last 2*W tokens
        // approximately match the same W-token pattern, the model is looping.
        // Uses Hamming distance with ~12% tolerance to catch loops where the
        // model narrates the same plan with slight wording variations.
        // Skip during tool calls: XML parameter tags have natural repetition
        // (<parameter=..>...</parameter>) that triggers false positives.
        // Use last occurrence positions — completed tool calls shouldn't
        // disable the detector for subsequent text generation.
        let last_tc_start = a
            .tool_call_start_token
            .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
        let last_tc_end = a
            .tool_call_end_token
            .and_then(|t| a.output_tokens.iter().rposition(|&tok| tok == t));
        let inside_tool_call = match (last_tc_start, last_tc_end) {
            (Some(start), Some(end)) => start > end,
            (Some(_), None) => true,
            _ => false,
        };
        if sched.levers.loop_watchdog()
            && !a.finished
            && !a.inside_thinking
            && watchdog_floor_reached(a.output_tokens.len(), a.min_tokens)
            && !inside_tool_call
            && let Some((pattern_len, mis_a, mis_b)) =
                detect_fuzzy_repetition(&a.output_tokens, sched.watchdog.fuzzy_repeat_tolerance_div)
        {
            // Phase-C: roll back past the repeated window and
            // re-steer. `min_keep` = pattern_len * 3 guarantees all
            // three near-copies of the detected pattern are dropped
            // so generation cannot resume straight back into the
            // loop. Falls back to the hard stop when declined.
            let min_keep = pattern_len * 3;
            // This detector runs AFTER the step pushed `tok`, which the
            // model has not decoded yet (`unfed_tail = 1`).
            match rollback_to_boundary(a, min_keep, model, sched, 1) {
                RollbackOutcome::RolledBack { dropped } => {
                    tracing::warn!(
                        pattern_len,
                        mismatches = mis_a + mis_b,
                        dropped,
                        rollback = a.rollback_count,
                        "Fuzzy repetition detected; rolled back to boundary, re-steering"
                    );
                }
                RollbackOutcome::Fallback(reason) => {
                    tracing::warn!(
                        "Fuzzy repetition: {pattern_len}-tok pattern x3 ({mis_a}+{mis_b} \
                         mismatches), stopping at {} tokens (rollback declined: {reason:?})",
                        a.output_tokens.len()
                    );
                    a.guard_stop = Some("fuzzy_repetition");
                    a.finished = true;
                }
            }
        }

        // The request-deadline check used to live here. It has moved to
        // `mod_helpers::enforce_request_deadlines`, which runs once per
        // scheduler iteration regardless of decode path — this function
        // is not called at all on the MTP/speculative path, so the
        // deadline was unenforced in the config of record.
    }
}

#[cfg(test)]
mod auto_tool_eos_tests;
