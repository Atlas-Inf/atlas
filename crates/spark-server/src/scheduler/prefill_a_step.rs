// SPDX-License-Identifier: AGPL-3.0-only

//! start_chunked_prefill (chunked prefill orchestration).

use super::*;

mod vision_dispatch;
use vision_dispatch::dispatch_chunk0_vision;

/// Start a chunked prefill: process chunk 0, return result.
pub fn start_chunked_prefill(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    model: &dyn Model,
    mut req: InferenceRequest,
    eos_tokens: &[u32],
    max_prefill_tokens: usize,
    prefill_stream: u64,
    prefill_event: u64,
    grammar_engine: &mut Option<GrammarEngine>,
    spontaneous_think_budget: u32,
    // Co-dispatch: when true, do all per-request setup (grammar/sink/seq
    // alloc/EP broadcast) but SKIP the inline chunk-0 prefill + first-token
    // sampling, returning `InProgress { chunk_offset: 0 }` so >=2 concurrent
    // streams batch into one `run_batched_prefill_step` forward. Vision is
    // excluded upstream (PrefillInProgress carries no pixel state).
    defer: bool,
    // Vision co-dispatch: when Some, this request's images were already encoded
    // (batched with other requests) into the shared buf_out; skip the per-request
    // encode and instead set the per-stream slice base around the chunk-0 splice.
    // None ⇒ legacy self-encode. Only ever Some for single-chunk-fit image prompts.
    vision_slice: Option<VisionSlice>,
    // Beam co-dispatch: Some ⇒ this beam request's winning hypothesis was already
    // computed (fused with other requests into one batched search by the pre-pass);
    // the beam branch uses it directly. None ⇒ run the search here per-request.
    precomputed_beam_hyp: Option<Vec<u32>>,
    // Shared-prefix admission: plant a checkpoint at this position for the
    // requests waiting on this prefill (`shared_prefix`). Sent to the worker
    // before the chunk-0 command, like the MTP fence.
    pc_plant: Option<usize>,
) -> Result<StartPrefillResult> {
    // This request's end tokens: model EOS + user stop tokens (no model EOS
    // under `ignore_eos`).
    let model_eos = eos_tokens;
    let eos_tokens = &req.take_end_tokens(model_eos);

    let top_k = req.top_k();
    let top_p = req.top_p();
    let top_n_sigma = req.top_n_sigma();
    let min_p = req.min_p();
    let repetition_penalty = req.repetition_penalty();
    let presence_penalty = req.presence_penalty();
    let frequency_penalty = req.frequency_penalty();
    let dry_multiplier = req.dry_multiplier();
    let dry_base = req.dry_base();
    let dry_allowed_length = req.dry_allowed_length();
    let req_lz_penalty = req.lz_penalty();
    let logit_bias = req.logit_bias().to_vec();
    let req_min_tokens = req.min_tokens();
    let req_session_hash = req.session_hash();
    let req_adapter_slot = req.adapter_slot(); // M2 per-request LoRA routing
    let req_src_lang = req.src_lang_id();
    let req_tgt_lang = req.tgt_lang_id();
    let req_num_beams = req.num_beams();
    let req_length_penalty = req.length_penalty();
    let req_early_stopping = req.early_stopping();
    let req_enable_thinking = req.enable_thinking();
    let req_thinking_budget = req.thinking_budget();
    let req_repetition_detection = req.repetition_detection();
    if req_enable_thinking {
        tracing::info!("Thinking enabled, budget={:?}", req_thinking_budget);
    }
    let req_require_tool_call = req.require_tool_call();
    let req_tools_present = req.tools_present();
    let req_suppress_tool_call = req.suppress_tool_call();
    let req_disable_mtp = req.disable_mtp();
    let req_seed = req.seed();
    let req_top_logprobs = req.top_logprobs();
    let req_prompt_logprobs = req.prompt_logprobs();
    let req_timeout_at = req.timeout_at();
    let grammar_spec = req.take_grammar_spec();
    let (prompt_tokens, max_tokens, mut sink, image_pixels, temperature, cancel_flag) = match req {
        InferenceRequest::Streaming {
            prompt_tokens,
            max_tokens,
            temperature,
            token_tx,
            image_pixels,
            cancel_flag,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Streaming(token_tx),
            image_pixels,
            temperature,
            Some(cancel_flag),
        ),
        InferenceRequest::Blocking {
            prompt_tokens,
            max_tokens,
            temperature,
            response_tx,
            image_pixels,
            ..
        } => (
            prompt_tokens,
            max_tokens,
            ResponseSink::Blocking(Some(response_tx)),
            image_pixels,
            temperature,
            None,
        ),
    };

    // After the sink exists: a strict grammar that cannot be armed is reported
    // to the client. Strict grammars start after `</think>` (see
    // `GrammarState::masks_first_token`).
    let grammar_state = compile_grammar_state(
        grammar_engine,
        &grammar_spec,
        // The model's EOS passes through the matcher even under `ignore_eos`.
        &[model_eos, eos_tokens].concat(),
        req_enable_thinking && think_end_token.is_some(),
        &mut sink,
    )?;

    let request_start = Instant::now();
    let total = prompt_tokens.len();
    let mut chunk_len = total.min(max_prefill_tokens);
    // ATLAS_QWEN4EXP_PREFILL_ROWINV: the next chunk starts on the GDN grid.
    let g = spark_runtime::prefill_chunk_granule();
    if chunk_len < total && g > 4 && chunk_len >= g {
        chunk_len -= chunk_len % g;
    }
    let is_last = chunk_len >= total;

    tracing::info!(
        "Chunked prefill start: {} prompt tokens, chunk_size={}, max_tokens={max_tokens}",
        total,
        chunk_len,
    );

    // From here on, `sink` holds the client's response channel. Any
    // error MUST be reported via send_error_to_sink before returning,
    // otherwise the API layer will turn the dropped channel into a
    // misleading "Inference cancelled" error.
    let mut seq = match model.alloc_sequence() {
        Ok(s) => s,
        Err(e) => {
            let msg = format!("alloc_sequence failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            return Err(e);
        }
    };
    // Carry the request-local native-decode fence into model-owned state so
    // prefill hooks cannot touch shared MTP capture/eager-drafter state.
    seq.disable_mtp = req_disable_mtp;
    seq.session_hash = req_session_hash;
    seq.adapter_slot = req_adapter_slot;
    seq.src_lang_id = req_src_lang;
    seq.tgt_lang_id = req_tgt_lang;
    seq.num_beams = req_num_beams;
    seq.length_penalty = req_length_penalty;
    seq.early_stopping = req_early_stopping;
    // Task #24: resolve the STABLE adapter_id NOW (model owns the authoritative
    // active slot for the `-1 = defer to active` case). Keys the KV/prefix cache
    // so this request reuses only same-adapter blocks.
    seq.adapter_id = model.adapter_id_for(req_adapter_slot);
    // Task #25: acquire a ref on the resolved LoRA slot so a swap into it is
    // refused while this seq is in-flight. Before the `defer` branch so both the
    // InProgress co-dispatch and the inline path (and all error frees below,
    // which route through free_sequence) hold + release the ref symmetrically.
    seq.acquired_adapter_slot = model.acquire_adapter_slot(req_adapter_slot);
    seq.collect_prompt_logprobs = req_prompt_logprobs;

    // Beam search (NLLB): run the whole search to completion in the model and
    // return a FINISHED sequence carrying the winning hypothesis, bypassing both
    // the co-dispatch defer path and the chunked prefill/decode loop. Only
    // reachable for blocking requests (streaming + beam is rejected upstream).
    if model.supports_beam() && seq.num_beams > 1 {
        let hyp = match resolve_beam_hyp(
            model,
            precomputed_beam_hyp,
            &seq,
            &prompt_tokens,
            max_tokens,
        ) {
            Ok(h) => h,
            Err(e) => {
                send_error_to_sink(&mut sink, &format!("beam search failed: {e:#}"));
                let _ = model.free_sequence(&mut seq);
                return Err(e);
            }
        };
        let last = hyp.last().copied().unwrap_or(0);
        let use_legacy_tool_call =
            req_require_tool_call && grammar_state.is_none() && tool_call_start_token.is_some();
        let tool_request = tool_request_for(grammar_state.as_ref(), use_legacy_tool_call);
        let now = Instant::now();
        let cached_prompt_tok = seq.cached_prefix_tokens as u32;
        let mut a = ActiveSeq {
            seq,
            session_hash: req_session_hash,
            last_token: last,
            output_tokens: hyp,
            remaining: 0,
            min_tokens: req_min_tokens,
            eos_tokens: eos_tokens.to_vec(),
            finished: true,
            engine_error: None,
            guard_stop: None,
            param_close_pending: 0,
            sink,
            cancel_flag: cancel_flag.clone(),
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            repetition_penalty_window: 256,
            presence_penalty,
            frequency_penalty,
            lz_penalty: req_lz_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            dry_sequence_breakers: Vec::new(),
            logit_bias: logit_bias.clone(),
            pending_drafts: Vec::new(),
            pending_draft_conf: Vec::new(),
            inside_thinking: req_enable_thinking && think_end_token.is_some(),
            enable_thinking: req_enable_thinking,
            thinking_budget: req_thinking_budget,
            repetition_detection: req_repetition_detection,
            spontaneous_think_budget,
            thinking_tokens: 0,
            cached_prompt_tokens: cached_prompt_tok,
            preempt_immune_until_tokens: 0,
            force_end_thinking: false,
            think_force_closed: false,
            sentence_defer_count: 0,
            consecutive_confident: 0,
            in_code_fence: false,
            think_end_token,
            think_start_token,
            think_ended: !req_enable_thinking && think_end_token.is_some(),
            think_just_ended: false,
            post_think_emitted: 0,
            spec_adapt: Default::default(),
            think_skip_count: 0,
            require_tool_call: use_legacy_tool_call,
            tool_request,
            tools_present: req_tools_present,
            tool_call_start_token,
            tool_call_opened: false,
            inside_tool_body: false,
            tool_call_completed: false,
            post_completion_tool_opens: 0,
            tool_body_streak_tokens: 0,
            inside_parameter_body: false,
            param_body_chars_emitted: 0,
            suppress_tool_call: req_suppress_tool_call,
            disable_mtp: req_disable_mtp,
            mtp_acct: Default::default(),
            content_started: false,
            content_tokens: 0,
            prose_tokens_since_last_tool: 0,
            think_watchdog_fires: 0,
            loop_steers: 0,
            loop_steer_max: crate::scheduler::loop_steer::budget_for_new_seq(),
            rollback_count: 0,
            ssm_rollback_ring: SsmDecodeRing::new(model.decode_rollback_ring_slots()),
            tool_call_end_token,
            grammar_state,
            last_token_time: now,
            request_start,
            decode_start: now,
            seed: req_seed,
            top_logprobs: req_top_logprobs,
            logprobs_data: Vec::new(),
            timeout_at: req_timeout_at,
            adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
        };
        finish_sequence(model, &mut a, sched.limits.max_seq_len);
        return Ok(StartPrefillResult::Finished);
    }

    // Deferred co-dispatch: setup + EP broadcast, then return InProgress at
    // chunk 0 WITHOUT prefilling — the batched step packs >=2 streams into one
    // forward. Vision is excluded upstream, so no chunk-0 embedding injection.
    if defer {
        debug_assert!(
            image_pixels.is_empty(),
            "vision must be excluded from co-dispatch"
        );
        // ATLAS_QWEN4EXP_PREFILL_MULTI: the multi pass sends the worker this
        // prompt (`super::prefill_multi`), so no prefill command here.
        let multi = super::prefill_multi::eligible(
            model,
            prompt_tokens.len(),
            is_last,
            req_prompt_logprobs.is_some(),
            req_adapter_slot,
        );
        if let Err(e) = (|| -> Result<()> {
            // EP: broadcast chunk 0 to worker (no-op on single-GPU; the batched
            // step does NOT re-broadcast, so this stays the only broadcast site).
            model.ep_broadcast_disable_mtp_for_seq(seq.slot_idx as u32, req_disable_mtp)?;
            model.ep_broadcast_vision_state_for_seq(seq.slot_idx as u32, false, 0, 0, 0, 0)?;
            plant_chunk0(model, &mut seq, pc_plant)?;
            if multi {
                return Ok(());
            }
            model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
            model.ep_broadcast_cmd(chunk_len as u32)?;
            model.ep_broadcast_cmd(0)?; // chunk_start
            model.ep_broadcast_cmd(prompt_tokens.len() as u32)?; // full prompt length
            model.ep_broadcast_tokens(&prompt_tokens)?;
            Ok(())
        })() {
            let msg = format!("deferred prefill EP broadcast failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            if let Err(fe) = model.free_sequence(&mut seq) {
                tracing::error!("prefill_a_step: free_sequence (deferred broadcast error): {fe:#}");
            }
            return Err(e);
        }
        // chunk_offset = 0 (deferred co-dispatch: nothing prefilled yet).
        let mut p = super::prefill_a_step_params::build_prefill_in_progress(
            prompt_tokens,
            req_session_hash,
            seq,
            0,
            max_tokens,
            req_min_tokens,
            eos_tokens.to_vec(),
            sink,
            cancel_flag,
            request_start,
            temperature,
            top_k,
            top_p,
            top_n_sigma,
            min_p,
            repetition_penalty,
            presence_penalty,
            frequency_penalty,
            req_lz_penalty,
            dry_multiplier,
            dry_base,
            dry_allowed_length,
            logit_bias,
            req_enable_thinking,
            req_thinking_budget,
            req_repetition_detection,
            spontaneous_think_budget,
            req_require_tool_call,
            req_tools_present,
            req_suppress_tool_call,
            req_disable_mtp,
            grammar_state,
            req_seed,
            req_top_logprobs,
            req_timeout_at,
        );
        p.multi = multi;
        return Ok(StartPrefillResult::InProgress(p));
    }

    // Guard: free SSM slot on any error after allocation.
    let prefill_result = (|| -> Result<DevicePtr> {
        // Vision: encode images and store embeddings for chunk 0 token overwrite.
        // Skipped when the images were already batch-encoded by the co-dispatch
        // pre-pass (vision_slice.is_some()) — that path runs ONE encode + fence
        // for the whole tick; here we only set the per-stream slice base below.
        if vision_slice.is_none() && !image_pixels.is_empty() {
            model.prepare_vision_embed(&image_pixels)?;
            // prepare_vision_embed() runs the vision encoder asynchronously on
            // the default stream, writing this request's patch embeddings into
            // the encoder's buf_out. The chunk-0 embedding injection inside
            // prefill_chunk() runs on prefill_stream and reads buf_out. Without
            // ordering between the two streams, the injection reads buf_out
            // BEFORE this request's encode lands and overlays the PREVIOUS
            // request's image embeddings — lag-by-one cross-image contamination
            // (and torn reads / illegal access under interleaved load). Make
            // prefill_stream wait for the encode to complete before injecting.
            model.record_event(prefill_event, model.default_stream())?;
            model.stream_wait_event(prefill_stream, prefill_event)?;
        }

        dispatch_chunk0_vision(model, &seq, vision_slice, &image_pixels, req_disable_mtp)?;
        plant_chunk0(model, &mut seq, pc_plant)?;
        // EP: broadcast chunk 0 tokens to worker.
        // Send full prompt length + all tokens so worker can do
        // identical Marconi prefix-cache lookups (bug #33 fix).
        // Uses bulk broadcast (single NCCL op) instead of per-token broadcast
        // which caused NCCL timeouts on long prompts (6K+ tokens = 6K+ broadcasts).
        model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF0)?;
        model.ep_broadcast_cmd(chunk_len as u32)?;
        model.ep_broadcast_cmd(0)?; // chunk_start
        model.ep_broadcast_cmd(prompt_tokens.len() as u32)?; // full prompt length
        model.ep_broadcast_tokens(&prompt_tokens)?;

        let _pt0 = std::time::Instant::now();
        let chunk_res = model.prefill_chunk(
            &prompt_tokens,
            &mut seq,
            0,
            chunk_len,
            is_last,
            prefill_stream,
        );
        if std::env::var("ATLAS_VISION_TIMING").is_ok() {
            let _ = model.synchronize(prefill_stream);
            tracing::info!(
                "VIT_TIMING prefill_chunk {} tok (img={}): {:.1}ms",
                chunk_len,
                vision_slice.is_some() || !image_pixels.is_empty(),
                _pt0.elapsed().as_secs_f64() * 1000.0
            );
        }
        if vision_slice.is_some() {
            model.set_vision_slice_base(0, 0, 0, 0);
        }
        let logits = chunk_res?;
        super::prefill_normalization::initial(model, &seq)?;
        Ok(logits)
    })();

    let logits = match prefill_result {
        Ok(l) => l,
        Err(e) => {
            let msg = format!("prefill_chunk failed: {e:#}");
            send_error_to_sink(&mut sink, &msg);
            if let Err(free_err) = model.free_sequence(&mut seq) {
                tracing::error!(
                    "prefill_a_step: free_sequence (after prefill error): {free_err:#}"
                );
            }
            if let Err(bcast_err) = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1)
            {
                tracing::error!(
                    "prefill_a_step: ep_broadcast (after prefill error): {bcast_err:#}"
                );
            }
            return Err(e);
        }
    };

    // Sync prefill stream before sampling or returning to decode.
    // Record event on prefill stream, make default stream wait.
    if let Err(e) = model.record_event(prefill_event, prefill_stream) {
        tracing::error!("prefill_a_step: record_event(prefill_event): {e:#}");
    }
    if let Err(e) = model.stream_wait_event(model.default_stream(), prefill_event) {
        tracing::error!("prefill_a_step: stream_wait_event(default_stream, prefill_event): {e:#}");
    }

    // chunk_offset = chunk_len; DRY comes from the request (api.rs
    // `sampling_presets.tools.dry_*`), 0.0 leaves it inert.
    let mut p = super::prefill_a_step_params::build_prefill_in_progress(
        prompt_tokens,
        req_session_hash,
        seq,
        chunk_len,
        max_tokens,
        req_min_tokens,
        eos_tokens.to_vec(),
        sink,
        cancel_flag,
        request_start,
        temperature,
        top_k,
        top_p,
        top_n_sigma,
        min_p,
        repetition_penalty,
        presence_penalty,
        frequency_penalty,
        req_lz_penalty,
        dry_multiplier,
        dry_base,
        dry_allowed_length,
        logit_bias,
        req_enable_thinking,
        req_thinking_budget,
        req_repetition_detection,
        spontaneous_think_budget,
        req_require_tool_call,
        req_tools_present,
        req_suppress_tool_call,
        req_disable_mtp,
        grammar_state,
        req_seed,
        req_top_logprobs,
        req_timeout_at,
    );

    if is_last {
        // Single chunk covered the entire prompt — get first token.
        // #131: constrain the FIRST token with the grammar (and advance the
        // matcher). Mirrors prefill_b_step; no-op when no grammar is active.
        // P1-4 (2026-07-09): thread the resolved `min_p` (request +
        // MODEL.toml floor) — previously a hardcoded 0.0 inside the sampler.
        // Kill-switch: ATLAS_NO_MTP_MINP=1.
        let first = match sample_first_token(
            model,
            logits,
            temperature,
            top_k,
            top_p,
            min_p,
            &crate::scheduler::min_tokens_ban::first_token_suppress(eos_tokens, req_min_tokens),
            p.grammar_state.as_mut(),
            &sched.levers.sampling(),
        ) {
            Ok(t) => {
                tracing::info!("Prefill first token: {t}");
                t
            }
            Err(e) => {
                let msg = format!("sample_token failed: {e:#}");
                send_error_to_sink(&mut p.sink, &msg);
                if let Err(free_err) = model.free_sequence(&mut p.seq) {
                    tracing::error!(
                        "prefill_a_step: free_sequence (after sample error): {free_err:#}"
                    );
                }
                if let Err(bcast_err) =
                    model.ep_broadcast_cmd_for_seq(p.seq.slot_idx as u32, 0xFFFFFFF1)
                {
                    tracing::error!(
                        "prefill_a_step: ep_broadcast (after sample error): {bcast_err:#}"
                    );
                }
                return Err(e);
            }
        };

        let spontaneous_think = !req_enable_thinking && think_start_token == Some(first);
        // Legacy echo+logprobs: hand prompt logprobs to a streaming client
        // BEFORE any token event (blocking carries them via finish_sequence).
        if req_prompt_logprobs.is_some()
            && let ResponseSink::Streaming(ref tx) = p.sink
        {
            let lps: Vec<crate::api::TokenLogprobs> = p
                .seq
                .prompt_logprobs
                .drain(..)
                .map(|lp| crate::api::TokenLogprobs {
                    token_id: lp.token_id,
                    logprob: lp.logprob,
                    top: lp.top,
                })
                .collect();
            if !super::mod_helpers::bounded_stream_send(
                tx,
                StreamEvent::PromptLogprobs(lps),
                "prefill_a prompt-logprobs",
            ) {
                tracing::warn!("prefill_a_step: prompt-logprobs send failed");
            }
        }
        // max_tokens==0 is a scoring-only call: no generated token leaves
        // the server (the sampled `first` is discarded below too).
        if !spontaneous_think
            && max_tokens > 0
            && let ResponseSink::Streaming(ref tx) = p.sink
            && !super::mod_helpers::bounded_stream_send(
                tx,
                StreamEvent::Token(first),
                "prefill_a first token",
            )
        {
            tracing::warn!("prefill_a_step: first-token send failed (receiver dropped)");
        }

        // When grammar is active, disable legacy require_tool_call (grammar handles EOS).
        let use_legacy_tool_call =
            req_require_tool_call && p.grammar_state.is_none() && tool_call_start_token.is_some();
        let immediate_finish =
            !spontaneous_think && (eos_tokens.contains(&first) || max_tokens <= 1);
        let cached_prompt_tok = p.seq.cached_prefix_tokens as u32;
        // The promotion constructor builds the ActiveSeq, so a prompt that
        // fits one chunk decodes under the same min_tokens end-token ban as a
        // multi-chunk one.
        let mut a = super::phase_promote_prefills::build_active_seq_from_prefill(
            p,
            first,
            spontaneous_think,
            use_legacy_tool_call,
            cached_prompt_tok,
            immediate_finish,
            Instant::now(),
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            model.decode_rollback_ring_slots(),
            sched.limits.glm_tool_boundary,
        );
        if immediate_finish {
            finish_sequence(model, &mut a, sched.limits.max_seq_len);
            Ok(StartPrefillResult::Finished)
        } else {
            Ok(StartPrefillResult::Active(a))
        }
    } else {
        Ok(StartPrefillResult::InProgress(p))
    }
}

/// Plant the requested shared-prefix checkpoint (`shared_prefix`) in a
/// prefill about to send its chunk-0 command.
fn plant_chunk0(model: &dyn Model, seq: &mut SequenceState, at: Option<usize>) -> Result<()> {
    at.map_or(Ok(()), |at| model.pc_plant(seq, at))
}
