// SPDX-License-Identifier: AGPL-3.0-only

//! Promote completed (fully prefilled) prefills into the active queue,
//! or surface error to the sink + free the sequence. Extracted from
//! `phase_continue_prefills` for the ≤500 LoC cap.

use spark_model::traits::Model;
use std::time::Instant;

use super::*;

#[cfg(test)]
#[path = "first_token_thinking_promotion_tests.rs"]
mod first_token_thinking_tests;

#[allow(clippy::too_many_arguments)]
pub(super) fn promote_completed_prefills(
    model: &dyn Model,
    prefilling: &mut Vec<PrefillInProgress>,
    mut completed_indices: Vec<(usize, Option<u32>)>,
    active: &mut Vec<ActiveSeq>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    // Served context ceiling (`sched.limits.max_seq_len`) — finish_sequence
    // needs it for the budget-derived `finish_reason` decision.
    max_seq_len: usize,
    glm_tool_boundary: Option<u32>,
) {
    // Process in reverse order so the remaining indices stay valid. `remove`,
    // not `swap_remove`: `prefilling` is the service order (the continue phase
    // advances its head), and a swap would move the newest prefill ahead of
    // every older one each time the head completes.
    completed_indices.sort_unstable_by_key(|x| std::cmp::Reverse(x.0));
    for (idx, maybe_token) in completed_indices {
        let mut p = prefilling.remove(idx);
        let Some(first) = maybe_token else {
            // Every producer uses None to mark a failed forward/sampling
            // operation.  Complete the request before releasing its owner;
            // otherwise the scheduler frees/reallocates the slot while the
            // client waits forever behind SSE keepalives.
            super::lifecycle::send_error_to_sink(&mut p.sink, "prefill failed");
            let mut seq = p.seq;
            if let Err(e) = model.free_sequence(&mut seq) {
                tracing::error!("phase_promote_prefills: free_sequence (error path): {e:#}");
            }
            if let Err(e) = model.ep_broadcast_cmd_for_seq(seq.slot_idx as u32, 0xFFFFFFF1) {
                tracing::error!(
                    "phase_promote_prefills: ep_broadcast free+realloc (error path): {e:#}"
                );
            }
            continue;
        };
        let spontaneous_think = !p.enable_thinking && think_start_token == Some(first);
        // Legacy echo+logprobs: prompt logprobs precede any token event.
        if p.seq.collect_prompt_logprobs.is_some()
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
                "promote prompt-logprobs",
            ) {
                tracing::warn!("phase_promote_prefills: prompt-logprobs send failed");
            }
        }
        // Only stream non-EOS tokens (OpenAI: stop seq not in output).
        if !spontaneous_think
            && p.max_tokens > 0
            && !p.eos_tokens.contains(&first)
            && let ResponseSink::Streaming(ref tx) = p.sink
            && !super::mod_helpers::bounded_stream_send(
                tx,
                StreamEvent::Token(first),
                "promote first token",
            )
        {
            tracing::warn!("phase_promote_prefills: first-token send failed (receiver dropped)");
        }
        let use_legacy_tool_call =
            p.require_tool_call && p.grammar_state.is_none() && tool_call_start_token.is_some();
        let now = Instant::now();
        let cached_prompt_tok = p.seq.cached_prefix_tokens as u32;
        let immediate_finish =
            !spontaneous_think && (p.eos_tokens.contains(&first) || p.max_tokens <= 1);

        let mut a = build_active_seq_from_prefill(
            p,
            first,
            spontaneous_think,
            use_legacy_tool_call,
            cached_prompt_tok,
            immediate_finish,
            now,
            think_end_token,
            think_start_token,
            tool_call_start_token,
            tool_call_end_token,
            model.decode_rollback_ring_slots(),
            glm_tool_boundary,
        );
        if immediate_finish {
            finish_sequence(model, &mut a, max_seq_len);
        } else {
            active.push(a);
        }
    }
}

/// Construct an `ActiveSeq` from a finished prefill plus its first sampled
/// token. The two original branches (immediate finish + continue) used
/// near-identical field initialisation; this helper reuses the same record
/// and the caller decides whether to push onto active or finish_sequence.
#[allow(clippy::too_many_arguments)]
pub(super) fn build_active_seq_from_prefill(
    p: PrefillInProgress,
    first: u32,
    spontaneous_think: bool,
    use_legacy_tool_call: bool,
    cached_prompt_tok: u32,
    immediate_finish: bool,
    now: Instant,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    // Phase-C decode-rollback ring capacity (`model.decode_rollback_ring_slots()`).
    ssm_ring_capacity: usize,
    glm_tool_boundary: Option<u32>,
) -> ActiveSeq {
    let temperature = p.temperature;
    // F4: sticky tool-request flag — grammar attached OR legacy tool path.
    // Computed before `p.grammar_state` is moved into the struct below.
    let tool_request = tool_request_for(p.grammar_state.as_ref(), use_legacy_tool_call);
    let native_tool_open = p.max_tokens > 0 && p.tools_present && glm_tool_boundary == Some(first);
    let thinking = first_token_thinking::FirstTokenThinking::resolve_with_tool_boundary(
        p.enable_thinking,
        first,
        think_start_token,
        think_end_token,
        glm_tool_boundary.filter(|_| p.tools_present),
    );
    let mut seq = p.seq;
    // The min_tokens end-token ban, for every chunked prompt (one chunk or more).
    spark_model::traits::EosBan::new(seq.prompt_len, p.min_tokens, &p.eos_tokens).arm(&mut seq);
    ActiveSeq {
        seq,
        session_hash: p.session_hash,
        last_token: first,
        output_tokens: if (!immediate_finish && spontaneous_think) || p.max_tokens == 0 {
            vec![]
        } else {
            vec![first]
        },
        remaining: if immediate_finish {
            0
        } else {
            p.max_tokens - 1
        },
        min_tokens: p.min_tokens,
        eos_tokens: p.eos_tokens,
        finished: immediate_finish,
        engine_error: None,
        guard_stop: None,
        param_close_pending: 0,
        sink: p.sink,
        cancel_flag: p.cancel_flag,
        temperature: p.temperature,
        top_k: p.top_k,
        top_p: p.top_p,
        top_n_sigma: p.top_n_sigma,
        min_p: p.min_p,
        repetition_penalty: p.repetition_penalty,
        presence_penalty: p.presence_penalty,
        frequency_penalty: p.frequency_penalty,
        repetition_penalty_window: 256,
        lz_penalty: p.lz_penalty,
        dry_multiplier: p.dry_multiplier,
        dry_base: p.dry_base,
        dry_allowed_length: p.dry_allowed_length,
        dry_sequence_breakers: Vec::new(),
        logit_bias: p.logit_bias,
        pending_drafts: Vec::new(),
        pending_draft_conf: Vec::new(),
        inside_thinking: thinking.inside_thinking,
        enable_thinking: p.enable_thinking,
        thinking_budget: if !immediate_finish && spontaneous_think {
            Some(p.spontaneous_think_budget)
        } else {
            p.thinking_budget
        },
        repetition_detection: p.repetition_detection,
        spontaneous_think_budget: p.spontaneous_think_budget,
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
        think_ended: thinking.think_ended,
        think_just_ended: thinking.think_just_ended,
        post_think_emitted: 0,
        spec_adapt: Default::default(),
        think_skip_count: 0,
        require_tool_call: use_legacy_tool_call && !native_tool_open,
        tool_request,
        tools_present: p.tools_present,
        tool_call_start_token,
        tool_call_opened: native_tool_open,
        inside_tool_body: native_tool_open,
        tool_call_completed: false,
        post_completion_tool_opens: 0,
        tool_body_streak_tokens: 0,
        inside_parameter_body: false,
        param_body_chars_emitted: 0,
        suppress_tool_call: p.suppress_tool_call,
        disable_mtp: p.disable_mtp,
        mtp_acct: Default::default(),
        content_started: false,
        content_tokens: 0,
        prose_tokens_since_last_tool: 0,
        think_watchdog_fires: 0,
        rollback_count: 0,
        ssm_rollback_ring: SsmDecodeRing::new(ssm_ring_capacity),
        tool_call_end_token,
        grammar_state: p.grammar_state,
        last_token_time: now,
        request_start: p.request_start,
        decode_start: now,
        seed: p.seed,
        top_logprobs: p.top_logprobs,
        logprobs_data: Vec::new(),
        timeout_at: p.timeout_at,
        adaptive: crate::adaptive_sampler::AdaptiveSamplingState::new(temperature),
    }
}

/// A 4-token prefill in progress with `max_tokens` as its only distinguishing
/// field, and the receiver its request completes on.
#[cfg(test)]
fn test_prefill(max_tokens: usize) -> (PrefillInProgress, super::test_support::RespRx) {
    let (a, response_rx) = super::test_support::test_seq(vec![], max_tokens, None, 4);
    let now = Instant::now();
    let p = super::prefill_a_step_params::build_prefill_in_progress(
        std::sync::Arc::new(vec![7; 4]),
        0,
        a.seq,
        4,
        max_tokens,
        0,
        super::test_support::EOS.to_vec(),
        a.sink,
        None,
        now,
        0.0,
        0,
        1.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0,
        vec![],
        false,
        None,
        None,
        0,
        false,
        false,
        false,
        false,
        None,
        None,
        None,
        None,
    );
    (p, response_rx)
}

#[cfg(test)]
#[test]
fn failed_prefill_notifies_request_before_releasing_owner() {
    let (p, mut response_rx) = test_prefill(8);
    let mut prefilling = vec![p];
    let mut active = Vec::new();

    promote_completed_prefills(
        &super::lifecycle_tests::StubModel::default(),
        &mut prefilling,
        vec![(0, None)],
        &mut active,
        None,
        None,
        None,
        None,
        64,
        None,
    );

    let delivered = response_rx
        .try_recv()
        .unwrap_or_else(|e| panic!("failed prefill did not terminate request: {e:?}"));
    match delivered {
        Ok(_) => panic!("failed prefill unexpectedly returned a successful response"),
        Err(e) => assert!(e.to_string().contains("prefill failed"), "{e:#}"),
    }
    assert!(prefilling.is_empty());
    assert!(active.is_empty());
}

/// The continue phase advances `prefilling[0]`, so the order is the service
/// order: a finished head must not let the newest prefill overtake older ones.
#[cfg(test)]
#[test]
fn finished_prefill_keeps_the_others_in_arrival_order() {
    let (mut prefilling, _receivers): (Vec<_>, Vec<_>) =
        [8, 9, 10].map(test_prefill).into_iter().unzip();
    let mut active = Vec::new();

    promote_completed_prefills(
        &super::lifecycle_tests::StubModel::default(),
        &mut prefilling,
        vec![(0, None)],
        &mut active,
        None,
        None,
        None,
        None,
        64,
        None,
    );

    let order: Vec<usize> = prefilling.iter().map(|p| p.max_tokens).collect();
    assert_eq!(order, [9, 10]);
}
