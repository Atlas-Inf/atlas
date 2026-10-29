// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[allow(clippy::too_many_arguments)]
pub(crate) fn finalize_response(
    state: Arc<AppState>,
    req_ctx: Option<axum::extract::Extension<crate::rate_limiter::RequestContext>>,
    all_choices: Vec<ir::Choice>,
    total_completion_tokens: usize,
    first_ttft: f64,
    last_decode_time_ms: f64,
    total_reasoning_tokens: u32,
    total_cached_prompt_tokens: u32,
    total_accepted_prediction_tokens: usize,
    prompt_len: usize,
) -> crate::api::chat::ChatOutcome {
    let tokens_per_second = if last_decode_time_ms > 0.0 && total_completion_tokens > 0 {
        (total_completion_tokens.saturating_sub(1)) as f64 / (last_decode_time_ms / 1000.0)
    } else {
        0.0
    };
    let usage = ir::Usage {
        prompt_tokens: prompt_len,
        completion_tokens: total_completion_tokens,
        cached_prompt_tokens: total_cached_prompt_tokens as usize,
        reasoning_tokens: total_reasoning_tokens as usize,
        accepted_prediction_tokens: total_accepted_prediction_tokens,
        time_to_first_token_ms: first_ttft,
        response_tokens_per_second: tokens_per_second,
    };

    // REQUESTS_ACTIVE released by the caller's ActiveRequestGuard on return.
    crate::metrics::PROMPT_TOKENS_TOTAL.inc_by(prompt_len as u64);
    crate::metrics::GENERATION_TOKENS_TOTAL.inc_by(total_completion_tokens as u64);
    crate::metrics::TTFT_SECONDS
        .with_label_values(&[state.model_name.as_str()])
        .observe(first_ttft / 1000.0);

    // Rate-limit true-up. Middleware admitted with a conservative
    // reservation of `max_seq_len` tokens; refund the difference.
    if let Some(axum::extract::Extension(ref ctx)) = req_ctx {
        let actual = (prompt_len + total_completion_tokens) as u64;
        let refund = ctx.reserved_tokens.saturating_sub(actual);
        if refund > 0 {
            state.rate_limiter.refund_tokens(&ctx.identity, refund);
        }
    }

    crate::api::chat::ChatOutcome::Blocking(Box::new(ir::ChatResponse {
        id: crate::ids::uuid_v4(),
        model: state.model_name.clone(),
        created: crate::ids::unix_timestamp(),
        choices: all_choices,
        usage,
    }))
}

// OpenAI contract: a response ended by a client stop sequence is
// `finish_reason="stop"`, never `"length"`. The blocking path only
// detects multi-token stop strings post-hoc (the suffix strip in the
// caller), so this corrects exactly the "length" misreport — EOS
// already reports "stop", and "timeout"/"tool_calls" keep outranking a
// stop match (same precedence as the streaming resolver in
// `chat_stream::handle_done::resolve_wire_finish_reason`).
