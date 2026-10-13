// SPDX-License-Identifier: AGPL-3.0-only

//! Blocking-mode refusal classification must PRESERVE the model's full
//! assistant text. Streaming already emits the text AND a `Refusal`
//! metadata delta (`chat_stream/done_finalization.rs`); the blocking path
//! used to null `content` after setting `refusal`, silently dropping every
//! sentence after the first. These tests drive the real production
//! assembly (`chat_blocking_choice::build_choice_message`) and the real
//! OpenAI wire projection (`openai::encode_chat_response`) — no logic is
//! copied, so a regression in either layer is caught here.

use crate::AppState;
use crate::ir;
use serde_json::json;
use std::sync::Arc;
use std::time::Duration;

/// A tiny real word-level tokenizer. This suite never decodes logprobs, so
/// its vocabulary only has to be valid for `ChatTokenizer::from_model_dir`.
fn tokenizer(dir: &std::path::Path) -> crate::tokenizer::ChatTokenizer {
    let vocab = ["[UNK]", "a", "</think>"]
        .iter()
        .enumerate()
        .map(|(i, w)| ((*w).to_string(), i as u32))
        .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(model);
    inner.with_decoder(Some(tokenizers::decoders::fuse::Fuse::new()));
    inner.save(dir.join("tokenizer.json"), false).unwrap();
    crate::tokenizer::ChatTokenizer::from_model_dir(dir, 0, true, "glm5_next", "", None, false)
        .unwrap()
}

/// A minimal `AppState` — only the fields `build_choice_message` and the
/// OpenAI encoder actually read are set to non-default values; the rest are
/// the same neutral defaults the production server builds for a text-only,
/// tool-free GLM serve.
fn state(dir: &std::path::Path) -> AppState {
    let (request_tx, _rx) = tokio::sync::mpsc::channel(1);
    AppState {
        tokenizer: tokenizer(dir),
        model_name: "refusal-fixture".into(),
        adapter_name: None,
        adapter_names: vec![],
        active_adapter: Arc::default(),
        max_seq_len: 4096,
        request_tx,
        rotation_tx: None,
        vision_config: None,
        vision_max_pixels: None,
        remote_image_policy: Default::default(),
        video_ffmpeg: Default::default(),
        video_fps: 1.0,
        default_temperature: 0.0,
        default_top_k: 0,
        default_top_p: 1.0,
        default_top_n_sigma: 0.0,
        default_min_p: 0.0,
        tool_call_parser: None,
        chat: Default::default(),
        reasoning_parser: Some(crate::reasoning_parser::ReasoningFormat::Qwen.into_parser()),
        think_end_token_id: Some(2),
        think_start_token_id: Some(3),
        tool_max_tokens: 192,
        sampling_presets: Default::default(),
        tool_call_start_token_id: None,
        glm_tool_boundary: None,
        auto_compact_threshold: None,
        request_timeout: 0,
        effective_context: 0,
        behavior: Default::default(),
        disable_thinking: false,
        default_thinking: Default::default(),
        default_reasoning_effort: None,
        response_store: crate::response_store::ResponseStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        conversation_store: crate::conversation_store::ConversationStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        rate_limiter: crate::rate_limiter::RateLimiter::with_config(
            crate::rate_limiter::RateLimitConfig {
                rpm: 0,
                tpm: 0,
                burst_rpm: 1,
                burst_tpm: 1,
            },
        ),
        dump_writer: None,
        lora_stageable: Default::default(),
        lora_peer_addr: None,
        promotion: None,
        promoted_slots: Arc::default(),
        lora_disk_stageable: Default::default(),
    }
}

/// A response carrying `finish_reason` only — this suite never decodes
/// tokens, it feeds `output_text` straight into `build_choice_message`.
fn response(finish_reason: &str) -> crate::api::inference_types::InferenceResponse {
    crate::api::inference_types::InferenceResponse {
        output_tokens: vec![],
        finish_reason: finish_reason.into(),
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        logprobs: vec![],
        reasoning_tokens: 0,
        cached_prompt_tokens: 0,
        accepted_prediction_tokens: 0,
        prompt_logprobs: vec![],
    }
}

/// A tool-free, single-turn chat request.
fn request() -> ir::ChatRequest {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model": "fixture",
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 192
    }))
    .unwrap();
    wire.into()
}

/// Project one piece of pre-assembled model text through the exact
/// production blocking assembly, then through the OpenAI wire encoder, and
/// return the serialized choice JSON object.
fn project(
    output_text: String,
    reasoning: Option<String>,
    tools_active: bool,
    finish_reason: &str,
) -> serde_json::Value {
    let dir = tempfile::tempdir().unwrap();
    let state = state(dir.path());
    let req = request();
    let response = response(finish_reason);
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &state,
        &req,
        &response,
        reasoning,
        output_text,
        tools_active,
        None,
        0,
    );
    let ir_response = ir::ChatResponse {
        id: "refusal-fixture".into(),
        model: "fixture".into(),
        created: 0,
        choices: vec![choice],
        usage: ir::Usage::default(),
    };
    let echo = crate::api::ResponseEcho {
        service_tier: None,
        metadata: None,
        store: false,
        include_usage: false,
    };
    let resp = crate::openai::encode_chat_response(&state, ir_response, &echo, None);
    let bytes = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(axum::body::to_bytes(resp.into_body(), usize::MAX))
        .unwrap();
    let mut body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    body["choices"][0].take()
}

/// Multi-sentence refusal: the message must keep EVERY sentence, not just
/// the first, while also carrying the classifier's `refusal` sentence.
#[test]
fn blocking_multi_sentence_refusal_keeps_full_content_and_refusal() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let text = "I cannot help with that request. Here is why: it would be \
                unsafe. Please ask something else.";
    let choice = project(text.into(), None, false, "stop");
    let message = &choice["message"];
    assert_eq!(
        message["content"].as_str(),
        Some(text),
        "full content must survive refusal classification: {message}"
    );
    assert_eq!(
        message["refusal"].as_str(),
        Some("I cannot help with that request."),
        "refusal metadata must still be set: {message}"
    );
}

/// A helpful answer can match a broad refusal prefix. Keep all its text.
#[test]
fn blocking_prefix_false_positive_keeps_full_answer() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let text = "I cannot see your screen. Please paste the error and I can help.";
    let choice = project(text.into(), None, false, "stop");
    assert_eq!(choice["message"]["content"].as_str(), Some(text));
    assert_eq!(choice["message"]["refusal"], "I cannot see your screen.");
}

/// Sympathy alone is not one of the classifier's refusal prefixes.
#[test]
fn blocking_sympathy_keeps_full_content_without_refusal() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let text = "I'm sorry for your loss. Take all the time you need.";
    let choice = project(text.into(), None, false, "stop");
    assert_eq!(choice["message"]["content"].as_str(), Some(text));
    assert!(choice["message"].get("refusal").is_none());
}

/// Ordinary (non-refusal) text must be untouched: content intact, no
/// spurious refusal.
#[test]
fn blocking_ordinary_text_has_content_and_no_refusal() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let text = "Sure — here is how to do that in three steps.";
    let choice = project(text.into(), None, false, "stop");
    let message = &choice["message"];
    assert_eq!(message["content"].as_str(), Some(text), "{message}");
    assert!(
        message.get("refusal").is_none(),
        "ordinary text must not gain a refusal: {message}"
    );
}

/// Empty text stays an empty string, without refusal metadata.
#[test]
fn blocking_empty_text_has_no_content_and_no_refusal() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let choice = project(String::new(), None, false, "stop");
    let message = &choice["message"];
    assert_eq!(message["content"].as_str(), Some(""), "{message}");
    assert!(message.get("refusal").is_none(), "{message}");
}

/// A refusal that also carries a reasoning trace: the reasoning survives
/// too, and does not suppress the content-preservation fix.
#[test]
fn blocking_refusal_preserves_reasoning_trace() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let text = "I must decline. This request falls outside what I can help \
                with safely.";
    let reasoning = "The user is asking for something disallowed.";
    let choice = project(text.into(), Some(reasoning.into()), false, "stop");
    let message = &choice["message"];
    assert_eq!(message["content"].as_str(), Some(text), "{message}");
    assert_eq!(
        message["reasoning_content"].as_str(),
        Some(reasoning),
        "{message}"
    );
    assert_eq!(
        message["refusal"].as_str(),
        Some("I must decline."),
        "{message}"
    );
}

/// A valid tool call suppresses refusal classification (the classifier is
/// gated on `tool_calls.is_none()`), and the finish reason stays
/// `tool_calls` — the preservation fix must not disturb that.
#[test]
fn blocking_valid_tool_call_keeps_call_and_finish_reason() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let dir = tempfile::tempdir().unwrap();
    let state = state(dir.path());
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model": "fixture",
        "messages": [{"role": "user", "content": "weather"}],
        "tools": [{"type": "function", "function": {"name": "get_weather",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                "required": ["city"]}}}],
        "max_tokens": 192
    }))
    .unwrap();
    let req: ir::ChatRequest = wire.into();
    let response = response("tool_calls");
    let call = "I cannot answer directly. <tool_call>get_weather<arg_key>city</arg_key>\
                <arg_value>Oslo</arg_value></tool_call>";
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &state,
        &req,
        &response,
        None,
        call.into(),
        true,
        None,
        0,
    );
    assert_eq!(choice.tool_calls.len(), 1, "{choice:?}");
    assert_eq!(choice.tool_calls[0].name, "get_weather");
    assert_eq!(choice.tool_calls[0].arguments["city"], "Oslo");
    assert_eq!(choice.finish_reason, ir::FinishReason::ToolCalls);
    assert!(choice.refusal.is_none(), "tool turn must not be a refusal");
}

/// Finish reasons other than a natural stop are preserved verbatim on a
/// refusal turn: a truncated ("length") or timed-out ("timeout") refusal
/// still keeps its text and its reason.
#[test]
fn blocking_refusal_preserves_length_and_timeout_finish_reasons() {
    let _guard = crate::refusal::TEST_ENV_LOCK
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for wire_reason in ["stop", "length", ir::FINISH_REASON_TIMEOUT] {
        let text = "I cannot provide that. Here is an alternative.";
        let choice = project(text.into(), None, false, wire_reason);
        assert_eq!(choice["message"]["content"].as_str(), Some(text));
        assert_eq!(choice["message"]["refusal"], "I cannot provide that.");
        assert_eq!(choice["finish_reason"], wire_reason);
    }
}
