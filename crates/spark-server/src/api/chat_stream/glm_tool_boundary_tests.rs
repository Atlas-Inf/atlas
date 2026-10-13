// SPDX-License-Identifier: AGPL-3.0-only

//! Real blocking decoder/choice and streaming-token paths, with a tiny real
//! tokenizer. No scheduler or model inference is replaced by these API tests.
use super::{ctx::StreamCtx, handle_token::handle_token, state::StreamState};
use crate::{AppState, ir, tool_parser};
use serde_json::json;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::time::Duration;

const END: u32 = 2;
const TOOL: u32 = 3;
const CALL: &[u32] = &[3, 4, 5, 6, 7, 8, 9, 10, 11];

#[path = "reasoning_done_tests.rs"]
mod reasoning_done_tests;

fn app(glm: bool) -> Arc<AppState> {
    let dir = tempfile::tempdir().unwrap();
    let words = [
        "[UNK]",
        "plan",
        "</think>",
        "<tool_call>",
        "get_weather",
        "<arg_key>",
        "city",
        "</arg_key>",
        "<arg_value>",
        "Oslo",
        "</arg_value>",
        "</tool_call>",
        "unknown_tool",
        "<think>",
        "42",
    ];
    let vocab = words
        .iter()
        .enumerate()
        .map(|(i, w)| (w.to_string(), i as u32))
        .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(model);
    inner.with_decoder(Some(tokenizers::decoders::fuse::Fuse::new()));
    inner
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        true,
        "glm5_next",
        "",
        None,
        false,
    )
    .unwrap();
    assert_eq!(
        tokenizer.decode(CALL).unwrap(),
        "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>"
    );
    let (request_tx, _) = tokio::sync::mpsc::channel(1);
    Arc::new(AppState {
        tokenizer,
        model_name: "display-name-is-not-model-type".into(),
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
        tool_call_parser: Some(Arc::from(
            tool_parser::ToolCallFormat::PoolsideV1.into_parser(),
        )),
        chat: Default::default(),
        reasoning_parser: Some(crate::reasoning_parser::ReasoningFormat::Qwen.into_parser()),
        think_end_token_id: Some(END),
        think_start_token_id: Some(13),
        tool_max_tokens: 192,
        sampling_presets: Default::default(),
        tool_call_start_token_id: Some(TOOL),
        glm_tool_boundary: crate::glm_tool_boundary::native_opener(
            if glm { "glm5_next" } else { "qwen3" },
            true,
            Some(TOOL),
        ),
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
    })
}

fn request() -> ir::ChatRequest {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model":"fixture", "messages":[{"role":"user","content":"weather"}],
        "tools":[{"type":"function","function":{"name":"get_weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                "required":["city"]}}}], "max_tokens":192
    }))
    .unwrap();
    wire.into()
}

fn response(tokens: Vec<u32>) -> crate::api::inference_types::InferenceResponse {
    crate::api::inference_types::InferenceResponse {
        output_tokens: tokens,
        finish_reason: "length".into(),
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        logprobs: vec![],
        reasoning_tokens: 1,
        cached_prompt_tokens: 0,
        accepted_prediction_tokens: 0,
        prompt_logprobs: vec![],
    }
}

fn tokens(explicit_close: bool, valid_name: bool) -> Vec<u32> {
    let mut tokens = vec![1];
    if explicit_close {
        tokens.push(END);
    }
    tokens.extend_from_slice(CALL);
    if !valid_name {
        let name = tokens.iter_mut().find(|id| **id == 4).unwrap();
        *name = 12;
    }
    tokens
}

#[test]
fn glm_native_tool_boundary_blocking_preserves_validated_call() {
    for explicit_close in [true, false] {
        let state = app(true);
        let req = request();
        let response = response(tokens(explicit_close, true));
        let (reasoning, content) =
            crate::api::chat_blocking::decode_response_text(&state, &response, true, true);
        assert_eq!(
            reasoning.as_deref().map(str::trim),
            Some("plan"),
            "explicit={explicit_close}"
        );
        assert!(
            content.starts_with("<tool_call>"),
            "opener must stay in content: {content:?}"
        );
        let choice = crate::api::chat_blocking_choice::build_choice_message(
            &state, &req, &response, reasoning, content, true, None, 0,
        );
        assert_eq!(
            choice.tool_calls.len(),
            1,
            "explicit={explicit_close}; {choice:?}"
        );
        assert_eq!(choice.tool_calls[0].name, "get_weather");
        assert_eq!(choice.tool_calls[0].arguments["city"], "Oslo");
        assert_eq!(choice.finish_reason, ir::FinishReason::ToolCalls);
    }
}

fn stream(
    glm: bool,
    tools: bool,
    explicit_close: bool,
    valid_name: bool,
) -> (StreamState, Vec<ir::StreamDelta>) {
    stream_tokens(glm, tools, tokens(explicit_close, valid_name))
}

fn stream_tokens(glm: bool, tools: bool, tokens: Vec<u32>) -> (StreamState, Vec<ir::StreamDelta>) {
    let (_, stream, deltas) = stream_tokens_with_context(glm, tools, tokens);
    (stream, deltas)
}

fn stream_tokens_with_context(
    glm: bool,
    tools: bool,
    tokens: Vec<u32>,
) -> (StreamCtx, StreamState, Vec<ir::StreamDelta>) {
    let state = app(glm);
    let defs = if tools { request().tools } else { vec![] };
    let ctx = StreamCtx {
        _active_guard: crate::metrics::ActiveRequestGuard::new(),
        state: state.clone(),
        model: "fixture".into(),
        id: "fixture".into(),
        prompt_len: 1,
        enable_thinking: true,
        tool_defs_for_backfill: defs.clone(),
        cwd_for_normalize: None,
        stop_strings: vec![],
        tool_retry_enabled: false,
        prompt_tokens: Arc::new(vec![13]),
        prompt_vocab: Arc::default(),
        grammar_spec: None,
        max_tokens: 192,
        timeout_at: None,
        stop_string_buffer_len: 0,
        leak_markers: state.tool_call_parser.as_ref().unwrap().leak_markers(),
        wants_typed_arguments: true,
        max_tool_calls_per_response: 12,
        min_tokens: 0,
        req_return_token_ids: true,
        req_ctx: None,
        dump_seq: None,
    };
    let cancel = Arc::new(AtomicBool::new(false));
    let mut stream = StreamState::new(tools, true, cancel, defs);
    if let Some(detector) = stream.detector.as_mut() {
        detector.set_promote_bare_names(true);
    }
    let mut deltas = vec![];
    for token in tokens {
        deltas.extend(handle_token(&mut stream, &ctx, token));
    }
    (ctx, stream, deltas)
}

#[test]
fn glm_native_tool_boundary_streaming_preserves_reasoning_then_tool() {
    for explicit_close in [true, false] {
        let (state, deltas) = stream(true, true, explicit_close, true);
        assert!(state.thinking_done, "explicit={explicit_close}; {deltas:?}");
        assert!(!state.cancel_flag.load(Ordering::Acquire));
        let calls: Vec<_> = deltas
            .iter()
            .filter_map(|d| match d {
                ir::StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(calls, ["get_weather"], "{deltas:?}");
        let reasoning: String = deltas
            .iter()
            .filter_map(|d| match d {
                ir::StreamDelta::Reasoning { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(reasoning.trim(), "plan");
        let mut ids: Vec<u32> = deltas
            .iter()
            .flat_map(|delta| match delta {
                ir::StreamDelta::Reasoning { token_ids, .. }
                | ir::StreamDelta::Content { token_ids, .. }
                | ir::StreamDelta::ToolCallArgs { token_ids, .. } => token_ids.as_slice(),
                _ => &[],
            })
            .copied()
            .collect();
        ids.extend_from_slice(&state.pending_token_ids);
        assert_eq!(ids, tokens(explicit_close, true), "one ID per actual token");
        assert!(!deltas.iter().any(
            |d| matches!(d, ir::StreamDelta::Content { text, .. } if text.contains("tool_call"))
        ));
    }
}

#[test]
fn glm_native_tool_boundary_first_token_and_malformed_envelope() {
    let state = app(true);
    let req = request();
    for complete in [true, false] {
        let raw = if complete {
            CALL
        } else {
            &CALL[..CALL.len() - 2]
        };
        let response = response(raw.to_vec());
        let (reasoning, content) =
            crate::api::chat_blocking::decode_response_text(&state, &response, true, true);
        assert!(reasoning.is_none());
        let choice = crate::api::chat_blocking_choice::build_choice_message(
            &state, &req, &response, reasoning, content, true, None, 0,
        );
        assert_eq!(choice.tool_calls.len(), usize::from(complete));
        let (stream, deltas) = stream_tokens(true, true, raw.to_vec());
        assert!(stream.thinking_done);
        assert!(!stream.cancel_flag.load(Ordering::Acquire));
        assert_eq!(
            deltas
                .iter()
                .filter(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
                .count(),
            usize::from(complete)
        );
        assert!(
            !deltas
                .iter()
                .any(|d| matches!(d, ir::StreamDelta::Reasoning { .. }))
        );
    }
}

#[test]
fn glm_native_tool_boundary_keeps_request_model_and_name_guards() {
    for (glm, tools) in [(false, true), (true, false)] {
        let state = app(glm);
        let response = response(tokens(false, true));
        let (_, content) =
            crate::api::chat_blocking::decode_response_text(&state, &response, true, tools);
        assert!(content.is_empty());
        let (stream, _) = stream(glm, tools, false, true);
        assert!(!stream.thinking_done);
    }
    // Opening a native tool phase does not authorize an unregistered name.
    let state = app(true);
    let req = request();
    let response = response(tokens(false, false));
    let (reasoning, content) =
        crate::api::chat_blocking::decode_response_text(&state, &response, true, true);
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &state, &req, &response, reasoning, content, true, None, 0,
    );
    assert!(choice.tool_calls.is_empty());
    let (_, deltas) = stream(true, true, false, false);
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
    );
    // Exact-name admission is GLM-scoped, including conventional post-think
    // calls. Other models retain the existing blocking fuzzy-name behavior.
    for glm in [false, true] {
        let state = app(glm);
        let response = self::response(tokens(true, false));
        let (reasoning, content) =
            crate::api::chat_blocking::decode_response_text(&state, &response, true, true);
        let choice = crate::api::chat_blocking_choice::build_choice_message(
            &state, &req, &response, reasoning, content, true, None, 0,
        );
        assert_eq!(choice.tool_calls.len(), usize::from(!glm));
    }
}
