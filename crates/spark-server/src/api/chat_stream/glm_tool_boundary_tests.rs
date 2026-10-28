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
const BACKTICK: u32 = 15;
const ORDINARY: u32 = 16;
const LITERAL_CONTENT: [&str; 4] = [
    "The tag is `<tool_call>` and this is prose. END.",
    "The tag is \"<tool_call>\" and this is prose. END.",
    "The delimiter <tool_call> opens a tool invocation. END.",
    "```xml\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>\n```\nEND",
];

#[path = "reasoning_done_tests.rs"]
mod reasoning_done_tests;

#[path = "blocking_vision_choices_tests.rs"]
mod blocking_vision_choices_tests;

fn app(glm: bool) -> Arc<AppState> {
    app_for_model(
        if glm { "glm5_next" } else { "qwen4_exp" },
        tool_parser::ToolCallFormat::PoolsideV1,
    )
}

fn app_for_model(model_type: &str, format: tool_parser::ToolCallFormat) -> Arc<AppState> {
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
        "`",
        "ordinary",
        LITERAL_CONTENT[0],
        LITERAL_CONTENT[1],
        LITERAL_CONTENT[2],
        LITERAL_CONTENT[3],
        "The closing tag is `</tool_call>`. </tool_call> END.",
        "<|im_start|>",
        "The tag is `",
        "</",
        "tool",
        "_call>",
        "` and this is prose. END.",
        "The delimiter ",
        " opens a tool invocation. END.",
        "<",
        "<function=get_weather><parameter=city>Oslo</parameter></function>",
        "`<tool_call><function=get_weather><parameter=not_a_parameter>BAD</parameter></function></tool_call>` END.",
        "<tool_call><function=not_registered><parameter=value>42</parameter></function></tool_call> END.",
        "<tool_call><function=not_registered><parameter=value>42",
        "get_weather<arg_key>not_a_parameter</arg_key><arg_value>BAD</arg_value></tool_call> END.",
        "get_weather<arg_key>city</arg_key><arg_value>http://169.254.169.254/</arg_value></tool_call>",
        "<function=get_weather><parameter=city>http://169.254.169.254/</parameter></function></tool_call>",
        "<function=get_url><parameter=url>http://169.254.169.254/</parameter></function></tool_call>",
        "get_url<arg_key>url</arg_key><arg_value>http://169.254.169.254/</arg_value></tool_call>",
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
        model_type,
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
        tool_guard: Default::default(),
        vision_budget: None,
        vision_limits: None,
        tokenizer,
        model_name: "display-name-is-not-model-type".into(),
        adapter_name: None,
        adapter_names: vec![],
        active_adapter: Arc::default(),
        max_seq_len: 4096,
        request_tx,
        reference_mode: false,
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
        tool_call_parser: Some(Arc::from(format.into_parser())),
        chat: Default::default(),
        reasoning_parser: Some(crate::reasoning_parser::ReasoningFormat::Qwen.into_parser()),
        think_end_token_id: Some(END),
        think_start_token_id: Some(13),
        tool_max_tokens: 192,
        sampling_presets: Default::default(),
        tool_call_start_token_id: Some(TOOL),
        glm_tool_boundary: crate::glm_tool_boundary::native_opener(model_type, true, Some(TOOL)),
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
        guard_stop: None,
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
            crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
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
    stream_tokens_with_state(app(glm), tools, tokens, request())
}

/// Same runner with a caller-built `AppState` and `ChatRequest` (the
/// tool-guard tests pick the policy and declare their own tools). Ported
/// from Enntity/atlas fa5331b2 (AGPL-3.0-only).
/// Enntity's test-entry name over our request-passing runner (the literal-XML
/// tests use the shared `request()` tools).
fn stream_for_app(
    state: Arc<AppState>,
    tools: bool,
    tokens: Vec<u32>,
) -> (StreamCtx, StreamState, Vec<ir::StreamDelta>) {
    stream_tokens_with_state(state, tools, tokens, request())
}

fn stream_tokens_with_state(
    state: Arc<AppState>,
    tools: bool,
    tokens: Vec<u32>,
    request: ir::ChatRequest,
) -> (StreamCtx, StreamState, Vec<ir::StreamDelta>) {
    let defs = if tools { request.tools } else { vec![] };
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
        leak_markers: state
            .tool_call_parser
            .as_ref()
            .unwrap()
            .leak_markers()
            .for_literal_xml(state.tokenizer.glm_tool_markup_as_text),
        role_literal_leaks: false,
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
        detector.set_promote_bare_names(
            state
                .tool_call_parser
                .as_ref()
                .is_some_and(|p| p.name() == "poolside_v1"),
        );
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
            crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
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
            crate::api::chat_blocking::decode_response_text(&state, &response, true, tools, false);
        assert!(content.is_empty());
        let (stream, _) = stream(glm, tools, false, true);
        assert!(!stream.thinking_done);
    }
    // Opening a native tool phase does not authorize an unregistered name.
    let state = app(true);
    let req = request();
    let response = response(tokens(false, false));
    let (reasoning, content) =
        crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
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
            crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
        let choice = crate::api::chat_blocking_choice::build_choice_message(
            &state, &req, &response, reasoning, content, true, None, 0,
        );
        assert_eq!(choice.tool_calls.len(), usize::from(!glm));
    }
}

#[test]
fn poolside_holds_headers_until_complete_on_every_model() {
    for glm in [false, true] {
        let mut partial = vec![END];
        partial.extend_from_slice(&CALL[..CALL.len() - 2]);
        let (state, deltas) = stream_tokens(glm, true, partial);
        assert!(state.thinking_done);
        assert_eq!(
            deltas
                .iter()
                .filter(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
                .count(),
            0,
            "Poolside headers must wait for a complete validated call"
        );
    }
    // Explicitly closing reasoning must not bypass GLM name admission either.
    let (_, deltas) = stream(true, true, true, false);
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
    );
}

#[test]
fn non_poolside_headers_still_stream_immediately() {
    // Non-guard streaming path: release the header immediately (the guard
    // would hold it until call end; the fixture enables it via Policy::default).
    let (mut ctx, mut state, _) = stream_tokens_with_context(false, true, vec![END]);
    Arc::get_mut(&mut ctx.state).unwrap().tool_guard.enabled = false;
    state
        .detector
        .as_mut()
        .unwrap()
        .set_promote_bare_names(false);
    let mut deltas = Vec::new();
    super::tool_handlers::handle_tool_call_start(
        &mut state,
        &ctx,
        "call_example".into(),
        "get_weather".into(),
        0,
        &mut deltas,
    );
    assert!(matches!(
        deltas.as_slice(),
        [ir::StreamDelta::ToolCallStart { name, .. }] if name == "get_weather"
    ));
    assert!(state.buffered_tool_chunks.is_empty());
}

/// #172: a blocking reply whose text opens with a refusal prefix keeps its
/// full `content` (the streamed path always did). `refusal` is not asserted
/// strictly: `refusal::tests` flips the process-wide kill switch in parallel.
#[test]
fn blocking_refusal_prefix_keeps_content() {
    let state = app(false);
    let req = request();
    let response = response(vec![]);
    let text = "I cannot verify that source. Here is what the paper itself says: ...";
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &state,
        &req,
        &response,
        None,
        text.to_string(),
        false,
        None,
        0,
    );
    assert_eq!(choice.content.as_deref(), Some(text));
    if let Some(r) = choice.refusal.as_deref() {
        assert_eq!(r, "I cannot verify that source.");
    }
}


fn an_over_cap_image_is_refused_through_prepare_chat_prompt_with_vision_too_large() {
    // #236 regression: production callers used AdmitCtx::default(), so the
    // per-request row cap never fired. This drives the real path —
    // prepare_chat_prompt -> build_msg_entries — with the state's limits
    // installed; an image over the cap must be refused vision_input_too_large.
    let mut state = app(false);
    {
        let s = Arc::get_mut(&mut state).unwrap();
        s.vision_config = Some(atlas_core::config::VisionConfig {
            is_glm5_next: false,
            is_qwen3vl_video: false,
            in_channels: 3,
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            intermediate_size: 4_304,
            out_hidden_size: 2_560,
            projection_intermediate_size: 0,
            rms_norm_eps: 1e-6,
            swiglu_limit: 0.0,
            deepstack_visual_indexes: vec![],
            image_pad_token_id: 0,
            video_pad_token_id: 0,
            image_start_token_id: 0,
            image_end_token_id: 0,
            video_start_token_id: 0,
            video_end_token_id: 0,
            max_pixels: None,
            min_pixels: None,
            video_pixels: None,
            token_budgets: atlas_core::config::VisionTokenBudgets::default(),
            omni: None,
        });
        s.vision_limits = Some(spark_model::vision_limits::VisionRowLimits {
            window_rows: 65_536,
            chunk_tokens: 16_388,
            request_rows: 100,
        });
        s.vision_budget = Some(spark_model::vision_budget::VisionRowBudget::new(65_536));
    }
    // PNG header claiming 20000x20000 (tiny body): probe sizes it way over
    // request_rows=100 before any pixel is decoded.
    let mut ihdr = vec![0u8; 24];
    ihdr[..8].copy_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    ihdr[8..12].copy_from_slice(&13u32.to_be_bytes());
    ihdr[12..16].copy_from_slice(b"IHDR");
    ihdr[16..20].copy_from_slice(&20_000u32.to_be_bytes());
    ihdr[20..24].copy_from_slice(&20_000u32.to_be_bytes());
    let mut req = request();
    req.messages = vec![ir::Message {
        role: ir::Role::User,
        content: vec![ir::ContentPart::Image(ir::message::ImageSource {
            data: ir::ImageData::Base64(
                format!("data:image/png;base64,{}", base64_encode(&ihdr)).into(),
            ),
        })],
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }];
    let err = match crate::api::chat::prepare::prepare_chat_prompt(&state, &mut req) {
        Err(e) => e,
        Ok(_) => panic!("over-cap image must be refused at admission"),
    };
    let (parts, body) = err.into_parts();
    assert_eq!(parts.status, 400);
    let bytes = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(axum::body::to_bytes(body, usize::MAX))
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["error"]["code"].as_str().unwrap_or_default(),
        "vision_input_too_large"
    );
}
fn base64_encode(b: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for c in b.chunks(3) {
        let n = (c[0] as u32) << 16
            | (*c.get(1).unwrap_or(&0) as u32) << 8
            | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            let idx = (n >> (18 - 6 * i)) & 63;
            out.push(if i < 2 + c.len() - 1 { T[idx as usize] as char } else { '=' });
        }
    }
    out
}

#[path = "tool_security_tests.rs"]
mod tool_security_tests;

#[path = "glm_literal_xml_tests.rs"]
mod glm_literal_xml_tests;

#[path = "qwen_literal_xml_tests.rs"]
mod qwen_literal_xml_tests;

#[test]
fn glm_native_literal_opener_stays_reasoning_and_next_real_call_still_binds() {
    let literal = vec![1, BACKTICK, TOOL, ORDINARY, BACKTICK];
    let (stream, deltas) = stream_tokens(true, true, literal.clone());
    assert!(!stream.thinking_done, "model=true; {deltas:?}");
    assert!(!stream.cancel_flag.load(Ordering::Acquire));
    let reasoning: String = deltas
        .iter()
        .filter_map(|delta| match delta {
            ir::StreamDelta::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(reasoning.contains("<tool_call>"), "model=true; {deltas:?}");
    assert!(
        !deltas
            .iter()
            .any(|delta| matches!(delta, ir::StreamDelta::ToolCallStart { .. }))
    );

    let mut with_call = literal.clone();
    with_call.extend_from_slice(CALL);
    let (stream, deltas) = stream_tokens(true, true, with_call);
    assert!(stream.thinking_done, "model=true; {deltas:?}");
    assert_eq!(
        deltas
            .iter()
            .filter(|delta| matches!(delta, ir::StreamDelta::ToolCallStart { .. }))
            .count(),
        1,
        "model=true; {deltas:?}"
    );
}

#[test]
fn glm_native_literal_opener_is_preserved_by_blocking_decode() {
    let state = app(true);
    let response = response(vec![1, BACKTICK, TOOL, ORDINARY, BACKTICK]);
    let (reasoning, content) =
        crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
    let combined = format!("{}{}", reasoning.as_deref().unwrap_or_default(), content);
    assert!(combined.contains("<tool_call>"), "combined={combined:?}");
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &state,
        &request(),
        &response,
        reasoning,
        content,
        true,
        None,
        0,
    );
    assert!(
        choice.tool_calls.is_empty(),
        "literal must not become a call"
    );
}
