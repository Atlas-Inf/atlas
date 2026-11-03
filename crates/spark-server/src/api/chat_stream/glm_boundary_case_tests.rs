// SPDX-License-Identifier: AGPL-3.0-only
//! Upstream boundary cases on the GLM stream fixture: Poolside header
//! holding, the #172 refusal prefix and the #236 vision admission cap.

use super::*;

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
    // Explicitly closing reasoning keeps an undeclared native call's own name:
    // it is never mapped onto the request's sole declared tool.
    let (_, deltas) = stream(true, true, true, false);
    assert_eq!(tool_names(&deltas), ["unknown_tool"]);
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
    super::super::tool_handlers::handle_tool_call_start(
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

#[test]
#[ignore = "#236 test never compiled on the stack (its module was unwired); fails once wired"]
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
            data: ir::ImageData::Base64(format!("data:image/png;base64,{}", base64_encode(&ihdr))),
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
            out.push(if i < 2 + c.len() - 1 {
                T[idx as usize] as char
            } else {
                '='
            });
        }
    }
    out
}

#[test]
fn glm_native_call_to_undeclared_tool_reaches_the_client() {
    // The native opener is a call, not prose. An undeclared name reaches the
    // client unchanged (never mapped onto a declared tool), whose own
    // unknown-tool result lets an agent's turn continue.
    for explicit_close in [true, false] {
        let (state, deltas) = stream(true, true, explicit_close, false);
        assert_eq!(
            tool_names(&deltas),
            ["unknown_tool"],
            "explicit={explicit_close}"
        );
        assert!(!text_of(&deltas).contains("<tool_call>"), "{deltas:?}");
        assert!(!state.stop_string_triggered);
        let blocking = blocking_choice(tokens(explicit_close, false));
        assert_eq!(blocking.tool_calls.len(), 1, "{blocking:?}");
        assert_eq!(blocking.tool_calls[0].name, "unknown_tool");
        assert_eq!(blocking.tool_calls[0].arguments["city"], "Oslo");
        assert_eq!(blocking.finish_reason, ir::FinishReason::ToolCalls);
    }
    assert_eq!(
        finish_reason(tokens(true, false), request()),
        ir::FinishReason::ToolCalls
    );
    // The same envelope spelled without the native opener stays prose.
    let mut spelled = vec![1, END, 30, 25, 26];
    spelled.extend_from_slice(&tokens(true, false)[3..]);
    let (state, deltas) = stream_tokens(true, true, spelled.clone());
    assert_eq!(
        text_of(&deltas),
        "<tool_call>unknown_tool<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>"
    );
    assert!(tool_names(&deltas).is_empty());
    assert!(!state.stop_string_triggered);
    assert!(blocking_choice(spelled).tool_calls.is_empty());
}

/// The wire finish reason after a stream ends on its scheduler "stop".
fn finish_reason(tokens: Vec<u32>, request: ir::ChatRequest) -> ir::FinishReason {
    let len = tokens.len();
    let (ctx, mut state, _) = stream_for_request(app(true), true, tokens, request);
    let done = crate::api::chat_stream::handle_done::handle_done(
        &mut state,
        &ctx,
        "stop".into(),
        len,
        1.0,
        1.0,
        0,
        0,
        0,
    );
    done.into_iter()
        .find_map(|d| match d {
            ir::StreamDelta::Finish { reason, .. } => Some(reason),
            _ => None,
        })
        .expect("a finish delta")
}

fn blocking_choice(tokens: Vec<u32>) -> ir::Choice {
    let (state, req) = (app(true), request());
    let response = response(tokens);
    let (reasoning, content) =
        crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
    crate::api::chat_blocking_choice::build_choice_message(
        &state, &req, &response, reasoning, content, true, None, 0,
    )
}

fn tool_names(deltas: &[ir::StreamDelta]) -> Vec<&str> {
    deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect()
}

fn text_of(deltas: &[ir::StreamDelta]) -> String {
    deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_call_that_validation_rejects_finishes_stop() {
    // A declared write whose path carries a command separator fails
    // validation, so no call reaches the client and the stream must not claim
    // a tool turn.
    let write = vec![1, END, TOOL, 44, 5, 45, 7, 8, 46, 10, 11];
    let (_, deltas) = stream_tokens_for_write(write.clone());
    assert!(tool_names(&deltas).is_empty());
    assert!(
        text_of(&deltas).contains("[atlas] Tool call rejected"),
        "{deltas:?}"
    );
    assert_eq!(
        finish_reason(write, write_request()),
        ir::FinishReason::Stop
    );
    assert_eq!(
        finish_reason(tokens(true, true), request()),
        ir::FinishReason::ToolCalls
    );
}

fn stream_tokens_for_write(tokens: Vec<u32>) -> (StreamState, Vec<ir::StreamDelta>) {
    let (_, state, deltas) = stream_for_request(app(true), true, tokens, write_request());
    (state, deltas)
}

fn write_request() -> ir::ChatRequest {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model":"fixture", "messages":[{"role":"user","content":"save"}],
        "tools":[{"type":"function","function":{"name":"write",
            "parameters":{"type":"object","properties":{"path":{"type":"string"}},
                "required":["path"]}}}], "max_tokens":192
    }))
    .unwrap();
    wire.into()
}
