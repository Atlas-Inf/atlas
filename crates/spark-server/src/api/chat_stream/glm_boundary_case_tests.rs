// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

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

#[path = "tool_security_tests.rs"]
mod tool_security_tests;

#[path = "glm_literal_xml_tests.rs"]
mod glm_literal_xml_tests;

#[path = "qwen_literal_xml_tests.rs"]
mod qwen_literal_xml_tests;

#[path = "glm_native_literal_tests.rs"]
mod glm_native_literal_tests;
