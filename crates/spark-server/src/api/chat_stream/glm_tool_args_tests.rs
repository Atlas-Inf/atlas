// SPDX-License-Identifier: AGPL-3.0-only
//! Delivered GLM tool arguments, blocking and streamed.

use super::*;

/// Delivered arguments keep the model's key and item order, blocking and
/// streamed, so a forced call returns what was decoded, re-encoded compactly.
#[test]
fn glm_tool_arguments_keep_generated_key_order() {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model":"fixture", "messages":[{"role":"user","content":"ops"}],
        "tools":[{"type":"function","function":{"name":"get_weather",
            "parameters":{"type":"object","properties":{"ops":{"type":"array"}},
                "required":["ops"]}}}],
        "tool_choice":{"type":"function","function":{"name":"get_weather"}},
        "max_tokens":192
    }))
    .unwrap();
    let req: ir::ChatRequest = wire.into();
    let expected = format!(
        "{{\"ops\":{}}}",
        serde_json::to_string(&serde_json::from_str::<serde_json::Value>(ORDERED_OPS).unwrap())
            .unwrap()
    );
    assert_eq!(
        expected,
        r#"{"ops":[{"url":"u","headers":{"b":1,"a":2}},{"command":"c"}]}"#
    );
    let tokens = vec![1, END, TOOL, 4, 5, OPS, 7, 8, OPS_VALUE, 10, 11];
    let state = app(true);
    let response = response(tokens.clone());
    let (reasoning, content) =
        crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &state, &req, &response, reasoning, content, true, None, 0,
    );
    assert_eq!(choice.tool_calls.len(), 1, "{choice:?}");
    assert_eq!(
        serde_json::to_string(&choice.tool_calls[0].arguments).unwrap(),
        expected
    );
    let (_, _, deltas) = stream_for_request(app(true), true, tokens, req);
    let streamed: String = deltas
        .iter()
        .filter_map(|delta| match delta {
            ir::StreamDelta::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(streamed, expected, "{deltas:?}");
}
