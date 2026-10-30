// SPDX-License-Identifier: AGPL-3.0-only
//! Literal GLM reasoning must not be mistaken for a native tool boundary.

use super::*;
use crate::api::chat_stream::handle_done::handle_done;

fn finish(tools: bool, tokens: &[u32], reason: &str) -> Vec<ir::StreamDelta> {
    let (ctx, mut stream, mut deltas) = stream_tokens_with_context(true, tools, tokens.to_vec());
    assert!(!stream.cancel_flag.load(Ordering::Acquire));
    assert_eq!(stream.guard_stop, None);
    assert!(!stream.reasoning_xml_leak_detected);
    deltas.extend(handle_done(
        &mut stream,
        &ctx,
        reason.into(),
        tokens.len(),
        1.0,
        1.0,
        0,
        0,
        0,
    ));
    let ids: Vec<_> = deltas
        .iter()
        .flat_map(|delta| match delta {
            ir::StreamDelta::Reasoning { token_ids, .. }
            | ir::StreamDelta::Content { token_ids, .. }
            | ir::StreamDelta::ToolCallArgs { token_ids, .. }
            | ir::StreamDelta::Finish { token_ids, .. } => token_ids.as_slice(),
            _ => &[],
        })
        .copied()
        .collect();
    assert_eq!(ids, tokens, "each sampled ID exactly once: {deltas:?}");
    assert!(stream.pending_token_ids.is_empty());
    assert!(deltas.iter().any(|delta| matches!(delta,
        ir::StreamDelta::Finish { usage, guard_stop: None, .. }
        if usage.completion_tokens == tokens.len()
    )));
    deltas
}

fn text(deltas: &[ir::StreamDelta], reasoning: bool) -> String {
    deltas
        .iter()
        .filter_map(|delta| match delta {
            ir::StreamDelta::Reasoning { text, .. } if reasoning => Some(text.as_str()),
            ir::StreamDelta::Content { text, .. } if !reasoning => Some(text.as_str()),
            _ => None,
        })
        .collect()
}

#[test]
fn glm_literal_xml_reasoning_matches_blocking_through_end_and_done() {
    // 19 contains a whole textual opener but is not native TOOL (3).
    // 30/25/26 spell the same opener across three ordinary token IDs.
    // 35 completes an invalid example naming a real tool and unknown argument.
    let cases = [vec![19], vec![28, 30, 25, 26, 29], vec![30, 25, 26, 35]];
    for tools in [false, true] {
        for explicit_end in [false, true] {
            for reason in ["stop", "length"] {
                for reasoning_tokens in &cases {
                    let state = app(true);
                    let expected = state.tokenizer.decode(reasoning_tokens).unwrap();
                    let mut tokens = reasoning_tokens.clone();
                    if explicit_end {
                        tokens.extend([END, ORDINARY]);
                    }
                    let (blocking_reasoning, blocking_content) =
                        crate::api::chat_blocking::decode_response_text(
                            &state,
                            &response(tokens.clone()),
                            true,
                            tools,
                            false,
                        );
                    let deltas = finish(tools, &tokens, reason);
                    assert_eq!(text(&deltas, true), expected, "{deltas:?}");
                    assert_eq!(blocking_reasoning.as_deref(), Some(expected.as_str()));
                    assert_eq!(text(&deltas, false), blocking_content);
                    assert_eq!(blocking_content, if explicit_end { "ordinary" } else { "" });
                    assert!(
                        !deltas
                            .iter()
                            .any(|delta| matches!(delta, ir::StreamDelta::ToolCallStart { .. }))
                    );
                    assert!(deltas.iter().any(|delta| matches!(
                        delta, ir::StreamDelta::Finish { reason: actual, .. }
                        if actual.as_wire() == reason
                    )));
                }
            }
        }
    }
}

#[test]
fn glm_literal_xml_reasoning_keeps_later_native_call_with_or_without_end() {
    for explicit_end in [false, true] {
        let reasoning_tokens = vec![28, 30, 25, 26, 29];
        let expected = app(true).tokenizer.decode(&reasoning_tokens).unwrap();
        let mut tokens = reasoning_tokens;
        if explicit_end {
            tokens.push(END);
        }
        tokens.extend_from_slice(CALL);
        let deltas = finish(true, &tokens, "stop");
        assert_eq!(text(&deltas, true), expected);
        assert!(text(&deltas, false).is_empty());
        let names: Vec<_> = deltas
            .iter()
            .filter_map(|delta| match delta {
                ir::StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, ["get_weather"]);
        let args: String = deltas
            .iter()
            .filter_map(|delta| match delta {
                ir::StreamDelta::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&args).unwrap(),
            json!({"city":"Oslo"})
        );
        assert!(deltas.iter().any(|delta| matches!(
            delta, ir::StreamDelta::Finish { reason, .. }
            if reason.as_wire() == "tool_calls"
        )));
    }
}

#[test]
fn glm_literal_xml_reasoning_preserves_closing_tag_bytes() {
    for tools in [false, true] {
        let expected = app(true).tokenizer.decode(&[21]).unwrap();
        let deltas = finish(tools, &[21], "stop");
        assert_eq!(text(&deltas, true), expected);
        assert_eq!(text(&deltas, true).matches("</tool_call>").count(), 2);
    }
}

#[test]
fn glm_native_opener_in_prose_stays_content_blocking_and_streamed() {
    // SparkGLM 0.2.0 harness "plain": the model writes the native opener
    // token in an unquoted sentence. No call follows, so blocking and the
    // stream both keep the whole sentence, the space before the opener too.
    let tokens = [1, END, 28, TOOL, 29];
    let expected = "The delimiter <tool_call> opens a tool invocation. END.";
    for tools in [false, true] {
        let state = app(true);
        let response = response(tokens.to_vec());
        let (reasoning, content) =
            crate::api::chat_blocking::decode_response_text(&state, &response, true, tools, false);
        let choice = crate::api::chat_blocking_choice::build_choice_message(
            &state,
            &request(),
            &response,
            reasoning,
            content,
            tools,
            None,
            0,
        );
        assert!(choice.tool_calls.is_empty(), "tools={tools}; {choice:?}");
        assert_eq!(choice.content.as_deref(), Some(expected), "tools={tools}");
        let deltas = finish(tools, &tokens, "stop");
        assert_eq!(text(&deltas, false), expected, "tools={tools}; {deltas:?}");
    }
}
