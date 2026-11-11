// SPDX-License-Identifier: AGPL-3.0-only
//! Token -> Done regressions for the Qwen XML failures found on the pair.
use super::*;
use crate::api::chat_stream::handle_done::handle_done;

fn qwen() -> Arc<AppState> {
    app_for_model("qwen4_exp", tool_parser::ToolCallFormat::Qwen3Coder)
}

fn run(tools: bool, tokens: Vec<u32>) -> Vec<ir::StreamDelta> {
    let count = tokens.len();
    let (ctx, mut stream, mut deltas) = stream_for_app(qwen(), tools, tokens);
    deltas.extend(handle_done(
        &mut stream,
        &ctx,
        "stop".into(),
        count,
        1.0,
        1.0,
        0,
        0,
        0,
    ));
    assert!(!stream.cancel_flag.load(Ordering::Acquire));
    deltas
}

#[test]
fn qwen_literal_xml_preserves_split_role_word_with_and_without_tools() {
    for tools in [false, true] {
        for (open, marker) in [(24, "</tool_call>"), (30, "<tool_call>")] {
            let deltas = run(tools, vec![END, 23, open, 25, 26, 27]);
            let content: String = deltas
                .iter()
                .filter_map(|d| match d {
                    ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                content,
                format!("The tag is `{marker}` and this is prose. END."),
                "{deltas:?}"
            );
            assert!(
                !deltas
                    .iter()
                    .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
            );
        }
    }
}

#[test]
fn qwen_literal_xml_plain_opener_survives_detector_done_flush() {
    let deltas = run(true, vec![END, 28, TOOL, 29]);
    let content: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        content, "The delimiter <tool_call> opens a tool invocation. END.",
        "{deltas:?}"
    );
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
    );
}

#[test]
fn qwen_literal_xml_fix_keeps_real_function_parameter_calls() {
    let deltas = run(true, vec![END, TOOL, 31, 11]);
    let names: Vec<_> = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["get_weather"], "{deltas:?}");
    let args: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&args).unwrap(),
        json!({"city":"Oslo"})
    );
    assert!(deltas.iter().any(
        |d| matches!(d, ir::StreamDelta::Finish { reason, .. } if reason.as_wire() == "tool_calls")
    ));
}

#[test]
fn qwen_literal_xml_does_not_disable_legacy_chatml_filtering() {
    assert!(!qwen().tokenizer.filters_bare_role_literals());
    assert!(!app(true).tokenizer.filters_bare_role_literals());
    assert!(app(false).tokenizer.filters_bare_role_literals());
    for word in ["tool", "user", "assistant"] {
        let mut text = word.to_owned();
        crate::api::chat_stream::handle_token::strip_bare_role_literal(&mut text, false);
        assert!(text.is_empty());
    }
}

#[test]
fn qwen_literal_xml_quoted_fenced_and_malformed_examples_stay_content() {
    // Existing fixture tokens contain complete literal examples; the split
    // token regression above covers the separate role-word emission.
    for token in [17, 18, 20, 32] {
        let expected = qwen().tokenizer.decode(&[token]).unwrap();
        let deltas = run(true, vec![END, token]);
        let content: String = deltas
            .iter()
            .filter_map(|d| match d {
                ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, expected, "{deltas:?}");
        assert!(
            !deltas
                .iter()
                .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
        );
    }
}

#[test]
fn qwen_literal_xml_unsupported_closed_and_unclosed_envelopes_are_content() {
    for token in [33, 34] {
        let expected = qwen().tokenizer.decode(&[token]).unwrap();
        let deltas = run(true, vec![END, token]);
        let content: String = deltas
            .iter()
            .filter_map(|d| match d {
                ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(content, expected, "{deltas:?}");
        assert!(
            !deltas
                .iter()
                .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
        );
    }
}

#[test]
fn qwen_literal_xml_reasoning_preserves_bare_and_split_tags_until_think_end() {
    for tools in [false, true] {
        for reasoning_tokens in [vec![19], vec![28, 30, 25, 26, 29], vec![33]] {
            let expected = qwen().tokenizer.decode(&reasoning_tokens).unwrap();
            let mut tokens = reasoning_tokens;
            tokens.extend([END, ORDINARY]);
            let deltas = run(tools, tokens);
            let reasoning: String = deltas
                .iter()
                .filter_map(|d| match d {
                    ir::StreamDelta::Reasoning { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            let content: String = deltas
                .iter()
                .filter_map(|d| match d {
                    ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
                    _ => None,
                })
                .collect();
            assert_eq!(reasoning, expected, "{deltas:?}");
            assert_eq!(content, "ordinary", "{deltas:?}");
            assert!(
                !deltas
                    .iter()
                    .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
            );
        }
    }
}

#[test]
fn qwen_literal_xml_reasoning_survives_done_without_think_end() {
    let tokens = vec![28, 30, 25, 26, 29];
    let expected = qwen().tokenizer.decode(&tokens).unwrap();
    let deltas = run(true, tokens);
    let reasoning: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, expected, "{deltas:?}");
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::Content { .. }))
    );
}

#[test]
fn qwen_literal_xml_reasoning_keeps_legacy_leak_guard() {
    let (_, stream, _) = stream_for_app(app(false), true, vec![28, 30, 25, 26, 29]);
    assert!(stream.cancel_flag.load(Ordering::Acquire));
    assert_eq!(stream.guard_stop, Some("in_think_tool_leak"));
}

#[test]
fn qwen_literal_xml_reasoning_does_not_consume_a_later_real_call() {
    let deltas = run(true, vec![28, 30, 25, 26, 29, END, TOOL, 31, 11]);
    let reasoning: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, LITERAL_CONTENT[2]);
    let names: Vec<_> = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::ToolCallStart { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, ["get_weather"]);
    assert!(deltas.iter().any(
        |d| matches!(d, ir::StreamDelta::Finish { reason, .. } if reason.as_wire() == "tool_calls")
    ));
}

/// N106: the fenced-XML newline survives when ```xml, \\n, and <tool_call>
/// arrive as SEPARATE tokens — the whitespace held while markup is ambiguous
/// must be released, not dropped.
#[test]
fn qwen_literal_xml_split_tokens_keep_the_fence_newline() {
    const XML_FENCE: u32 = 42;
    const NEWLINE: u32 = 43;
    // "```xml" "\n" "<tool_call>" body "</tool_call>" "\n" "```"
    let deltas = run(
        true,
        vec![
            END, XML_FENCE, NEWLINE, 3, 4, 5, 6, 7, 8, 9, 10, 11, NEWLINE, 15, 15, 15,
        ],
    );
    let content: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        content,
        "```xml\n<tool_call>get_weather<arg_key>city</arg_key><arg_value>Oslo</arg_value></tool_call>\n```",
        "{deltas:?}"
    );
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. })),
        "{deltas:?}"
    );
}
