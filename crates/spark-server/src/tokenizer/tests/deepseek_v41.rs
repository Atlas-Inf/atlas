// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4.1 prompt encoding against the checkpoint's own encoder
//! (deepseek-ai/DeepSeek-V4.1-Flash-1017 `encoding/encoding.py`, revision
//! 1bae0e10). `test_*` are DeepSeek's five fixture pairs, unedited. `extra_*`
//! are further inputs encoded by that same file's `encode_case`.

use serde_json::{Value, json};

use super::deepseek_v4::assert_exact;
use crate::tokenizer::deepseek_v4::{Dialect, encode_messages};

macro_rules! pairs {
    ($input:literal, $output:literal, $($n:literal),+) => {
        [$((
            include_str!(concat!("../../../test_data/deepseek_v41_encoding/", $input, $n, ".json")),
            include_str!(concat!("../../../test_data/deepseek_v41_encoding/", $output, $n, ".txt")),
        )),+]
    };
}

/// Encode a fixture as the reference's `encode_case` does: chat mode unless
/// the case says otherwise, tools and effort from the case.
fn render(input: &str) -> anyhow::Result<String> {
    let case: Value = serde_json::from_str(input).expect("fixture JSON");
    let messages = case.get("messages").unwrap_or(&case);
    let effort = case.get("reasoning_effort").map(|effort| match effort {
        Value::String(name) => name.clone(),
        number => number.to_string(),
    });
    encode_messages(
        Dialect::V41,
        messages.as_array().expect("messages"),
        case.get("tools")
            .and_then(Value::as_array)
            .map(Vec::as_slice),
        case.get("thinking_mode").and_then(Value::as_str) == Some("thinking"),
        effort.as_deref(),
    )
}

fn chat(messages: Value) -> anyhow::Result<String> {
    encode_messages(
        Dialect::V41,
        messages.as_array().unwrap(),
        None,
        false,
        None,
    )
}

#[test]
fn official_encoding_fixtures_are_byte_exact() {
    for (input, output) in pairs!("test_input_", "test_output_", 1, 2, 3, 4, 5) {
        assert_exact(&render(input).expect("official encoding"), output);
    }
}

/// Mid-conversation system messages, numeric effort, namespaced tools and
/// calls, out-of-order and image-bearing tool results, a transcript with no
/// user message, tasks, and reasoning history with and without tools.
#[test]
fn reference_encoder_cases_are_byte_exact() {
    for (input, output) in pairs!("extra_input_", "extra_output_", 1, 2, 3, 4, 5, 6, 7, 8) {
        assert_exact(&render(input).expect("reference case"), output);
    }
}

#[test]
fn a_developer_message_is_a_system_message() {
    let conversation = |role: &str| {
        chat(json!([
            {"role": role, "content": "Rules."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello."},
            {"role": role, "content": "New rule."},
        ]))
        .unwrap()
    };
    assert_eq!(conversation("developer"), conversation("system"));
    assert!(conversation("developer").ends_with("<｜System｜>New rule.<｜Assistant｜></think>"));
}

#[test]
fn request_tools_without_a_system_message_open_a_system_block() {
    let tools = json!([{"type": "function", "function": {"name": "now", "parameters": {}}}]);
    let messages = json!([{"role": "user", "content": "time?"}]);
    let rendered = encode_messages(
        Dialect::V41,
        messages.as_array().unwrap(),
        tools.as_array().map(Vec::as_slice),
        false,
        None,
    )
    .unwrap();
    assert!(rendered.starts_with("<｜begin▁of▁sentence｜><｜System｜>\n\n## Tools\n"));
    assert!(rendered.contains("writing a \"<｜DSML｜ calls>\" block"));
    assert!(rendered.ends_with("<｜User｜>time?<｜Assistant｜></think>"));
}

#[test]
fn reasoning_effort_is_a_budget_from_1_to_100() {
    let messages = json!([{"role": "user", "content": "test"}]);
    let messages = messages.as_array().unwrap();
    let thinking = |effort| encode_messages(Dialect::V41, messages, None, true, effort);
    for (effort, budget) in [
        (None, 75),
        (Some("low"), 50),
        (Some("xhigh"), 100),
        (Some("37"), 37),
    ] {
        assert!(
            thinking(effort).unwrap().starts_with(&format!(
                "<｜begin▁of▁sentence｜><｜System｜>Reasoning Effort: {budget} (range 1-100, "
            )),
            "{effort:?}"
        );
    }
    for effort in ["0", "101", "fast", "7.5"] {
        assert!(thinking(Some(effort)).is_err(), "{effort}");
    }
    // Chat mode renders no effort line and no system token for it.
    let direct = encode_messages(Dialect::V41, messages, None, false, Some("max")).unwrap();
    assert_eq!(
        direct,
        "<｜begin▁of▁sentence｜><｜User｜>test<｜Assistant｜></think>"
    );
}

#[test]
fn text_may_not_spell_the_image_token_and_an_image_needs_a_source() {
    for messages in [
        json!([{"role": "user", "content": "see <｜deepseek_image｜>"}]),
        json!([{"role": "assistant", "content": "", "reasoning_content": "<｜deepseek_image｜>"}]),
        json!([{"role": "user", "content": [{"type": "text", "text": "<｜deepseek_image｜>"}]}]),
        json!([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": ""}}]}]),
        json!([{"role": "user", "content": [{"type": "image"}]}]),
    ] {
        assert!(chat(messages.clone()).is_err(), "{messages}");
    }
}

#[test]
fn a_tool_named_under_two_namespaces_is_refused() {
    let call = |function: Value, namespace: Value| {
        chat(json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "namespace": namespace, "function": function}
            ]},
        ]))
    };
    assert!(call(json!({"name": "a::f", "arguments": {}}), json!("a")).is_ok());
    assert!(call(json!({"name": "a::f", "arguments": {}}), json!("b")).is_err());
    assert!(call(json!({"name": "a::b::f", "arguments": {}}), Value::Null).is_err());
    assert!(call(json!({"name": "f", "arguments": {}}), json!("a::b")).is_err());
}

/// The reference reads `tool_call.get("namespace") or function.get("namespace")`.
#[test]
fn an_empty_call_namespace_falls_back_to_the_function_namespace() {
    for (namespace, name) in [
        (json!(""), "fs::read"),
        (json!({}), "fs::read"),
        (json!("io"), "io::read"),
        (json!({"name": "io"}), "io::read"),
    ] {
        let prompt = chat(json!([
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": "", "tool_calls": [{
                "type": "function",
                "namespace": namespace,
                "function": {"name": "read", "namespace": "fs", "arguments": {}},
            }]},
        ]))
        .unwrap();
        assert!(
            prompt.contains(&format!("invoke name=\"{name}\">")),
            "{prompt}"
        );
    }
}
