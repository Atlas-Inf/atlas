// SPDX-License-Identifier: AGPL-3.0-only

//! HY4 forced tool calls (named and `tool_choice: "required"`) stream the
//! grammar's `<tool_calls:opensource>` envelope. `hy4::canonical` strips the
//! atomic envelope token per decode window, but when the model spells the
//! envelope with ordinary BPE pieces no window ever holds the whole marker,
//! so the pieces reached `choices[].delta.content`. The sanitizer must drop
//! them while the structured call still streams.

use super::super::*;
use super::streaming_frag::tool_def;
use super::streaming_leak::stream_through_pipeline_with;
use crate::tool_parser::tail::hy4::Hy4Parser;

const SPELLED_OPEN: [&str; 7] = ["<", "tool", "_calls", ":", "opens", "ource", ">"];
const SPELLED_CLOSE: [&str; 7] = ["</", "tool", "_calls", ":", "opens", "ource", ">"];

fn weather_call(city: &'static str) -> [&'static str; 8] {
    [
        "<tool_call>",
        "get_weather",
        "<arg_key>",
        "city",
        "</arg_key>",
        "<arg_value>",
        city,
        "</tool_call>",
    ]
}

fn run(calls: &[[&'static str; 8]]) -> (Vec<(String, String)>, String) {
    let mut chunks: Vec<&str> = SPELLED_OPEN.to_vec();
    for call in calls {
        chunks.extend(call.iter().copied());
        // The arg value closer sits before the call closer.
        let at = chunks.len() - 1;
        chunks.insert(at, "</arg_value>");
    }
    chunks.extend(SPELLED_CLOSE);
    let mut det = StreamingToolDetector::new_with_tools(vec![tool_def(
        "get_weather",
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    )]);
    det.set_promote_bare_names(Hy4Parser.promotes_bare_call_names());
    stream_through_pipeline_with(Hy4Parser.leak_markers(), det, &chunks)
}

fn assert_clean(calls: &[(String, String)], content: &str, cities: &[&str]) {
    assert!(
        !content.contains("tool_calls") && !content.contains("opensource"),
        "envelope marker text leaked into content: {content:?}"
    );
    assert!(
        content.trim().is_empty(),
        "no content expected: {content:?}"
    );
    assert_eq!(calls.len(), cities.len(), "{calls:?}");
    for ((name, args), city) in calls.iter().zip(cities) {
        assert_eq!(name, "get_weather");
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(args).unwrap(),
            serde_json::json!({ "city": city })
        );
    }
}

#[test]
fn hy4_named_forced_call_spelled_envelope_not_in_content() {
    let (calls, content) = run(&[weather_call("Paris")]);
    assert_clean(&calls, &content, &["Paris"]);
}

#[test]
fn hy4_required_parallel_calls_spelled_envelope_not_in_content() {
    let (calls, content) = run(&[weather_call("Paris"), weather_call("Oslo")]);
    assert_clean(&calls, &content, &["Paris", "Oslo"]);
}
