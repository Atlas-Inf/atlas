// SPDX-License-Identifier: AGPL-3.0-only
//
// Golden tests for the Anthropic streaming translator's event framing.
// These characterize the wire output Claude Code depends on; the
// translator now consumes neutral `ir::StreamDelta`s instead of
// re-parsed OpenAI chunk JSON, but the emitted event sequences are
// unchanged.

use super::super::translator::{AnthropicTranslator, SseEvent};
use crate::ir::response::{FinishReason, Usage};
use crate::ir::stream::StreamDelta;

fn usage(prompt: usize, completion: usize) -> Usage {
    Usage {
        prompt_tokens: prompt,
        completion_tokens: completion,
        cached_prompt_tokens: 0,
        reasoning_tokens: 0,
        accepted_prediction_tokens: 0,
        time_to_first_token_ms: 0.0,
        response_tokens_per_second: 0.0,
    }
}

fn drive(deltas: Vec<StreamDelta>) -> Vec<SseEvent> {
    let mut t = AnthropicTranslator::new("m".to_string());
    let mut out = Vec::new();
    for d in &deltas {
        t.on_delta(d, &mut out);
    }
    t.finalize(&mut out);
    out
}

fn names(evs: &[SseEvent]) -> Vec<&str> {
    evs.iter().map(|e| e.event.as_str()).collect()
}

#[test]
fn text_stream_framing() {
    let evs = drive(vec![
        StreamDelta::Content {
            text: "Hi".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage: usage(3, 1),
            token_ids: Vec::new(),
            guard_stop: None,
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "text");
    assert_eq!(evs[2].data["delta"]["type"], "text_delta");
    assert_eq!(evs[2].data["delta"]["text"], "Hi");
    assert_eq!(evs[4].data["delta"]["stop_reason"], "end_turn");
    assert_eq!(evs[4].data["usage"]["output_tokens"], 1);
    // B1: the final message_delta patches input_tokens (usage arrives on
    // the terminal delta, after message_start already reported 0).
    assert_eq!(evs[4].data["usage"]["input_tokens"], 3);
    assert_eq!(evs[4].data["usage"]["cache_read_input_tokens"], 0);
    // message_start still opens with zero (usage unknown at that point)
    // and a minted msg_ id.
    assert_eq!(evs[0].data["message"]["usage"]["input_tokens"], 0);
    let id = evs[0].data["message"]["id"].as_str().unwrap();
    assert!(id.starts_with("msg_"), "unexpected id shape: {id}");
}

#[test]
fn final_usage_excludes_cache_reads_from_input_tokens() {
    let evs = drive(vec![StreamDelta::Finish {
        reason: FinishReason::Stop,
        usage: Usage {
            cached_prompt_tokens: 3,
            ..usage(10, 1)
        },
        token_ids: Vec::new(),
        guard_stop: None,
    }]);
    let delta = evs.iter().find(|e| e.event == "message_delta").unwrap();
    assert_eq!(delta.data["usage"]["input_tokens"], 7);
    assert_eq!(delta.data["usage"]["cache_read_input_tokens"], 3);
}

#[test]
fn thinking_then_text_framing() {
    let evs = drive(vec![
        StreamDelta::Reasoning {
            text: "think".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Content {
            text: "answer".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage: usage(0, 0),
            token_ids: Vec::new(),
            guard_stop: None,
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "thinking");
    assert_eq!(evs[2].data["delta"]["type"], "thinking_delta");
    assert_eq!(evs[2].data["delta"]["thinking"], "think");
    assert_eq!(evs[5].data["delta"]["type"], "text_delta");
    assert_eq!(evs[5].data["delta"]["text"], "answer");
}

#[test]
fn tool_call_stream_framing() {
    let evs = drive(vec![
        StreamDelta::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "get_weather".into(),
        },
        StreamDelta::ToolCallArgs {
            index: 0,
            fragment: "{\"city\":\"SF\"}".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::ToolCalls,
            usage: usage(0, 0),
            token_ids: Vec::new(),
            guard_stop: None,
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["type"], "tool_use");
    assert_eq!(evs[1].data["content_block"]["name"], "get_weather");
    assert_eq!(evs[1].data["content_block"]["id"], "call_1");
    assert_eq!(evs[2].data["delta"]["type"], "input_json_delta");
    assert_eq!(evs[2].data["delta"]["partial_json"], "{\"city\":\"SF\"}");
    assert_eq!(evs[4].data["delta"]["stop_reason"], "tool_use");
}

#[test]
fn stream_error_is_not_reported_as_success() {
    for (code, upstream_type, expected_type, message) in [
        (500, "server_error", "api_error", "GPU unavailable"),
        (
            400,
            "invalid_request_error",
            "invalid_request_error",
            "Image too large",
        ),
    ] {
        let evs = drive(vec![StreamDelta::Error {
            message: serde_json::json!({
                "error": {"type": upstream_type, "code": code, "message": message}
            })
            .to_string(),
        }]);
        assert_eq!(names(&evs), ["error"]);
        assert_eq!(
            evs[0].data,
            serde_json::json!({
                "type": "error",
                "error": {"type": expected_type, "message": message}
            })
        );
    }
}

#[test]
fn stream_error_interrupts_an_open_block() {
    let evs = drive(vec![
        StreamDelta::ToolCallStart {
            index: 0,
            id: "call_1".into(),
            name: "get_weather".into(),
        },
        StreamDelta::ToolCallArgs {
            index: 0,
            fragment: "{\"city\":".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Error {
            message: "GPU unavailable".into(),
        },
        StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage: usage(3, 1),
            token_ids: Vec::new(),
            guard_stop: None,
        },
    ]);
    assert_eq!(
        names(&evs),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "error"
        ]
    );
    assert_eq!(evs[3].data["error"]["type"], "api_error");
    assert_eq!(evs[3].data["error"]["message"], "GPU unavailable");
}

#[tokio::test]
async fn stream_error_closes_the_sse_body() {
    use futures::StreamExt;

    let deltas = futures::stream::iter([StreamDelta::Error {
        message: "GPU unavailable".into(),
    }])
    .chain(futures::stream::pending());
    let response = super::super::handlers_stream::anthropic_sse_from_deltas(
        Box::pin(deltas),
        "m".into(),
        None,
    );
    let bytes = tokio::time::timeout(
        std::time::Duration::from_secs(1),
        axum::body::to_bytes(response.into_body(), 1024),
    )
    .await
    .expect("error must close the SSE body without waiting for upstream EOF")
    .unwrap();
    let body = std::str::from_utf8(&bytes).unwrap();
    let data = body
        .strip_prefix("event: error\ndata: ")
        .and_then(|data| data.strip_suffix("\n\n"))
        .expect("a single error SSE frame");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(data).unwrap(),
        serde_json::json!({
            "type": "error",
            "error": {"type": "api_error", "message": "GPU unavailable"}
        })
    );
}

#[test]
fn stream_refusal_sets_stop_reason_without_repeating_text() {
    let refusal = "I cannot help with that request.";
    let text = format!("{refusal} Here is a safe alternative.");
    let evs = drive(vec![
        StreamDelta::Content {
            text: text.clone(),
            token_ids: Vec::new(),
        },
        StreamDelta::Refusal {
            text: refusal.into(),
        },
        StreamDelta::Finish {
            reason: FinishReason::Stop,
            usage: usage(3, 12),
            token_ids: Vec::new(),
            guard_stop: None,
        },
    ]);
    assert_eq!(
        names(&evs),
        [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[2].data["delta"]["text"], text);
    assert_eq!(evs[4].data["delta"]["stop_reason"], "refusal");
    assert_eq!(evs[4].data["usage"]["output_tokens"], 12);
}

#[test]
fn multi_tool_calls_close_and_reopen_blocks() {
    // Two tool calls in one turn: the second start must close the first
    // block and open a fresh one at the next index (Claude Code executes
    // each block once).
    let evs = drive(vec![
        StreamDelta::ToolCallStart {
            index: 0,
            id: "c1".into(),
            name: "read".into(),
        },
        StreamDelta::ToolCallArgs {
            index: 0,
            fragment: "{}".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::ToolCallStart {
            index: 1,
            id: "c2".into(),
            name: "bash".into(),
        },
        StreamDelta::ToolCallArgs {
            index: 1,
            fragment: "{\"command\":\"ls\"}".into(),
            token_ids: Vec::new(),
        },
        StreamDelta::Finish {
            reason: FinishReason::ToolCalls,
            usage: usage(0, 2),
            token_ids: Vec::new(),
            guard_stop: None,
        },
    ]);
    assert_eq!(
        names(&evs),
        vec![
            "message_start",
            "content_block_start", // tool 0
            "content_block_delta",
            "content_block_stop",  // tool 0 closed by tool 1 start
            "content_block_start", // tool 1
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(evs[1].data["content_block"]["id"], "c1");
    assert_eq!(evs[1].data["index"], 0);
    assert_eq!(evs[4].data["content_block"]["id"], "c2");
    assert_eq!(evs[4].data["index"], 1);
}

#[test]
fn finalize_without_finish_emits_end_turn() {
    // Upstream died before the Finish delta: finalize must still close
    // the block and end the message coherently.
    let mut t = AnthropicTranslator::new("m".to_string());
    let mut out = Vec::new();
    t.on_delta(
        &StreamDelta::Content {
            text: "partial".into(),
            token_ids: Vec::new(),
        },
        &mut out,
    );
    t.finalize(&mut out);
    assert_eq!(
        names(&out),
        vec![
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
    );
    assert_eq!(out[4].data["delta"]["stop_reason"], "end_turn");
}
