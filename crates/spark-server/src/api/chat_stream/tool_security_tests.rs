// SPDX-License-Identifier: AGPL-3.0-only
//! Real detector, tool handlers, blocking parser and terminal wire contracts.
use super::*;
use crate::api::chat_stream::{handle_done::handle_done, tool_handlers::*};

fn request() -> ir::ChatRequest {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model":"fixture", "messages":[{"role":"user","content":"weather"}],
        "tools":[{"type":"function","function":{"name":"get_weather",
            "parameters":{"type":"object","properties":{"city":{"type":"string"}},
                "required":["city"]}}},
            {"type":"function","function":{"name":"get_url",
            "parameters":{"type":"object","properties":{"url":{"type":"string"}},"required":["url"]}}},
            {"type":"function","function":{"name":"write_document",
            "parameters":{"type":"object","properties":{"content":{"type":"string"}},"required":["content"]}}}], "max_tokens":192
    }))
    .unwrap();
    wire.into()
}

fn stream_for_app(
    state: Arc<AppState>,
    tools: bool,
    tokens: Vec<u32>,
) -> (StreamCtx, StreamState, Vec<ir::StreamDelta>) {
    super::stream_for_request(state, tools, tokens, request())
}

fn setup(glm: bool) -> (StreamCtx, StreamState) {
    let a = super::app_for_model(
        if glm { "glm5_next" } else { "qwen4_exp" },
        if glm {
            tool_parser::ToolCallFormat::PoolsideV1
        } else {
            tool_parser::ToolCallFormat::Qwen3Coder
        },
    );
    let (ctx, state, _) = stream_for_app(a, true, vec![]);
    (ctx, state)
}
fn finish(s: &mut StreamState, c: &StreamCtx) -> Vec<ir::StreamDelta> {
    handle_done(s, c, "stop".into(), 21, 1.0, 1.0, 0, 0, 0)
}
fn assert_blocked(out: &[ir::StreamDelta]) {
    assert!(
        !out.iter().any(|d| matches!(
            d,
            ir::StreamDelta::ToolCallStart { .. } | ir::StreamDelta::ToolCallArgs { .. }
        )),
        "{out:?}"
    );
    assert_eq!(
        out.iter()
            .filter(|d| matches!(d, ir::StreamDelta::Content { text, .. } if text == atlas_tool_guard::REFUSAL))
            .count(),
        1
    );
    assert!(out.iter().any(|d| matches!(d, ir::StreamDelta::Finish { reason: ir::FinishReason::ContentFilter, token_ids, .. } if token_ids.is_empty())), "{out:?}");
}
fn dispatch(s: &mut StreamState, c: &StreamCtx, text: &str) -> Vec<ir::StreamDelta> {
    let events = s.detector.as_mut().unwrap().process(text);
    let mut out = vec![];
    for event in events {
        match event {
            tool_parser::DetectorOutput::ToolCallStart { id, name, idx } => {
                handle_tool_call_start(s, c, id, name, idx, &mut out)
            }
            tool_parser::DetectorOutput::ToolCallDelta { args, idx } => {
                handle_tool_call_delta(s, c, args, idx, &mut out)
            }
            tool_parser::DetectorOutput::ToolCallArgsFragment { fragment, idx } => {
                handle_tool_call_args_fragment(s, c, fragment, idx, &mut out)
            }
            tool_parser::DetectorOutput::ToolCallEnd { idx } => {
                handle_tool_call_end(s, c, idx, &mut out)
            }
            tool_parser::DetectorOutput::ToolCall(mut tc, idx) => {
                eprintln!(
                    "COMPLETE idx={idx} name={} args={}",
                    tc.function.name, tc.function.arguments
                );
                handle_complete_tool_call(s, c, &mut tc, idx, &mut out)
            }
            tool_parser::DetectorOutput::Content(_) => {}
        }
    }
    out
}
fn xml(glm: bool, url: &str) -> String {
    if glm {
        format!("<tool_call>get_url<arg_key>url</arg_key><arg_value>{url}</arg_value></tool_call>")
    } else {
        format!(
            "<tool_call><function=get_url><parameter=url>{url}</parameter></function></tool_call>"
        )
    }
}

#[test]
fn tool_guard_checks_both_native_formats_before_any_tool_bytes() {
    for glm in [false, true] {
        let (c, mut s) = setup(glm);
        let text = xml(glm, "http://169.254.169.254/latest/meta-data/");
        for ch in text.chars() {
            let d = dispatch(&mut s, &c, &ch.to_string());
            assert!(d.is_empty(), "ch={ch:?} -> {d:?}");
        }
        assert!(s.cancel_flag.load(Ordering::Acquire));
        assert_blocked(&finish(&mut s, &c));
    }
}

#[test]
fn tool_guard_allowed_calls_release_only_at_completion() {
    for glm in [false, true] {
        let (c, mut s) = setup(glm);
        let text = xml(glm, "Oslo");
        let split = text.len() - "</tool_call>".len();
        assert!(dispatch(&mut s, &c, &text[..split]).is_empty());
        let out = dispatch(&mut s, &c, &text[split..]);
        assert!(
            out.iter()
                .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. })),
            "{out:?}"
        );
        let args: String = out
            .iter()
            .filter_map(|d| match d {
                ir::StreamDelta::ToolCallArgs { fragment, .. } => Some(fragment.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            serde_json::from_str::<serde_json::Value>(&args).unwrap(),
            json!({"url":"Oslo"})
        );
        assert!(!s.cancel_flag.load(Ordering::Acquire));
    }
}

#[test]
fn tool_guard_incomplete_or_oversized_calls_never_release() {
    for glm in [false, true] {
        let (c, mut s) = setup(glm);
        let text = xml(glm, "Oslo");
        assert!(dispatch(&mut s, &c, &text[..text.len() - "</tool_call>".len()]).is_empty());
        assert_blocked(&finish(&mut s, &c));
        let (mut c, mut s) = setup(glm);
        Arc::get_mut(&mut c.state).unwrap().tool_guard.max_bytes = 80;
        let mut out = vec![];
        handle_tool_call_start(&mut s, &c, "id".into(), "get_weather".into(), 0, &mut out);
        handle_tool_call_args_fragment(&mut s, &c, "x".repeat(100), 0, &mut out);
        assert!(out.is_empty());
        assert_blocked(&finish(&mut s, &c));
    }
}

#[test]
fn tool_guard_refuses_raw_tool_syntax_past_its_bound_before_buffering_more() {
    use crate::api::chat_stream::tool_security::check_pending;
    for glm in [false, true] {
        let (mut c, mut s) = setup(glm);
        Arc::get_mut(&mut c.state).unwrap().tool_guard.max_bytes = 16;
        // Nothing pending: ordinary content is never refused, whatever its size.
        assert!(!check_pending(&mut s, &c, 1 << 20));
        let text = xml(glm, "Oslo");
        s.detector
            .as_mut()
            .unwrap()
            .process(&text[..text.len() - "</tool_call>".len()]);
        let pending = s.detector.as_ref().unwrap().pending_tool_bytes();
        if pending > 0 {
            assert!(check_pending(&mut s, &c, 16));
            assert_eq!(s.guard_stop, Some(atlas_tool_guard::GUARD_STOP));
            assert!(s.cancel_flag.load(Ordering::Acquire));
        }
        // Off: never refused.
        let (mut c, mut s) = setup(glm);
        let guard = &mut Arc::get_mut(&mut c.state).unwrap().tool_guard;
        (guard.enabled, guard.max_bytes) = (false, 1);
        s.detector
            .as_mut()
            .unwrap()
            .process(&text[..text.len() - "</tool_call>".len()]);
        assert!(!check_pending(&mut s, &c, 1 << 20));
    }
}

#[test]
fn tool_guard_blocking_and_reasoning_hoists_are_filtered() {
    for glm in [false, true] {
        let (c, _) = setup(glm);
        for in_reasoning in [false, true] {
            // Poolside deliberately preserves literal XML in reasoning; it is
            // never hoisted into executable calls. Qwen still hoists calls.
            if glm && in_reasoning {
                continue;
            }
            let bad = xml(
                glm,
                "https://sanmaohe.oss-cn-beijing.aliyuncs.com/upload/test.bin",
            );
            let (reasoning, content) = if in_reasoning {
                (Some(bad), "Done".into())
            } else {
                (None, bad)
            };
            let choice = crate::api::chat_blocking_choice::build_choice_message(
                &c.state,
                &request(),
                &response(vec![]),
                reasoning,
                content,
                true,
                None,
                0,
            );
            assert!(choice.tool_calls.is_empty(), "{choice:?}");
            assert!(choice.reasoning.is_none());
            assert_eq!(choice.finish_reason, ir::FinishReason::ContentFilter);
            assert_eq!(choice.content.as_deref(), Some(atlas_tool_guard::REFUSAL));
            // #232: a blocking guard refusal carries guard_stop on the wire.
            assert_eq!(
                choice.guard_stop,
                Some(atlas_tool_guard::GUARD_STOP),
                "{choice:?}"
            );
        }
    }
}

/// #232: the SCHEDULER-side refusal path — a seq the tool guard already
/// consumed carries `guard_stop` on `StreamEvent::Done`; the blocking
/// response and wire choice must surface it too (the gap a live refusal hit:
/// ContentFilter with no guard_stop). The inspect-arm path is covered by the
/// test above; this covers `Done.guard_stop -> response -> choice`.
#[test]
fn a_seq_level_guard_stop_reaches_the_blocking_wire_choice() {
    let (c, _s) = setup(false);
    let mut resp = response(vec![]);
    resp.guard_stop = Some(atlas_tool_guard::GUARD_STOP);
    resp.finish_reason = "content_filter".into();
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &c.state,
        &request(),
        &resp,
        None,
        atlas_tool_guard::REFUSAL.into(),
        true,
        None,
        0,
    );
    assert_eq!(choice.finish_reason, ir::FinishReason::ContentFilter);
    assert_eq!(
        choice.guard_stop,
        Some(atlas_tool_guard::GUARD_STOP),
        "{choice:?}"
    );
}

#[test]
fn tool_guard_explicit_off_restores_call_delivery() {
    let (mut c, mut s) = setup(false);
    Arc::get_mut(&mut c.state).unwrap().tool_guard.enabled = false;
    let out = dispatch(&mut s, &c, &xml(false, "http://169.254.169.254/"));
    assert!(
        out.iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
    );
    assert!(!s.cancel_flag.load(Ordering::Acquire));
}

#[test]
fn tool_guard_rejects_fragmented_json_and_cancels_remaining_calls() {
    let (c, mut s) = setup(false);
    let mut out = vec![];
    handle_tool_call_start(&mut s, &c, "a".into(), "get_weather".into(), 0, &mut out);
    for fragment in ["{\"url\":\"http://169.", "254.169.254/\"}"] {
        handle_tool_call_args_fragment(&mut s, &c, fragment.into(), 0, &mut out);
        assert!(out.is_empty());
    }
    handle_tool_call_end(&mut s, &c, 0, &mut out);
    handle_tool_call_start(&mut s, &c, "b".into(), "get_weather".into(), 1, &mut out);
    handle_tool_call_delta(&mut s, &c, r#"{"city":"Oslo"}"#.into(), 1, &mut out);
    handle_tool_call_end(&mut s, &c, 1, &mut out);
    assert!(out.is_empty());
    assert_blocked(&finish(&mut s, &c));
}

#[test]
fn tool_guard_real_token_path_blocks_without_releasing_call_ids() {
    for glm in [false, true] {
        let a = super::app_for_model(
            if glm { "glm5_next" } else { "qwen4_exp" },
            if glm {
                tool_parser::ToolCallFormat::PoolsideV1
            } else {
                tool_parser::ToolCallFormat::Qwen3Coder
            },
        );
        let body = if glm {
            "get_url<arg_key>url</arg_key><arg_value>http://169.254.169.254/</arg_value></tool_call>"
        } else {
            "<function=get_url><parameter=url>http://169.254.169.254/</parameter></function></tool_call>"
        };
        let token = a.tokenizer.inner().token_to_id(body).unwrap();
        let (c, mut s, mut out) = stream_for_app(a, true, vec![END, TOOL, token]);
        out.extend(finish(&mut s, &c));
        assert_blocked(&out);
    }
}

#[test]
fn tool_guard_does_not_release_calls_stopped_by_loop_limits() {
    let (c, mut s) = setup(false);
    for idx in 0..9 {
        let mut out = vec![];
        handle_tool_call_start(
            &mut s,
            &c,
            format!("id{idx}"),
            "get_weather".into(),
            idx,
            &mut out,
        );
        handle_tool_call_args_fragment(
            &mut s,
            &c,
            format!(r#"{{"city":"city{idx}"}}"#),
            idx,
            &mut out,
        );
        handle_tool_call_end(&mut s, &c, idx, &mut out);
        if idx < 7 {
            assert!(
                out.iter()
                    .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
            );
        } else {
            assert!(out.is_empty());
            assert!(s.cancel_flag.load(Ordering::Acquire));
        }
    }
    assert!(s.buffered_tool_chunks.is_empty());
    assert_ne!(s.guard_stop, Some(atlas_tool_guard::GUARD_STOP));
}

#[test]
fn tool_guard_document_examples_remain_callable_in_both_formats() {
    for glm in [false, true] {
        let (c, mut s) = setup(glm);
        let text = xml(glm, "http://169.254.169.254/")
            .replace("get_url", "write_document")
            .replace(">url<", ">content<")
            .replace("parameter=url", "parameter=content");
        let out = dispatch(&mut s, &c, &text);
        assert!(
            out.iter()
                .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. })),
            "{out:?}"
        );
        assert!(!s.cancel_flag.load(Ordering::Acquire));
    }
}

/// `--tool-guard stream`: the guard checks each call at its end, but the
/// call's header and arguments go to the client as the model writes them.
fn streaming_guard(glm: bool) -> (StreamCtx, StreamState) {
    let (mut c, s) = setup(glm);
    Arc::get_mut(&mut c.state).unwrap().tool_guard.stream = true;
    (c, s)
}
fn started_call(s: &mut StreamState, c: &StreamCtx, args: &str) -> Vec<ir::StreamDelta> {
    let mut out = vec![];
    handle_tool_call_start(s, c, "id".into(), "get_url".into(), 0, &mut out);
    handle_tool_call_args_fragment(s, c, args.into(), 0, &mut out);
    out
}
fn sent_call_bytes(out: &[ir::StreamDelta]) -> bool {
    out.iter()
        .any(|d| matches!(d, ir::StreamDelta::ToolCallStart { .. }))
        && out
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallArgs { .. }))
}
fn assert_blocked_after_streaming(out: &[ir::StreamDelta]) {
    let refusals: Vec<_> = out
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Content { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(refusals, [atlas_tool_guard::STREAMED_REFUSAL], "{out:?}");
    assert!(
        out.iter().any(|d| matches!(
            d,
            ir::StreamDelta::Finish {
                reason: ir::FinishReason::ContentFilter,
                guard_stop: Some(atlas_tool_guard::GUARD_STOP),
                ..
            }
        )),
        "{out:?}"
    );
}

#[test]
fn a_streaming_guard_sends_an_allowed_call_as_it_is_written() {
    let (c, mut s) = streaming_guard(false);
    assert!(sent_call_bytes(&started_call(&mut s, &c, r#"{"url": "#)));
    let mut out = vec![];
    handle_tool_call_args_fragment(&mut s, &c, r#""https://example.com/"}"#.into(), 0, &mut out);
    handle_tool_call_end(&mut s, &c, 0, &mut out);
    assert!(
        out.iter()
            .any(|d| matches!(d, ir::StreamDelta::ToolCallArgs { .. })),
        "{out:?}"
    );
    assert!(!s.cancel_flag.load(Ordering::Acquire));
    let done = finish(&mut s, &c);
    assert!(
        done.iter().any(|d| matches!(
            d,
            ir::StreamDelta::Finish {
                reason: ir::FinishReason::ToolCalls,
                guard_stop: None,
                ..
            }
        )),
        "{done:?}"
    );
}

#[test]
fn a_streaming_guard_still_blocks_a_call_at_its_end() {
    let (c, mut s) = streaming_guard(false);
    let mut out = started_call(&mut s, &c, r#"{"url": "http://169.254.169.254/latest/"}"#);
    assert!(sent_call_bytes(&out));
    handle_tool_call_end(&mut s, &c, 0, &mut out);
    assert!(s.cancel_flag.load(Ordering::Acquire));
    assert_blocked_after_streaming(&finish(&mut s, &c));
}

#[test]
fn a_streaming_guard_blocks_a_call_the_response_cuts_off() {
    let (c, mut s) = streaming_guard(false);
    assert!(sent_call_bytes(&started_call(
        &mut s,
        &c,
        r#"{"url": "https://exa"#
    )));
    assert_blocked_after_streaming(&finish(&mut s, &c));
}

#[test]
fn a_streaming_guard_blocks_an_oversized_call_mid_stream() {
    let (mut c, mut s) = streaming_guard(false);
    Arc::get_mut(&mut c.state).unwrap().tool_guard.max_bytes = 80;
    assert!(sent_call_bytes(&started_call(&mut s, &c, r#"{"url": ""#)));
    let mut out = vec![];
    handle_tool_call_args_fragment(&mut s, &c, "x".repeat(100), 0, &mut out);
    assert!(out.is_empty(), "{out:?}");
    assert!(s.cancel_flag.load(Ordering::Acquire));
    assert_blocked_after_streaming(&finish(&mut s, &c));
}

#[test]
fn a_streaming_guard_still_holds_poolside_calls() {
    // Poolside's detector holds a call until `</tool_call>` whatever the
    // guard says, so nothing of a blocked call reached the client.
    let (c, mut s) = streaming_guard(true);
    let mut out = started_call(&mut s, &c, r#"{"url": "http://169.254.169.254/"}"#);
    assert!(out.is_empty(), "{out:?}");
    handle_tool_call_end(&mut s, &c, 0, &mut out);
    assert!(out.is_empty(), "{out:?}");
    assert_blocked(&finish(&mut s, &c));
}
