// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::ir::Usage;
use std::sync::{Arc, atomic::AtomicBool};

fn state() -> StreamState {
    let mut state = StreamState::new(false, false, Arc::new(AtomicBool::new(false)), vec![]);
    state.tag_scan_buf = "The lantern".into();
    state.reasoning_tag_scan_buf = "buffered reasoning".into();
    state.pending_token_ids = vec![7, 8];
    state
}

fn usage() -> Usage {
    Usage {
        prompt_tokens: 28,
        completion_tokens: 63,
        cached_prompt_tokens: 4,
        reasoning_tokens: 2,
        accepted_prediction_tokens: 3,
        time_to_first_token_ms: 321.0,
        response_tokens_per_second: 22.0,
    }
}

fn flush_sanitizer(state: &mut StreamState) -> DeltaVec {
    let text = crate::api::stream_guards::flush_content_sanitizer(
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &crate::tool_parser::LeakMarkers::EMPTY,
    );
    vec![StreamDelta::Content {
        text,
        token_ids: state.take_ids_if(true),
    }]
}

fn assert_finish_only(deltas: &[StreamDelta], expected: &str) {
    assert!(
        matches!(deltas, [StreamDelta::Finish { reason, usage: actual, token_ids }]
            if reason.as_wire() == expected && *actual == usage() && token_ids.is_empty()),
        "guard must emit metadata/finish only, got {deltas:?}"
    );
}

#[test]
fn terminal_done_semantic_guard_does_not_flush_lantern_or_refusal() {
    let mut s = state();
    s.guard_stop = Some("simhash_semantic_loop");
    s.refusal_scan_buf = "I cannot help with that request.".into();
    let (deltas, reason) = finalize_done(&mut s, "stop", usage(), true, flush_sanitizer);
    assert_eq!(reason, "length");
    assert_finish_only(&deltas, "length");
    assert_eq!(
        s.tag_scan_buf, "The lantern",
        "guarded flush body must not run"
    );
    assert!(s.pending_token_ids.is_empty());
}

#[test]
fn terminal_done_tool_or_generic_guard_skips_buffered_fragments() {
    for tool_cap in [false, true] {
        let mut s = state();
        s.tool_loop_capped = tool_cap;
        s.stop_string_triggered = !tool_cap;
        s.detector = Some(crate::tool_parser::StreamingToolDetector::new_with_tools(
            vec![],
        ));
        s.detector
            .as_mut()
            .unwrap()
            .process("<tool_call><function=test>");
        let (deltas, _) = finalize_done(&mut s, "length", usage(), true, |_| {
            panic!("terminal guard must not flush the tool detector")
        });
        assert_finish_only(&deltas, "length");
    }
}

#[test]
fn terminal_done_guard_during_flush_discards_all_generated_delta_kinds_and_ids() {
    let mut s = state();
    let (deltas, _) = finalize_done(&mut s, "stop", usage(), true, |s| {
        let mut deltas = flush_sanitizer(s);
        s.guard_stop = Some("token_loop_watchdog");
        s.pending_token_ids.push(99);
        deltas.extend([
            StreamDelta::Reasoning {
                text: std::mem::take(&mut s.reasoning_tag_scan_buf),
                token_ids: vec![9],
            },
            StreamDelta::ToolCallStart {
                index: 0,
                id: "call-1".into(),
                name: "test".into(),
            },
            StreamDelta::ToolCallArgs {
                index: 0,
                fragment: "{\"late\":true}".into(),
                token_ids: vec![10],
            },
            StreamDelta::Refusal {
                text: "buffered refusal".into(),
            },
        ]);
        deltas
    });
    assert_finish_only(&deltas, "length");
    assert!(s.pending_token_ids.is_empty());
}

#[test]
fn terminal_done_normal_eos_and_explicit_stop_keep_safe_sanitizer_prefix() {
    for explicit in [false, true] {
        let mut s = state();
        s.stop_string_triggered = explicit;
        s.stop_string_matched = explicit;
        s.tag_scan_buf = if explicit {
            "safe pre-stop prefix"
        } else {
            "The lantern"
        }
        .into();
        let expected = s.tag_scan_buf.clone();
        let (deltas, reason) = finalize_done(&mut s, "stop", usage(), true, flush_sanitizer);
        assert_eq!(reason, "stop");
        assert!(
            matches!(&deltas[..], [StreamDelta::Content { text, token_ids }, StreamDelta::Finish { usage: actual, .. }]
            if text == &expected && token_ids == &[7,8] && *actual == usage())
        );
    }
}

#[test]
fn terminal_done_keeps_existing_finish_precedence_and_tool_history() {
    for (scheduler, tool_cap, salvaged, matched, expected) in [
        ("timeout", true, true, true, "timeout"),
        ("stop", true, true, true, "length"),
        ("stop", false, true, false, "tool_calls"),
        ("length", false, false, true, "stop"),
    ] {
        let mut s = state();
        s.guard_stop = Some("simhash_semantic_loop");
        s.tool_loop_capped = tool_cap;
        s.salvaged_tool_call = salvaged;
        s.stop_string_matched = matched;
        let (deltas, reason) = finalize_done(&mut s, scheduler, usage(), true, flush_sanitizer);
        assert_eq!(reason, expected);
        assert_eq!(s.salvaged_tool_call, salvaged);
        assert_finish_only(&deltas, expected);
    }
}
