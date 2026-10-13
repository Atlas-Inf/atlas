// SPDX-License-Identifier: AGPL-3.0-only
//! Actual Token -> Done handlers preserve sanitized reasoning without duplication.

use super::*;
use crate::api::chat_stream::handle_done::handle_done;

fn done(stream: &mut StreamState, ctx: &StreamCtx, reason: &str) -> Vec<ir::StreamDelta> {
    handle_done(stream, ctx, reason.into(), 2, 1.0, 1.0, 0, 0, 0)
}

#[test]
fn reasoning_done_matches_blocking_when_eos_has_no_think_close() {
    for reason in ["stop", "length"] {
        let raw = vec![14]; // The real live failure was one token "42", then EOS.
        let app = app(true);
        let (reasoning, content) = crate::api::chat_blocking::decode_response_text(
            &app,
            &response(raw.clone()),
            true,
            false,
        );
        assert_eq!(reasoning.as_deref(), Some("42"));
        assert!(content.is_empty());
        let (ctx, mut stream, mut deltas) = stream_tokens_with_context(true, false, raw);
        assert!(
            matches!(deltas.as_slice(), [ir::StreamDelta::Reasoning { text, .. }] if text == "42")
        );
        assert!(stream.reasoning_tag_scan_buf.is_empty());
        deltas.extend(done(&mut stream, &ctx, reason));
        assert!(
            matches!(deltas.as_slice(), [
            ir::StreamDelta::Reasoning { text, token_ids },
            ir::StreamDelta::Finish { reason: finish, .. }
        ] if text == "42" && token_ids == &[14] && finish.as_wire() == reason),
            "{deltas:?}"
        );
        assert!(stream.reasoning_tag_scan_buf.is_empty());
    }
}

#[test]
fn reasoning_done_does_not_duplicate_a_closed_thinking_tail() {
    let (ctx, mut stream, mut deltas) = stream_tokens_with_context(true, false, vec![14, END]);
    deltas.extend(done(&mut stream, &ctx, "stop"));
    let reasoning: String = deltas
        .iter()
        .filter_map(|d| match d {
            ir::StreamDelta::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(reasoning, "42");
    assert!(
        !deltas
            .iter()
            .any(|d| matches!(d, ir::StreamDelta::Content { .. }))
    );
}

#[test]
fn reasoning_done_preserves_rejected_output_and_leak_suppression() {
    for guard in [false, true] {
        let (ctx, mut stream, deltas) = stream_tokens_with_context(true, false, vec![]);
        assert!(deltas.is_empty());
        if guard {
            stream.guard_stop = Some("simhash_semantic_loop");
        } else {
            stream.reasoning_suppressing_leak = true;
        }
        assert!(handle_token(&mut stream, &ctx, 14).is_empty());
        let deltas = done(&mut stream, &ctx, "stop");
        assert!(
            matches!(deltas.as_slice(), [ir::StreamDelta::Finish { .. }]),
            "{deltas:?}"
        );
    }
}

#[test]
fn sanitizer_prefix_first_content_token_reaches_neutral_delta_immediately() {
    let (ctx, mut stream, mut deltas) = stream_tokens_with_context(true, false, vec![END, 14]);
    assert!(matches!(deltas.as_slice(), [ir::StreamDelta::Content { text, .. }] if text == "42"));
    assert!(stream.tag_scan_buf.is_empty());
    deltas.extend(done(&mut stream, &ctx, "stop"));
    assert!(
        matches!(deltas.as_slice(), [ir::StreamDelta::Content { text, .. }, ir::StreamDelta::Finish { .. }] if text == "42")
    );
}
