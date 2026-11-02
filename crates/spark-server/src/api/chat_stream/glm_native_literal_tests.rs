// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn glm_native_literal_opener_stays_reasoning_and_next_real_call_still_binds() {
    let literal = vec![1, BACKTICK, TOOL, ORDINARY, BACKTICK];
    let (stream, deltas) = stream_tokens(true, true, literal.clone());
    assert!(!stream.thinking_done, "model=true; {deltas:?}");
    assert!(!stream.cancel_flag.load(Ordering::Acquire));
    let reasoning: String = deltas
        .iter()
        .filter_map(|delta| match delta {
            ir::StreamDelta::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert!(reasoning.contains("<tool_call>"), "model=true; {deltas:?}");
    assert!(
        !deltas
            .iter()
            .any(|delta| matches!(delta, ir::StreamDelta::ToolCallStart { .. }))
    );

    let mut with_call = literal.clone();
    with_call.extend_from_slice(CALL);
    let (stream, deltas) = stream_tokens(true, true, with_call);
    assert!(stream.thinking_done, "model=true; {deltas:?}");
    assert_eq!(
        deltas
            .iter()
            .filter(|delta| matches!(delta, ir::StreamDelta::ToolCallStart { .. }))
            .count(),
        1,
        "model=true; {deltas:?}"
    );
}

#[test]
fn glm_native_literal_opener_is_preserved_by_blocking_decode() {
    let state = app(true);
    let response = response(vec![1, BACKTICK, TOOL, ORDINARY, BACKTICK]);
    let (reasoning, content) =
        crate::api::chat_blocking::decode_response_text(&state, &response, true, true, false);
    let combined = format!("{}{}", reasoning.as_deref().unwrap_or_default(), content);
    assert!(combined.contains("<tool_call>"), "combined={combined:?}");
    let choice = crate::api::chat_blocking_choice::build_choice_message(
        &state,
        &request(),
        &response,
        reasoning,
        content,
        true,
        None,
        0,
    );
    assert!(
        choice.tool_calls.is_empty(),
        "literal must not become a call"
    );
}
