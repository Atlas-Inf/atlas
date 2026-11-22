// SPDX-License-Identifier: AGPL-3.0-only
//! Conversation-linked user-item capture for `responses_endpoint_stream`
//! (extracted; identical content).

use crate::openai::IncomingMessage;

/// Capture the conversation-linked user items (the new turn's user input)
/// before `chat_req` is handed off. Used to append back to the
/// conversation after the stream finishes.
pub(super) fn conv_new_user_items(
    conversation_prefix_len: Option<usize>,
    input_messages: &[IncomingMessage],
) -> Vec<serde_json::Value> {
    let Some(prefix_len) = conversation_prefix_len else {
        return Vec::new();
    };
    input_messages
        .iter()
        .skip(prefix_len)
        .map(|m| {
            serde_json::json!({
                "type": "message",
                "role": m.role,
                "content": [{"type": "input_text", "text": m.content.text}],
            })
        })
        .collect()
}
