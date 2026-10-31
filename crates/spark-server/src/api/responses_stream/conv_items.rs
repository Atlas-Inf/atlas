// SPDX-License-Identifier: AGPL-3.0-only
//! Conversation-linked user-item capture for `responses_endpoint_stream`
//! (extracted; identical content).

use super::*;
use crate::openai::ChatMessage;

/// Capture the conversation-linked user items (the new turn's user input)
/// before `chat_req` is handed off. Used to append back to the
/// conversation after the stream finishes.
pub(super) fn conv_new_user_items(
    state_arc: &Arc<AppState>,
    conversation_id: Option<&String>,
    input_messages: &[ChatMessage],
) -> Vec<serde_json::Value> {
    let Some(cid) = conversation_id else {
        return Vec::new();
    };
    let prior = state_arc
        .conversation_store
        .get(cid)
        .map(|s| s.items.len())
        .unwrap_or(0);
    input_messages
        .iter()
        .skip(prior)
        .map(|m| {
            serde_json::json!({
                "type": "message",
                "role": m.role,
                "content": [{"type": "input_text", "text": m.content.text}],
            })
        })
        .collect()
}
