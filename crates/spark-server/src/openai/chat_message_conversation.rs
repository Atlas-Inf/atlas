// SPDX-License-Identifier: AGPL-3.0-only

use super::IncomingMessage;
use serde_json::{Value, json};

impl IncomingMessage {
    pub fn from_conversation_items(items: &[Value]) -> Vec<Self> {
        let mut messages = Vec::new();
        for message in items.iter().filter_map(Self::from_conversation_item) {
            Self::push_responses_message(&mut messages, message);
        }
        messages
    }

    pub(crate) fn push_responses_message(messages: &mut Vec<Self>, mut message: Self) {
        if message.role == "assistant"
            && message.tool_calls.is_some()
            && let Some(previous) = messages.last_mut().filter(|m| m.role == "assistant")
        {
            previous
                .tool_calls
                .get_or_insert_with(Vec::new)
                .extend(message.tool_calls.take().unwrap());
        } else {
            messages.push(message);
        }
    }

    pub fn to_conversation_items(&self) -> Vec<Value> {
        if self.role == "tool" {
            return vec![json!({
                "type": "function_call_output", "call_id": self.tool_call_id,
                "name": self.name, "output": self.content.text,
            })];
        }
        let mut items = Vec::new();
        if self.tool_calls.is_none()
            || !self.content.text.is_empty()
            || self.reasoning_content.is_some()
        {
            let mut item = json!({
                "type": "message", "role": self.role,
                "content": [{"type": "input_text", "text": self.content.text}],
            });
            if let Some(reasoning) = &self.reasoning_content {
                item["reasoning_content"] = json!(reasoning);
            }
            items.push(item);
        }
        for call in self.tool_calls.iter().flatten() {
            items.push(json!({
                "type": "function_call", "call_id": call.id,
                "name": call.function.name, "arguments": call.function.arguments,
            }));
        }
        items
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversation_assistant_turn_merges_text_and_parallel_calls() {
        let items = vec![
            json!({"role":"user", "content":"look up"}),
            json!({"role":"assistant", "content":"Checking", "reasoning_content":"plan"}),
            json!({"type":"function_call", "call_id":"a", "name":"lookup", "arguments":"{}"}),
            json!({"type":"function_call", "call_id":"b", "name":"lookup", "arguments":"{}"}),
            json!({"type":"function_call_output", "call_id":"a", "output":"found"}),
        ];
        let replay = IncomingMessage::from_conversation_items(&items);
        assert_assistant_turn(&replay);
        assert_eq!(replay[1].reasoning_content.as_deref(), Some("plan"));
        let request: crate::openai::ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture", "input":items,
        }))
        .unwrap();
        let input = crate::openai::lower_responses_to_chat(request, |_| None).unwrap();
        assert_assistant_turn(&input.messages);
    }

    fn assert_assistant_turn(messages: &[IncomingMessage]) {
        assert_eq!(
            messages.len(),
            3,
            "one assistant turn for text and parallel calls"
        );
        assert_eq!(messages[1].content.text, "Checking");
        let calls = messages[1].tool_calls.as_ref().unwrap();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].id.as_deref(), Some("a"));
        assert_eq!(calls[1].id.as_deref(), Some("b"));
        assert_eq!(messages[2].role, "tool");
        assert_eq!(messages[2].tool_call_id.as_deref(), Some("a"));
    }

    #[test]
    fn conversation_parallel_calls_without_text_merge() {
        let items = vec![
            json!({"type":"function_call", "call_id":"a", "name":"lookup"}),
            json!({"type":"function_call", "call_id":"b", "name":"lookup"}),
        ];
        let replay = IncomingMessage::from_conversation_items(&items);
        assert_eq!(replay.len(), 1);
        assert_eq!(replay[0].tool_calls.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn conversation_input_parallel_calls_merge() {
        let request: crate::openai::ResponsesRequest = serde_json::from_value(json!({
            "model":"fixture", "input":[
                {"role":"user", "content":"look up"},
                {"role":"assistant", "content":"Checking"},
                {"type":"function_call", "call_id":"a", "name":"lookup"},
                {"type":"function_call", "call_id":"b", "name":"lookup"},
                {"type":"function_call_output", "call_id":"a", "output":"found"},
            ],
        }))
        .unwrap();
        let input = crate::openai::lower_responses_to_chat(request, |_| None).unwrap();
        assert_assistant_turn(&input.messages);
    }

    #[test]
    fn conversation_tool_call_replay() {
        for item in [
            json!({"type":"function_call", "call_id":"call_1", "name":"lookup", "arguments":"{}"}),
            json!({"type":"function_call_output", "call_id":"call_1", "output":"found", "name":"lookup"}),
            json!({"type":"function_call_output", "call_id":"call_1", "output":""}),
        ] {
            let expected = IncomingMessage::from_responses_input_item(&item)
                .unwrap()
                .unwrap();
            let store = crate::conversation_store::ConversationStore::with_config(
                10,
                std::time::Duration::from_secs(60),
            );
            let id = store.create(expected.to_conversation_items(), Default::default());
            let stored = store.get(&id).unwrap();
            let actual = IncomingMessage::from_conversation_item(&stored.items[0])
                .expect("tool item must replay");
            assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
        }
    }

    #[test]
    fn conversation_tool_call_replay_invalid_items() {
        for item in [
            json!({"type":"function_call"}),
            json!({"type":"unknown"}),
            json!(null),
        ] {
            assert!(IncomingMessage::from_conversation_item(&item).is_none());
        }
    }
}
