// SPDX-License-Identifier: AGPL-3.0-only

//! A response cut off at the output-token budget must surface as
//! `status: "incomplete"` with `incomplete_details.reason:
//! "max_output_tokens"` on both `/v1/responses` encoders, not as a
//! completed answer.

use std::sync::Arc;

use axum::response::{IntoResponse, Sse};
use serde_json::Value;

use super::responses_stream_finalize::{FinalizeCtx, finalize_responses_stream};
use super::responses_translate::translate_chat_response_to_responses;
use crate::api::chat::ChatOutcome;
use crate::api::test_app_state;
use crate::ir;
use crate::openai::{ResponsesContentPart, ResponsesOutputItem};

fn tokenizer() -> crate::tokenizer::ChatTokenizer {
    let dir = tempfile::tempdir().unwrap();
    let vocab = [("[UNK]".to_string(), 0)].into_iter().collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    tokenizers::Tokenizer::new(model)
        .save(dir.path().join("tokenizer.json"), false)
        .unwrap();
    crate::tokenizer::ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        true,
        "glm5_next",
        "",
        None,
        false,
        false,
    )
    .unwrap()
}

fn chat_response(finish_reason: ir::FinishReason) -> ir::ChatResponse {
    ir::ChatResponse {
        id: "test".into(),
        model: "fixture".into(),
        created: 0,
        choices: vec![ir::Choice {
            index: 0,
            content: Some("The answer is".into()),
            reasoning: None,
            tool_calls: vec![],
            refusal: None,
            finish_reason,
            matched_stop: None,
            logprobs: None,
            guard_stop: None,
        }],
        usage: ir::Usage {
            prompt_tokens: 3,
            completion_tokens: 4,
            ..Default::default()
        },
    }
}

async fn blocking_body(finish_reason: ir::FinishReason) -> Value {
    let outcome = ChatOutcome::Blocking(Box::new(chat_response(finish_reason)));
    let resp = translate_chat_response_to_responses(outcome, None, None, vec![], false, None).await;
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&bytes).unwrap()
}

fn large_turn_items() -> Vec<Value> {
    (0..21)
        .map(|i| {
            serde_json::json!({
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": format!("message {i}")}],
            })
        })
        .collect()
}

fn assert_large_turn_persisted(items: &[Value]) {
    assert_eq!(items.len(), 22);
    for (i, item) in items[..21].iter().enumerate() {
        assert_eq!(item["role"], "user");
        assert_eq!(item["content"][0]["text"], format!("message {i}"));
    }
    assert_eq!(items[21]["role"], "assistant");
    assert_eq!(items[21]["content"][0]["text"], "The answer is");
}

#[tokio::test]
async fn blocking_response_persists_large_conversation_turn() {
    let state = test_app_state::app_state(tokenizer());
    let store = state.conversation_store;
    let id = store.create(vec![], Default::default());
    let req = serde_json::from_value(serde_json::json!({
        "model": "fixture", "input": large_turn_items(), "conversation": id,
    }))
    .unwrap();
    let input = crate::openai::lower_responses_to_chat(req, |_| None)
        .unwrap()
        .messages;
    let outcome = ChatOutcome::Blocking(Box::new(chat_response(ir::FinishReason::Stop)));
    let resp = translate_chat_response_to_responses(
        outcome,
        None,
        None,
        input,
        false,
        Some((store.clone(), id.clone())),
    )
    .await;
    assert_eq!(resp.status(), axum::http::StatusCode::OK);
    assert_large_turn_persisted(&store.get(&id).unwrap().items);
}

#[tokio::test]
async fn streaming_response_persists_large_conversation_turn() {
    let state = Arc::new(test_app_state::app_state(tokenizer()));
    let id = state.conversation_store.create(vec![], Default::default());
    let (tx, _rx) = tokio::sync::mpsc::channel(1);
    let ctx = FinalizeCtx {
        seq: 0,
        completed_items: vec![ResponsesOutputItem::Message {
            id: "msg_test".into(),
            status: "completed",
            role: "assistant",
            content: vec![ResponsesContentPart::OutputText {
                text: "The answer is".into(),
                annotations: None,
            }],
        }],
        final_usage: None,
        finish_reason: "stop".into(),
        refusal_text: None,
        message_item_id: "msg_test".into(),
        output_index: 0,
        resp_id: "resp_test".into(),
        created_at: 0,
        model: "fixture".into(),
        metadata_for_done: None,
        store_flag: false,
        input_messages: vec![],
        conversation_id: Some(id.clone()),
        conv_new_user_items: large_turn_items(),
    };
    finalize_responses_stream(&tx, state.clone(), ctx).await;
    assert_large_turn_persisted(&state.conversation_store.get(&id).unwrap().items);
}

/// Runs the stream finalizer and returns the terminal SSE frame as
/// `(event name, data)`.
async fn stream_terminal(finish_reason: &str) -> (String, Value) {
    let (tx, rx) = tokio::sync::mpsc::channel(16);
    let state = Arc::new(test_app_state::app_state(tokenizer()));
    let ctx = FinalizeCtx {
        seq: 0,
        completed_items: vec![],
        final_usage: None,
        finish_reason: finish_reason.into(),
        refusal_text: None,
        message_item_id: "msg_test".into(),
        output_index: 0,
        resp_id: "resp_test".into(),
        created_at: 0,
        model: "fixture".into(),
        metadata_for_done: None,
        store_flag: false,
        input_messages: vec![],
        conversation_id: None,
        conv_new_user_items: vec![],
    };
    finalize_responses_stream(&tx, state, ctx).await;
    drop(tx);
    let sse = Sse::new(tokio_stream::wrappers::ReceiverStream::new(rx)).into_response();
    let bytes = axum::body::to_bytes(sse.into_body(), usize::MAX)
        .await
        .unwrap();
    let text = String::from_utf8(bytes.to_vec()).unwrap();
    let frame = text.trim_end().rsplit("\n\n").next().unwrap();
    let field = |name: &str| {
        frame
            .lines()
            .find_map(|l| l.strip_prefix(name))
            .unwrap()
            .to_string()
    };
    (
        field("event: "),
        serde_json::from_str(&field("data: ")).unwrap(),
    )
}

#[tokio::test]
async fn blocking_response_cut_at_token_limit_is_incomplete() {
    let body = blocking_body(ir::FinishReason::Length).await;
    assert_eq!(body["status"], "incomplete");
    assert_eq!(body["incomplete_details"]["reason"], "max_output_tokens");
}

#[tokio::test]
async fn blocking_response_cut_at_timeout_is_incomplete() {
    let body = blocking_body(ir::FinishReason::Other(ir::FINISH_REASON_TIMEOUT.into())).await;
    assert_eq!(body["status"], "incomplete");
    assert_eq!(body["incomplete_details"]["reason"], "timeout");
}

#[tokio::test]
async fn stream_cut_at_timeout_ends_with_response_incomplete() {
    let (event, data) = stream_terminal(ir::FINISH_REASON_TIMEOUT).await;
    assert_eq!(event, "response.incomplete");
    assert_eq!(data["type"], "response.incomplete");
    assert_eq!(data["response"]["status"], "incomplete");
    assert_eq!(data["response"]["incomplete_details"]["reason"], "timeout");
}

#[tokio::test]
async fn blocking_response_that_stops_is_completed() {
    let body = blocking_body(ir::FinishReason::Stop).await;
    assert_eq!(body["status"], "completed");
    assert!(body.get("incomplete_details").is_none());
}

#[tokio::test]
async fn stream_cut_at_token_limit_ends_with_response_incomplete() {
    let (event, data) = stream_terminal("length").await;
    assert_eq!(event, "response.incomplete");
    assert_eq!(data["type"], "response.incomplete");
    assert_eq!(data["response"]["status"], "incomplete");
    assert_eq!(
        data["response"]["incomplete_details"]["reason"],
        "max_output_tokens"
    );
}

#[tokio::test]
async fn stream_that_stops_ends_with_response_completed() {
    let (event, data) = stream_terminal("stop").await;
    assert_eq!(event, "response.completed");
    assert_eq!(data["response"]["status"], "completed");
    assert!(data["response"].get("incomplete_details").is_none());
}
