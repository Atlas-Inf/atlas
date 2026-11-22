// SPDX-License-Identifier: AGPL-3.0-only

use std::sync::Arc;

use axum::{Json, http::StatusCode};
use serde_json::{Value, json};

use super::{InferenceRequest, InferenceResponse, StreamEvent, responses_endpoint, test_app_state};
use crate::main_modules::model_host::CurrentModel;
use crate::response_store::StoredKind;

fn tokenizer() -> crate::tokenizer::ChatTokenizer {
    let dir = tempfile::tempdir().unwrap();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab([("[UNK]".to_string(), 0)].into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(model);
    inner
        .add_special_tokens(["X", "Y", "POLICY"].map(|s| tokenizers::AddedToken::from(s, true)))
        .unwrap();
    inner
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

async fn turn(state: Arc<crate::AppState>, body: Value) -> String {
    let streaming = body["stream"].as_bool().unwrap();
    let response = responses_endpoint(
        CurrentModel(state),
        Default::default(),
        Ok(Json(serde_json::from_value(body).unwrap())),
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    let response: Value = if streaming {
        body.lines()
            .filter_map(|l| l.strip_prefix("data: "))
            .filter_map(|l| serde_json::from_str::<Value>(l).ok())
            .find(|v| v["type"] == "response.completed")
            .unwrap()["response"]
            .clone()
    } else {
        serde_json::from_str(&body).unwrap()
    };
    response["id"].as_str().unwrap().to_string()
}

async fn check_resume(stream: bool, conversation: bool) {
    for instructions in [Some("Y"), None] {
        let (request_tx, mut rx) = tokio::sync::mpsc::channel(2);
        let state = Arc::new(crate::AppState {
            request_tx,
            ..test_app_state::app_state(tokenizer())
        });
        let conversation_id = conversation.then(|| {
            state.conversation_store.create(
                vec![json!({"type": "message", "role": "system", "content": "POLICY"})],
                Default::default(),
            )
        });
        let scheduler = tokio::spawn(async move {
            for turn in 0..2 {
                let request = rx.recv().await.unwrap();
                let (InferenceRequest::Blocking { prompt_tokens, .. }
                | InferenceRequest::Streaming { prompt_tokens, .. }) = &request;
                assert_eq!(
                    prompt_tokens.iter().filter(|&&t| t == 1).count(),
                    usize::from(turn == 0 && !conversation),
                    "X in turn {turn}"
                );
                assert_eq!(
                    prompt_tokens.iter().filter(|&&t| t == 2).count(),
                    usize::from(turn == 1 && instructions.is_some()),
                    "Y in turn {turn}"
                );
                assert_eq!(
                    prompt_tokens.contains(&3),
                    conversation && (turn == 0 || instructions.is_none()),
                    "explicit system message lost"
                );
                match request {
                    InferenceRequest::Blocking { response_tx, .. } => {
                        assert!(!stream);
                        assert!(
                            response_tx
                                .send(Ok(InferenceResponse {
                                    output_tokens: vec![0],
                                    finish_reason: "stop".into(),
                                    time_to_first_token_ms: 0.0,
                                    decode_time_ms: 0.0,
                                    logprobs: vec![],
                                    reasoning_tokens: 0,
                                    cached_prompt_tokens: 0,
                                    accepted_prediction_tokens: 0,
                                    prompt_logprobs: vec![],
                                    guard_stop: None,
                                }))
                                .is_ok()
                        );
                    }
                    InferenceRequest::Streaming { token_tx, .. } => {
                        assert!(stream);
                        token_tx.send(StreamEvent::Token(0)).await.unwrap();
                        token_tx
                            .send(StreamEvent::Done {
                                finish_reason: "stop".into(),
                                prompt_tokens: 1,
                                completion_tokens: 1,
                                time_to_first_token_ms: 0.0,
                                decode_time_ms: 0.0,
                                reasoning_tokens: 0,
                                cached_prompt_tokens: 0,
                                accepted_prediction_tokens: 0,
                                guard_stop: None,
                            })
                            .await
                            .unwrap();
                    }
                }
            }
        });
        let first = turn(
            state.clone(),
            json!({
                "model": "fixture", "input": "first", "instructions": "X",
                "store": true, "stream": stream, "max_output_tokens": 1,
                "conversation": conversation_id,
            }),
        )
        .await;
        let stored = state
            .response_store
            .get(&first, StoredKind::Response)
            .unwrap();
        assert!(
            !stored.messages.iter().any(|m| m.content.text == "X"),
            "stored A retained instructions X"
        );
        let systems: Vec<_> = stored
            .messages
            .iter()
            .filter(|m| m.role == "system")
            .map(|m| m.content.text.as_str())
            .collect();
        assert_eq!(systems, if conversation { vec!["POLICY"] } else { vec![] });
        let second = turn(
            state.clone(),
            json!({
                "model": "fixture", "input": "second", "instructions": instructions,
                "previous_response_id": first, "store": true, "stream": stream,
                "max_output_tokens": 1,
            }),
        )
        .await;
        let stored = state
            .response_store
            .get(&second, StoredKind::Response)
            .unwrap();
        assert!(
            !stored
                .messages
                .iter()
                .any(|m| matches!(m.content.text.as_str(), "X" | "Y"))
        );
        assert_eq!(
            stored.messages.iter().filter(|m| m.role == "user").count(),
            2
        );
        scheduler.await.unwrap();
    }
}

#[tokio::test]
async fn blocking_previous_instructions() {
    check_resume(false, false).await;
}

#[tokio::test]
async fn streaming_previous_instructions() {
    check_resume(true, false).await;
}

#[tokio::test]
async fn blocking_conversation_instructions() {
    check_resume(false, true).await;
}

#[tokio::test]
async fn streaming_conversation_instructions() {
    check_resume(true, true).await;
}

async fn check_filtered_conversation_prefix(stream: bool) {
    let (request_tx, mut rx) = tokio::sync::mpsc::channel(1);
    let state = Arc::new(crate::AppState {
        request_tx,
        ..test_app_state::app_state(tokenizer())
    });
    let conversation_id = state.conversation_store.create(
        vec![json!({"type": "reasoning", "summary": []})],
        Default::default(),
    );
    let scheduler = tokio::spawn(async move {
        match rx.recv().await.unwrap() {
            InferenceRequest::Blocking { response_tx, .. } => {
                assert!(
                    response_tx
                        .send(Ok(InferenceResponse {
                            output_tokens: vec![0],
                            finish_reason: "stop".into(),
                            time_to_first_token_ms: 0.0,
                            decode_time_ms: 0.0,
                            logprobs: vec![],
                            reasoning_tokens: 0,
                            cached_prompt_tokens: 0,
                            accepted_prediction_tokens: 0,
                            prompt_logprobs: vec![],
                            guard_stop: None,
                        }))
                        .is_ok()
                );
            }
            InferenceRequest::Streaming { token_tx, .. } => {
                token_tx.send(StreamEvent::Token(0)).await.unwrap();
                token_tx
                    .send(StreamEvent::Done {
                        finish_reason: "stop".into(),
                        prompt_tokens: 1,
                        completion_tokens: 1,
                        time_to_first_token_ms: 0.0,
                        decode_time_ms: 0.0,
                        reasoning_tokens: 0,
                        cached_prompt_tokens: 0,
                        accepted_prediction_tokens: 0,
                        guard_stop: None,
                    })
                    .await
                    .unwrap();
            }
        }
    });
    turn(
        state.clone(),
        json!({
            "model": "fixture", "input": "new user input", "instructions": "X",
            "conversation": conversation_id, "store": true, "stream": stream,
            "max_output_tokens": 1,
        }),
    )
    .await;
    scheduler.await.unwrap();
    let conversation = state.conversation_store.get(&conversation_id).unwrap();
    let appended = &conversation.items[1..];
    assert!(
        appended
            .iter()
            .any(|item| item["role"] == "user" && item["content"][0]["text"] == "new user input"),
        "new user input missing from conversation append: {appended:?}"
    );
    assert!(!appended.iter().any(|item| item["role"] == "system"));
}

#[tokio::test]
async fn blocking_filtered_conversation_prefix() {
    check_filtered_conversation_prefix(false).await;
}

#[tokio::test]
async fn streaming_filtered_conversation_prefix() {
    check_filtered_conversation_prefix(true).await;
}
