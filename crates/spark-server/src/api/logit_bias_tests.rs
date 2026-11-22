// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::main_modules::model_host::{CurrentModel, ModelHost};
use serde_json::json;

async fn request(
    chat: bool,
    stream: bool,
    bias: serde_json::Value,
) -> (Response, Option<Vec<(u32, f32)>>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let state = Arc::new(AppState {
        request_tx: tx,
        ..crate::api::test_app_state::app_state(super::spec_policy_tests::tokenizer())
    });
    let mut body = json!({"model": "fixture", "max_tokens": 1, "stream": stream,
        "temperature": 0.7, "logit_bias": bias});
    let scheduler = tokio::spawn(async move {
        match rx.recv().await? {
            InferenceRequest::Blocking {
                logit_bias,
                response_tx,
                ..
            } => {
                let _ = response_tx.send(Ok(super::super::inference_types::InferenceResponse {
                    output_tokens: vec![1],
                    finish_reason: "length".into(),
                    time_to_first_token_ms: 0.0,
                    decode_time_ms: 0.0,
                    logprobs: vec![],
                    reasoning_tokens: 0,
                    cached_prompt_tokens: 0,
                    accepted_prediction_tokens: 0,
                    prompt_logprobs: vec![],
                    guard_stop: None,
                }));
                Some(logit_bias)
            }
            InferenceRequest::Streaming { logit_bias, .. } => Some(logit_bias),
        }
    });
    let response = if chat {
        body["messages"] = json!([{"role": "user", "content": "prompt"}]);
        let host = ModelHost::empty();
        host.publish(state);
        crate::api::chat_completions(
            axum::extract::State(Arc::new(host)),
            None,
            axum::http::HeaderMap::new(),
            serde_json::to_vec(&body).unwrap().into(),
        )
        .await
    } else {
        body["prompt"] = json!([1]);
        crate::api::completions::completions(
            CurrentModel(state),
            Ok(Json(serde_json::from_value(body).unwrap())),
        )
        .await
    };
    (response, scheduler.await.unwrap())
}

#[tokio::test]
async fn logit_bias_rejects_malformed_keys_before_dispatch() {
    for chat in [false, true] {
        for stream in [false, true] {
            for key in ["abc", "-1", "1.5", "", "4294967296"] {
                let (response, dispatched) =
                    request(chat, stream, json!({key: 2.0, "1": -0.5})).await;
                assert_eq!(
                    response.status(),
                    StatusCode::BAD_REQUEST,
                    "chat={chat}, stream={stream}, key={key:?}"
                );
                assert_eq!(dispatched, None);
                let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                    .await
                    .unwrap();
                let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
                assert_eq!(body["error"]["type"], "invalid_request_error");
                assert_eq!(body["error"]["param"], "logit_bias");
                let message = body["error"]["message"].as_str().unwrap();
                assert!(message.contains("logit_bias") && message.contains(key));
            }
        }
    }
}

#[tokio::test]
async fn logit_bias_preserves_valid_empty_and_absent_maps() {
    for chat in [false, true] {
        for stream in [false, true] {
            for bias in [
                json!({"0": -1.5, "1": 2.0, "4294967295": 0.5}),
                json!({}),
                json!(null),
            ] {
                let mut expected: Vec<(u32, f32)> = bias
                    .as_object()
                    .map(|map| {
                        map.iter()
                            .map(|(k, v)| (k.parse().unwrap(), v.as_f64().unwrap() as f32))
                            .collect()
                    })
                    .unwrap_or_default();
                let (response, dispatched) = request(chat, stream, bias).await;
                assert_eq!(
                    response.status(),
                    StatusCode::OK,
                    "chat={chat}, stream={stream}"
                );
                let mut actual = dispatched.unwrap();
                actual.sort_by_key(|&(id, _)| id);
                expected.sort_by_key(|&(id, _)| id);
                assert_eq!(actual, expected);
            }
        }
    }
}
