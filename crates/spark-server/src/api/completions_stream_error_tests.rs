// SPDX-License-Identifier: AGPL-3.0-only

//! Streaming `/v1/completions` error frames must be valid JSON whatever the
//! scheduler's error text contains.

use super::*;
use crate::api::inference_types::{InferenceRequest, StreamEvent};
use crate::main_modules::model_host::CurrentModel;
use serde_json::{Value, json};

/// Drive one streaming completion whose scheduler answers with `event`, and
/// return every SSE `data:` payload except the closing `[DONE]`.
async fn stream_payloads(event: StreamEvent) -> Vec<String> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let state = Arc::new(AppState {
        request_tx: tx,
        ..crate::api::test_app_state::app_state(super::spec_policy_tests::tokenizer())
    });
    let req: CompletionRequest = serde_json::from_value(json!({
        "model": "fixture", "prompt": [1], "max_tokens": 1, "stream": true
    }))
    .unwrap();
    let scheduler = tokio::spawn(async move {
        let Some(InferenceRequest::Streaming { token_tx, .. }) = rx.recv().await else {
            panic!("expected a streaming request");
        };
        token_tx.send(event).await.unwrap();
    });
    let response =
        crate::api::completions::completions(CurrentModel(state), Ok(axum::Json(req))).await;
    assert_eq!(response.status(), StatusCode::OK);
    scheduler.await.unwrap();
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    String::from_utf8(body.to_vec())
        .unwrap()
        .lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(str::to_owned)
        .collect()
}

#[tokio::test]
async fn stream_error_text_with_json_metacharacters_is_escaped() {
    let msg = "CUDA error: \"illegal address\"\nat C:\\kernels\\moe.cu";
    let events = [
        StreamEvent::Error(msg.into()),
        StreamEvent::InvalidRequest(msg.into()),
    ];
    for event in events {
        let payloads = stream_payloads(event).await;
        assert_eq!(payloads.len(), 1, "{payloads:?}");
        let frame: Value = serde_json::from_str(&payloads[0])
            .unwrap_or_else(|e| panic!("malformed error frame {:?}: {e}", payloads[0]));
        assert_eq!(frame, json!({ "error": msg }));
    }
}
