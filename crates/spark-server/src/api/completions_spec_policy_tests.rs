// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise both completion handlers through the real scheduler request
//! boundary: the request-level speculation policy.
//! Ported from Enntity/atlas f4d95a10 (AGPL-3.0-only).

use super::*;
use crate::api::inference_types::{InferenceRequest, InferenceResponse};
use crate::main_modules::model_host::CurrentModel;
use serde_json::json;

fn tokenizer() -> crate::tokenizer::ChatTokenizer {
    let dir = tempfile::tempdir().unwrap();
    let vocab = [
        "[UNK]", "answer", "<think>", "secret", "</think>", "STOP", "prompt",
    ]
    .iter()
    .enumerate()
    .map(|(i, w)| (w.to_string(), i as u32))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut inner = tokenizers::Tokenizer::new(model);
    inner.with_decoder(Some(tokenizers::decoders::fuse::Fuse::new()));
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

#[tokio::test]
async fn completion_disable_mtp_reaches_scheduler_for_blocking_and_streaming() {
    for stream in [false, true] {
        // Omitted is the default-on speculative behavior; true is an
        // explicit request-level opt-out and must reach both request kinds.
        for disabled in [false, true] {
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            let state = Arc::new(AppState {
                request_tx: tx,
                ..crate::api::test_app_state::app_state(tokenizer())
            });
            let mut body = json!({"model": "fixture", "prompt": [1],
                                  "max_tokens": 1, "stream": stream,
                                  "temperature": 0.7, "top_k": 20, "top_p": 0.95});
            if disabled {
                body["disable_mtp"] = json!(true);
            }
            let req: CompletionRequest = serde_json::from_value(body).unwrap();
            let scheduler = tokio::spawn(async move {
                let request: InferenceRequest = rx.recv().await.unwrap();
                assert_eq!(request.disable_mtp(), disabled);
                match request {
                    InferenceRequest::Blocking {
                        temperature,
                        top_k,
                        top_p,
                        response_tx,
                        ..
                    } => {
                        assert!(!stream);
                        assert_eq!((temperature, top_k, top_p), (0.7, 20, 0.95));
                        let result = response_tx.send(Ok(InferenceResponse {
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
                        assert!(result.is_ok());
                    }
                    InferenceRequest::Streaming {
                        temperature,
                        top_k,
                        top_p,
                        ..
                    } => {
                        assert!(stream);
                        assert_eq!((temperature, top_k, top_p), (0.7, 20, 0.95));
                    }
                }
            });
            let response =
                crate::api::completions::completions(CurrentModel(state), Ok(axum::Json(req)))
                    .await;
            assert_eq!(response.status(), StatusCode::OK);
            scheduler.await.unwrap();
        }
    }
}
