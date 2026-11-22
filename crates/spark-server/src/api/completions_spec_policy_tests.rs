// SPDX-License-Identifier: AGPL-3.0-only

//! Exercise both completion handlers through the real scheduler request
//! boundary: request-level speculation policy and streamed logprobs.
//! Ported from Enntity/atlas f4d95a10 (AGPL-3.0-only).

use super::*;
use crate::api::inference_types::{
    InferenceRequest, InferenceResponse, StreamEvent, TokenLogprobs,
};
use crate::main_modules::model_host::CurrentModel;
use serde_json::json;

pub(super) fn tokenizer() -> crate::tokenizer::ChatTokenizer {
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

async fn completion_stream_chunks(
    logprobs: Option<u8>,
    echo: bool,
    events: Vec<StreamEvent>,
) -> Vec<serde_json::Value> {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let state = Arc::new(AppState {
        request_tx: tx,
        ..crate::api::test_app_state::app_state(tokenizer())
    });
    let scheduler = tokio::spawn(async move {
        let InferenceRequest::Streaming {
            token_tx,
            top_logprobs,
            prompt_logprobs,
            ..
        } = rx.recv().await.unwrap()
        else {
            panic!("expected a streaming inference request");
        };
        assert_eq!(top_logprobs, logprobs);
        assert_eq!(prompt_logprobs, if echo { logprobs } else { None });
        for event in events {
            token_tx.send(event).await.unwrap();
        }
    });
    let req = serde_json::from_value(json!({
        "model": "fixture", "prompt": [6], "max_tokens": 3,
        "stream": true, "logprobs": logprobs, "echo": echo,
    }))
    .unwrap();
    let response =
        crate::api::completions::completions(CurrentModel(state), Ok(axum::Json(req))).await;
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    scheduler.await.unwrap();
    let body = String::from_utf8(bytes.to_vec()).unwrap();
    assert!(body.ends_with("data: [DONE]\n\n"));
    body.lines()
        .filter_map(|line| line.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test]
async fn completion_stream_preserves_generated_logprobs_and_offsets() {
    for echo in [false, true] {
        let mut events = Vec::new();
        if echo {
            events.push(StreamEvent::PromptLogprobs(vec![]));
        }
        for (token_id, logprob) in [(1, -0.5), (3, -1.0)] {
            events.push(StreamEvent::TokenWithLogprobs(
                token_id,
                TokenLogprobs {
                    token_id,
                    logprob,
                    top: vec![(1, -0.5), (3, -1.0)],
                },
            ));
        }
        let chunks = completion_stream_chunks(Some(2), echo, events).await;
        assert_eq!(chunks.len(), 2 + usize::from(echo));
        let mut offset = 0;
        if echo {
            assert_eq!(chunks[0]["choices"][0]["text"], "prompt");
            assert_eq!(
                chunks[0]["choices"][0]["logprobs"],
                json!({"tokens": ["prompt"], "token_logprobs": [null],
                       "top_logprobs": [null], "text_offset": [0]})
            );
            offset = "prompt".len();
        }
        for (chunk, (text, logprob)) in chunks[usize::from(echo)..]
            .iter()
            .zip([("answer", -0.5), ("secret", -1.0)])
        {
            let choice = &chunk["choices"][0];
            assert_eq!(choice["text"], text);
            assert_eq!(
                choice["logprobs"],
                json!({"tokens": [text], "token_logprobs": [logprob],
                       "top_logprobs": [{"answer": -0.5, "secret": -1.0}],
                       "text_offset": [offset]})
            );
            offset += text.len();
        }
    }
}

#[tokio::test]
async fn completion_stream_logprobs_zero_preserves_scores_and_missing_first_score() {
    let chunks = completion_stream_chunks(
        Some(0),
        false,
        vec![
            StreamEvent::Token(5),
            StreamEvent::TokenWithLogprobs(
                1,
                TokenLogprobs {
                    token_id: 1,
                    logprob: -0.5,
                    top: vec![],
                },
            ),
        ],
    )
    .await;
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        chunks[0]["choices"][0]["logprobs"],
        json!({"tokens": ["STOP"], "token_logprobs": [null],
               "top_logprobs": [null], "text_offset": [0]})
    );
    assert_eq!(
        chunks[1]["choices"][0]["logprobs"],
        json!({"tokens": ["answer"], "token_logprobs": [-0.5],
               "top_logprobs": [{}], "text_offset": [4]})
    );
}

#[tokio::test]
async fn completion_stream_omits_logprobs_unless_requested() {
    let chunks = completion_stream_chunks(
        None,
        true,
        vec![StreamEvent::Token(1), StreamEvent::Token(3)],
    )
    .await;
    let texts: Vec<_> = chunks
        .iter()
        .map(|chunk| {
            let choice = &chunk["choices"][0];
            assert!(choice.get("logprobs").is_none());
            choice["text"].as_str().unwrap()
        })
        .collect();
    assert_eq!(texts, ["prompt", "answer", "secret"]);
}

#[tokio::test]
async fn completion_with_top_p(top_p: Option<f32>, stream: bool) -> (Response, Option<f32>) {
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    let state = Arc::new(AppState {
        request_tx: tx,
        default_top_p: 0.75,
        ..crate::api::test_app_state::app_state(tokenizer())
    });
    let mut body = json!({"model": "fixture", "prompt": [1], "max_tokens": 1, "stream": stream});
    if let Some(p) = top_p {
        body["top_p"] = json!(p);
    }
    let scheduler = tokio::spawn(async move {
        match rx.recv().await? {
            InferenceRequest::Blocking {
                top_p, response_tx, ..
            } => {
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
                Some(top_p)
            }
            InferenceRequest::Streaming { top_p, .. } => Some(top_p),
        }
    });
    let req = serde_json::from_value(body).unwrap();
    let response =
        crate::api::completions::completions(CurrentModel(state), Ok(axum::Json(req))).await;
    (response, scheduler.await.unwrap())
}

#[tokio::test]
async fn completion_top_p_rejects_out_of_range_before_dispatch() {
    for stream in [true, false] {
        for top_p in [-0.1, 0.0, 1.1] {
            let (response, dispatched) = completion_with_top_p(Some(top_p), stream).await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "top_p={top_p}, stream={stream}, dispatched={dispatched:?}"
            );
            assert_eq!(dispatched, None);
            let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
                .await
                .unwrap();
            let body: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            assert_eq!(body["error"]["type"], "invalid_request_error");
            assert_eq!(body["error"]["param"], "top_p");
            assert!(body["error"]["message"].as_str().unwrap().contains("top_p"));
        }
    }
}

#[tokio::test]
async fn completion_top_p_accepts_valid_values_and_server_default() {
    for stream in [false, true] {
        for top_p in [Some(f32::MIN_POSITIVE), Some(0.5), Some(1.0), None] {
            let (response, dispatched) = completion_with_top_p(top_p, stream).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert_eq!(dispatched, Some(top_p.unwrap_or(0.75)));
        }
    }
}
