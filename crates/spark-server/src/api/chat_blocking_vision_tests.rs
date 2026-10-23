// SPDX-License-Identifier: AGPL-3.0-only

//! Regression coverage for vision inputs across blocking `n > 1` choices.

use std::sync::Arc;

use serde_json::json;

use super::{BlockingPathArgs, run_blocking_path};
use crate::AppState;
use crate::api::inference_types::{InferenceRequest, InferenceResponse};
use crate::api::test_app_state;
use spark_model::{VisionItem, vision_budget::VisionRowBudget};

fn tokenizer() -> crate::tokenizer::ChatTokenizer {
    let dir = tempfile::tempdir().unwrap();
    let vocab = ["[UNK]", "answer"]
        .iter()
        .enumerate()
        .map(|(i, word)| (word.to_string(), i as u32))
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

fn request() -> crate::ir::ChatRequest {
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
        "model": "fixture",
        "messages": [{"role": "user", "content": "describe this"}],
        "max_tokens": 1,
        "n": 2
    }))
    .unwrap();
    wire.into()
}

fn response() -> InferenceResponse {
    InferenceResponse {
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
    }
}

#[tokio::test]
async fn blocking_choices_each_receive_vision_and_share_admission() {
    let (request_tx, mut request_rx) = tokio::sync::mpsc::channel(1);
    let state = Arc::new(AppState {
        request_tx,
        ..test_app_state::app_state(tokenizer())
    });

    let budget = VisionRowBudget::new(8);
    let permit = budget.try_acquire(4).unwrap();
    let image = VisionItem::image(vec![42.0; 4], 2, 2).holding(Arc::new(permit));
    let scheduler_budget = budget.clone();

    let scheduler = tokio::spawn(async move {
        let mut first_groups = None;
        for choice_idx in 0..2 {
            let request = request_rx.recv().await.expect("blocking request");
            let InferenceRequest::Blocking {
                image_pixels,
                response_tx,
                ..
            } = request
            else {
                panic!("expected blocking request for choice {choice_idx}");
            };

            assert_eq!(image_pixels.len(), 1, "choice {choice_idx} lost its image");
            assert_eq!(image_pixels[0].groups[0][0], 42.0);
            assert_eq!(image_pixels[0].held.rows(), 4);
            if choice_idx == 0 {
                first_groups = Some(image_pixels[0].groups.clone());
            } else {
                assert!(Arc::ptr_eq(
                    first_groups.as_ref().unwrap(),
                    &image_pixels[0].groups
                ));
            }
            assert_eq!(
                scheduler_budget.in_use(),
                4,
                "choice {choice_idx} released the shared admission permit"
            );
            assert!(response_tx.send(Ok(response())).is_ok());
        }
    });

    let outcome = run_blocking_path(BlockingPathArgs {
        state,
        req: request(),
        req_ctx: None,
        prompt_tokens: vec![1],
        session_hash: 0,
        adapter_slot: -1,
        src_lang_id: 0,
        tgt_lang_id: 0,
        num_beams: 1,
        length_penalty: 1.0,
        early_stopping: false,
        image_pixels: vec![image],
        max_tokens: 1,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        top_n_sigma: 0.0,
        min_p: 0.0,
        repetition_penalty: 1.0,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        dry_multiplier: 0.0,
        dry_base: 1.0,
        dry_allowed_length: 0,
        lz_penalty: 0.0,
        logit_bias: vec![],
        stop_tokens: vec![],
        enable_thinking: false,
        thinking_budget: None,
        tools_active: false,
        tool_choice_required: false,
        suppress_tool_call: false,
        grammar_spec: None,
        top_logprobs: None,
        timeout_at: None,
        cwd_hint: None,
        prompt_len: 1,
    })
    .await;

    match outcome {
        crate::api::chat::ChatOutcome::Blocking(response) => {
            assert_eq!(response.choices.len(), 2);
        }
        _ => panic!("expected a blocking response"),
    }
    scheduler.await.unwrap();
    assert_eq!(
        budget.in_use(),
        0,
        "the final choice did not release vision rows"
    );
}
