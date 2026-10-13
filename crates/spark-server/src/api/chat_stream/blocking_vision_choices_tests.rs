// SPDX-License-Identifier: AGPL-3.0-only

//! Independent blocking-dispatch regression with a small text-tokenizer fixture.
use crate::AppState;
use crate::api::chat::ChatOutcome;
use crate::api::chat_blocking::{BlockingPathArgs, run_blocking_path};
use crate::api::inference_types::InferenceRequest;
use serde_json::json;
use spark_model::VisionItem;
use std::{sync::Arc, time::Duration};

#[tokio::test]
async fn every_blocking_choice_carries_its_own_vision_input() {
    let image = VisionItem::image(vec![0.25; 8], 2, 4);
    let video = VisionItem {
        groups: vec![vec![0.5; 4], vec![0.75; 4]],
        grid_h: 2,
        grid_w: 2,
    };
    // Include text-only requests and ordered mixed media; n=1 must retain
    // its behavior while every independently scheduled n>1 choice sees it.
    for media in [vec![], vec![image.clone()], vec![video, image]] {
        for n in [1, 3] {
            let mut state = app();
            let (tx, mut rx) = tokio::sync::mpsc::channel(1);
            Arc::get_mut(&mut state).unwrap().request_tx = tx;
            let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(json!({
                "model":"fixture", "messages":[{"role":"user","content":"picture"}],
                "n":n, "seed":23, "max_tokens":1
            }))
            .unwrap();
            let args = BlockingPathArgs {
                state,
                req: wire.into(),
                req_ctx: None,
                prompt_tokens: vec![14, 0, 14],
                session_hash: 9,
                adapter_slot: -1,
                src_lang_id: 0,
                tgt_lang_id: 0,
                num_beams: 1,
                length_penalty: 1.0,
                early_stopping: false,
                image_pixels: media.clone(),
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
                dry_base: 1.75,
                dry_allowed_length: 2,
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
                prompt_len: 3,
            };
            let handler = tokio::spawn(run_blocking_path(args));
            for index in 0..n {
                let request = tokio::time::timeout(Duration::from_secs(5), rx.recv())
                    .await
                    .expect("choice was not scheduled")
                    .expect("handler stopped before all choices were scheduled");
                let InferenceRequest::Blocking {
                    image_pixels,
                    prompt_tokens,
                    seed,
                    response_tx,
                    ..
                } = request
                else {
                    panic!("unexpected streaming choice");
                };
                assert_eq!(image_pixels, media, "choice {index} lost its media");
                assert_eq!(*prompt_tokens, [14, 0, 14]);
                assert_eq!(seed, Some(23 + index as u64));
                // Each scheduler call must complete before the next arrives.
                // Consume its media, just as scheduler prefill does.
                drop(image_pixels);
                assert!(response_tx.send(Ok(response(vec![14]))).is_ok());
            }
            let outcome = tokio::time::timeout(Duration::from_secs(5), handler)
                .await
                .unwrap()
                .unwrap();
            let ChatOutcome::Blocking(output) = outcome else {
                panic!("expected the assembled blocking response");
            };
            assert_eq!(output.choices.len(), n);
            for (index, choice) in output.choices.iter().enumerate() {
                assert_eq!(choice.index, index);
            }
            assert!(rx.try_recv().is_err());
        }
    }
}

fn app() -> Arc<crate::AppState> {
    let dir = tempfile::tempdir().unwrap();
    let vocab = (0..15)
        .map(|i| {
            let word = match i {
                0 => "[UNK]".into(),
                14 => "42".into(),
                _ => format!("token{i}"),
            };
            (word, i)
        })
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
    let tokenizer = crate::tokenizer::ChatTokenizer::from_model_dir(
        dir.path(),
        0,
        true,
        "qwen3",
        "",
        None,
        false,
    )
    .unwrap();
    let (request_tx, _) = tokio::sync::mpsc::channel(1);
    Arc::new(AppState {
        tokenizer,
        model_name: "fixture".into(),
        adapter_name: None,
        adapter_names: vec![],
        active_adapter: Arc::default(),
        max_seq_len: 4096,
        request_tx,
        rotation_tx: None,
        vision_config: None,
        vision_max_pixels: None,
        remote_image_policy: Default::default(),
        video_ffmpeg: Default::default(),
        video_fps: 1.0,
        default_temperature: 0.0,
        default_top_k: 0,
        default_top_p: 1.0,
        default_top_n_sigma: 0.0,
        default_min_p: 0.0,
        tool_call_parser: None,
        chat: Default::default(),
        reasoning_parser: None,
        think_end_token_id: None,
        think_start_token_id: None,
        tool_max_tokens: 192,
        sampling_presets: Default::default(),
        tool_call_start_token_id: None,
        auto_compact_threshold: None,
        request_timeout: 0,
        effective_context: 0,
        behavior: Default::default(),
        disable_thinking: false,
        default_thinking: Default::default(),
        default_reasoning_effort: None,
        response_store: crate::response_store::ResponseStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        conversation_store: crate::conversation_store::ConversationStore::with_config(
            1,
            Duration::from_secs(60),
        ),
        rate_limiter: crate::rate_limiter::RateLimiter::with_config(
            crate::rate_limiter::RateLimitConfig {
                rpm: 0,
                tpm: 0,
                burst_rpm: 1,
                burst_tpm: 1,
            },
        ),
        dump_writer: None,
        lora_stageable: Default::default(),
        lora_peer_addr: None,
        promotion: None,
        promoted_slots: Arc::default(),
        lora_disk_stageable: Default::default(),
    })
}

fn response(tokens: Vec<u32>) -> crate::api::inference_types::InferenceResponse {
    crate::api::inference_types::InferenceResponse {
        output_tokens: tokens,
        finish_reason: "length".into(),
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        logprobs: vec![],
        reasoning_tokens: 1,
        cached_prompt_tokens: 0,
        accepted_prediction_tokens: 0,
        prompt_logprobs: vec![],
    }
}
