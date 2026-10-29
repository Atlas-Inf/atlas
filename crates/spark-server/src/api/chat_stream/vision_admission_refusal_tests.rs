// SPDX-License-Identifier: AGPL-3.0-only

//! Vision-admission refusal through `prepare_chat_prompt` (#236).

use super::*;

#[test]
fn an_over_cap_image_is_refused_through_prepare_chat_prompt_with_vision_too_large() {
    // #236 regression: production callers used AdmitCtx::default(), so the
    // per-request row cap never fired. This drives the real path —
    // prepare_chat_prompt -> build_msg_entries — with the state's limits
    // installed; an image over the cap must be refused vision_input_too_large.
    let mut state = app(false);
    {
        let s = Arc::get_mut(&mut state).unwrap();
        s.vision_config = Some(atlas_core::config::VisionConfig {
            is_glm5_next: false,
            is_qwen3vl_video: false,
            in_channels: 3,
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            intermediate_size: 4_304,
            out_hidden_size: 2_560,
            projection_intermediate_size: 0,
            rms_norm_eps: 1e-6,
            swiglu_limit: 0.0,
            deepstack_visual_indexes: vec![],
            image_pad_token_id: 0,
            video_pad_token_id: 0,
            image_start_token_id: 0,
            image_end_token_id: 0,
            video_start_token_id: 0,
            video_end_token_id: 0,
            max_pixels: None,
            min_pixels: None,
            video_pixels: None,
            token_budgets: atlas_core::config::VisionTokenBudgets::default(),
            omni: None,
        });
        s.vision_limits = Some(spark_model::vision_limits::VisionRowLimits {
            window_rows: 65_536,
            chunk_tokens: 16_388,
            request_rows: 100,
        });
        s.vision_budget = Some(spark_model::vision_budget::VisionRowBudget::new(65_536));
    }
    // PNG header claiming 20000x20000 (tiny body): probe sizes it way over
    // request_rows=100 before any pixel is decoded.
    let mut ihdr = vec![0u8; 24];
    ihdr[..8].copy_from_slice(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]);
    ihdr[8..12].copy_from_slice(&13u32.to_be_bytes());
    ihdr[12..16].copy_from_slice(b"IHDR");
    ihdr[16..20].copy_from_slice(&20_000u32.to_be_bytes());
    ihdr[20..24].copy_from_slice(&20_000u32.to_be_bytes());
    let mut req = request();
    req.messages = vec![ir::Message {
        role: ir::Role::User,
        content: vec![ir::ContentPart::Image(ir::message::ImageSource {
            data: ir::ImageData::Base64(
                format!("data:image/png;base64,{}", base64_encode(&ihdr)).into(),
            ),
        })],
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }];
    let err = match crate::api::chat::prepare::prepare_chat_prompt(&state, &mut req) {
        Err(e) => e,
        Ok(_) => panic!("over-cap image must be refused at admission"),
    };
    let (parts, body) = err.into_parts();
    assert_eq!(parts.status, 400);
    let bytes = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(axum::body::to_bytes(body, usize::MAX))
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["error"]["code"].as_str().unwrap_or_default(),
        "vision_input_too_large"
    );
}

#[test]
fn a_video_request_with_a_spent_in_flight_budget_is_refused_busy_before_decode() {
    // #237 regression: a video request must take the conservative ceiling
    // reservation from the in-flight budget before decode. With the bound
    // set (FfmpegPolicy::max_total_decoded_pixels, wired from
    // --video-max-pixels), a spent budget must yield 503 vision_capacity_busy.
    let mut state = app(false);
    {
        let s = Arc::get_mut(&mut state).unwrap();
        s.vision_config = Some(atlas_core::config::VisionConfig {
            is_glm5_next: false,
            is_qwen3vl_video: false,
            in_channels: 3,
            depth: 27,
            hidden_size: 1152,
            num_heads: 16,
            patch_size: 16,
            temporal_patch_size: 2,
            spatial_merge_size: 2,
            intermediate_size: 4_304,
            out_hidden_size: 2_560,
            projection_intermediate_size: 0,
            rms_norm_eps: 1e-6,
            swiglu_limit: 0.0,
            deepstack_visual_indexes: vec![],
            image_pad_token_id: 0,
            video_pad_token_id: 0,
            image_start_token_id: 0,
            image_end_token_id: 0,
            video_start_token_id: 0,
            video_end_token_id: 0,
            max_pixels: None,
            min_pixels: None,
            video_pixels: None,
            token_budgets: atlas_core::config::VisionTokenBudgets::default(),
            omni: None,
        });
        s.vision_limits = Some(spark_model::vision_limits::VisionRowLimits {
            window_rows: 65_536,
            chunk_tokens: 16_388,
            request_rows: 65_536,
        });
        // One row left: any nonzero reservation spends it.
        s.vision_budget = Some(spark_model::vision_budget::VisionRowBudget::new(1));
        s.video_ffmpeg = spark_model::video_decode_ffmpeg::FfmpegPolicy {
            enabled: true,
            binary: "/nonexistent/ffmpeg".into(),
            max_total_decoded_pixels: Some(65_536),
            ..Default::default()
        };
    }
    let mut req = request();
    req.messages = vec![ir::Message {
        role: ir::Role::User,
        content: vec![ir::ContentPart::Video(ir::message::VideoSource {
            data: ir::ImageData::Base64("data:video/mp4;base64,AAAA".into()),
        })],
        tool_calls: vec![],
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }];
    let err = match crate::api::chat::prepare::prepare_chat_prompt(&state, &mut req) {
        Err(e) => e,
        Ok(_) => panic!("a spent vision budget must refuse the video before decode"),
    };
    let (parts, body) = err.into_parts();
    assert_eq!(
        parts.status, 503,
        "expected vision_capacity_busy, got {parts:?}"
    );
    let bytes = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(axum::body::to_bytes(body, usize::MAX))
        .unwrap();
    let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        v["error"]["code"].as_str().unwrap_or_default(),
        "vision_capacity_busy"
    );
}
