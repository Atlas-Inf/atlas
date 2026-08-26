// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B media preprocessing tests (Wave 5A): images, video frames
//! and audio clips pass through the gemma preprocessors (the gate that
//! 501'd ALL gemma media in Wave 1 now admits image + video + audio).
//! Split out of `msg_entry_tests.rs` so both files stay under the
//! 500-LoC cap.

use axum::http::StatusCode;

use crate::ir::message::{ContentPart, ImageData, ImageSource, Message, Role};
use crate::ir::{MediaData, MediaSource, VideoSource};

use super::super::build_msg_entries;

fn text(role: Role, t: &str) -> Message {
    Message {
        role,
        content: vec![ContentPart::Text(t.into())],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn image(role: Role) -> Message {
    Message {
        role,
        content: vec![
            ContentPart::Image(ImageSource {
                data: ImageData::Base64("data:image/png;base64,AAA".into()),
            }),
            ContentPart::Text("result".into()),
        ],
        tool_calls: Vec::new(),
        tool_call_id: Some("c1".into()),
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn audio(role: Role) -> Message {
    Message {
        role,
        content: vec![ContentPart::Audio(MediaSource {
            data: MediaData::Base64("data:audio/wav;base64,QUJD".into()),
        })],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn video(role: Role) -> Message {
    Message {
        role,
        content: vec![ContentPart::Video(VideoSource {
            frames: vec![MediaData::Base64("data:video/mp4;base64,REVG".into())],
        })],
        tool_calls: Vec::new(),
        tool_call_id: None,
        name: None,
        reasoning: None,
        tool_error: false,
    }
}

fn gemma_vision_config() -> atlas_core::config::GemmaVisionConfig {
    atlas_core::config::GemmaVisionConfig {
        hidden_size: 1152,
        intermediate_size: 4304,
        num_hidden_layers: 27,
        num_attention_heads: 16,
        head_dim: 72,
        patch_size: 16,
        pooling_kernel_size: 2,
        position_embedding_size: 1024,
        use_clipped_linears: false,
        image_token_id: 262144,
        rope_theta: 100.0,
        max_patches: 1120,
        max_soft_tokens: 280,
        position_table_shape: (2, 1024, 1152),
        norm_eps: 1e-6,
        video_frames: 32,
        video_soft_tokens_per_frame: 70,
        video_token_id: 262146,
        boi_token_id: 262147,
        eoi_token_id: 262148,
    }
}

/// A real decodable 1×1 PNG as a base64 data URI — the gemma image
/// preprocessor decodes frames, so fake base64 ("AAA") would fail at
/// the decode step, not the gate.
fn png_uri() -> String {
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGMIiEoBAAIMAQ+7sRU/AAAAAElFTkSuQmCC"
        .to_string()
}

/// A real decodable mono PCM16 16 kHz WAV as a base64 data URI — 40 ms
/// of a 1 kHz sine (640 samples), so the mel front end yields frames and
/// the audio path exercises the real decode→mel pipeline, not a gate.
fn wav_uri() -> String {
    use base64::Engine;
    let n = 640usize;
    let mut bytes = Vec::with_capacity(44 + n * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36u32 + (n * 2) as u32).to_le_bytes());
    bytes.extend_from_slice(b"WAVE");
    bytes.extend_from_slice(b"fmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&16_000u32.to_le_bytes());
    bytes.extend_from_slice(&32_000u32.to_le_bytes()); // byte rate
    bytes.extend_from_slice(&2u16.to_le_bytes()); // block align
    bytes.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&((n * 2) as u32).to_le_bytes());
    for i in 0..n {
        let v = (0.3
            * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 16_000.0).sin()
            * i16::MAX as f32) as i16;
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
    format!("data:audio/wav;base64,{b64}")
}

/// A minimal valid gemma-4 E2B audio config: the mel front end needs
/// `fft_size` a power of two, `frame_length`/`hop_length`/`mel_bins`/
/// `token_cap` non-zero (all validated by `mel::validate` before any
/// spectrogram is computed).
fn gemma_audio_config() -> atlas_core::config::GemmaAudioConfig {
    atlas_core::config::GemmaAudioConfig {
        hidden_size: 1024,
        num_hidden_layers: 12,
        num_attention_heads: 16,
        subsampling_conv_channels: vec![128, 32],
        conv_kernel_size: 3,
        attention_chunk_size: 12,
        attention_context_left: 13,
        attention_context_right: 0,
        output_proj_dims: 1536,
        residual_weight: 0.5,
        use_clipped_linears: true,
        audio_token_id: 258_881,
        mel_bins: 128,
        frame_length: 320,
        hop_length: 160,
        fft_size: 512,
        mel_floor: 1e-3,
        mel_scale: "htk".to_string(),
        token_cap: 750,
        norm_eps: 1e-6,
        activation: "silu".to_string(),
        boa_token_id: 256_000,
        eoa_token_id: 258_883,
    }
}

#[test]
fn gemma_image_with_undecodable_base64_is_400_not_501() {
    // Wave 5A lifts the gemma media gate: gemma images now flow to
    // the vision preprocessor. Undecodable base64 ("AAA" is not a
    // PNG) fails at the DECODE step with the named 400 — the Wave-1
    // catch-all 501 is gone for images.
    let gemma = gemma_vision_config();
    match build_msg_entries(
        None,
        None,
        Some(&gemma),
        None,
        &[text(Role::User, "hi"), image(Role::User)],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    ) {
        Ok(_) => panic!("expected 400, got Ok"),
        Err(resp) => assert_eq!(resp.status(), StatusCode::BAD_REQUEST),
    }
}

#[test]
fn gemma_image_passes_through_vision_preprocessor() {
    // Wave 5A lifts the Wave-1 gemma media gate: gemma images go
    // through `preprocess_gemma_image` (the vision tower exists) at
    // the image soft-token budget (280).
    let gemma = gemma_vision_config();
    let mut img = image(Role::User);
    img.content[0] = ContentPart::Image(ImageSource {
        data: ImageData::Base64(png_uri()),
    });
    let out = build_msg_entries(
        None,
        None,
        Some(&gemma),
        None,
        &[img],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    )
    .expect("gemma image builds");
    assert_eq!(out.gemma_media.len(), 1);
    // 1×1 frame at the 280-soft-token budget: unit = 2×16 = 32,
    // target area 280×1024 → 512×512 grid → 32×32 patches → 1024/4.
    assert_eq!(out.gemma_media[0].soft_token_count, 256);
}

#[test]
fn gemma_video_clip_frames_preprocess_at_video_budget() {
    // Wave 5A: a gemma video clip is N base64 image frames; msg_entry
    // preprocesses each frame with the gemma image preprocessor at
    // `video_soft_tokens_per_frame` (70), NOT the image budget (280).
    let gemma = gemma_vision_config();
    let mut vid = video(Role::User);
    vid.content[0] = ContentPart::Video(VideoSource {
        frames: vec![MediaData::Base64(png_uri()), MediaData::Base64(png_uri())],
    });
    let out = build_msg_entries(
        None,
        None,
        Some(&gemma),
        None,
        &[vid],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    )
    .expect("gemma video builds");
    assert_eq!(out.video_clip_frame_counts, vec![2]);
    assert_eq!(out.gemma_media.len(), 2);
    // 1×1 frame at the 70-soft-token budget: target area 70×1024 →
    // 256×256 grid → 16×16 patches → 256/4 = 64 (vs 256 at the image
    // budget) — proves the video budget was used.
    for frame in &out.gemma_media {
        assert_eq!(
            frame.soft_token_count, 64,
            "video-frame budget is 70, not 280"
        );
    }
    // The two frame inputs stay ordered (frame 0, frame 1) — the
    // splice consumes them in encounter order.
    assert_eq!(out.messages[0].video_count, 1);
}

#[test]
fn gemma_video_and_image_interleave_in_encounter_order() {
    // The gemma vision buf_out is shared by image AND video slots, so
    // the gemma media vec must interleave in token-stream order:
    // message-1 image, message-1 video frames, message-2 image.
    let gemma = gemma_vision_config();
    let mut m1_img = image(Role::User);
    m1_img.content[0] = ContentPart::Image(ImageSource {
        data: ImageData::Base64(png_uri()),
    });
    let mut m1_vid = video(Role::User);
    m1_vid.content[0] = ContentPart::Video(VideoSource {
        frames: vec![MediaData::Base64(png_uri())],
    });
    let mut m2_img = image(Role::User);
    m2_img.content[0] = ContentPart::Image(ImageSource {
        data: ImageData::Base64(png_uri()),
    });
    let out = build_msg_entries(
        None,
        None,
        Some(&gemma),
        None,
        &[m1_img, m1_vid, m2_img],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    )
    .expect("mixed gemma media builds");
    let soft: Vec<usize> = out.gemma_media.iter().map(|g| g.soft_token_count).collect();
    assert_eq!(
        soft,
        vec![256, 64, 256],
        "image(280) then video frame(70) per message"
    );
    assert_eq!(out.video_clip_frame_counts, vec![1]);
}

#[test]
fn gemma_audio_without_audio_config_is_400() {
    // Wave 5A lifts the gemma audio gate: audio now flows to the WAV→mel
    // front end. With only a VISION config wired (no audio encoder) an
    // audio part is rejected with a named 400, not the Wave-1 501.
    let gemma = gemma_vision_config();
    match build_msg_entries(
        None,
        None,
        Some(&gemma),
        None,
        &[text(Role::User, "hi"), audio(Role::User)],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    ) {
        Ok(_) => panic!("expected 400, got Ok"),
        Err(resp) => assert_eq!(resp.status(), StatusCode::BAD_REQUEST),
    }
}

#[test]
fn gemma_audio_with_config_builds_mel_input() {
    // Wave 5A lifts the gemma audio gate: a decodable WAV clip flows
    // through decode→resample→mel and lands in `gemma_audios` in
    // encounter order (the splice draws audio rows from that buffer).
    let gemma = gemma_vision_config();
    let mut clip = audio(Role::User);
    clip.content[0] = ContentPart::Audio(MediaSource {
        data: MediaData::Base64(wav_uri()),
    });
    let out = build_msg_entries(
        None,
        None,
        Some(&gemma),
        Some(&gemma_audio_config()),
        &[text(Role::User, "hi"), clip],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    )
    .expect("audio request ok");
    assert_eq!(out.gemma_audios.len(), 1);
    assert!(out.gemma_audios[0].n_frames >= 1);
    assert!(out.gemma_audios[0].n_mels > 0);
    assert_eq!(out.gemma_audios[0].mask.len(), out.gemma_audios[0].n_frames);
    assert!(out.gemma_audios[0].mask.iter().all(|&v| v == 1));
}

#[test]
fn gemma_audio_with_undecodable_wav_is_400() {
    // Undecodable audio base64 ("AAA" is not a RIFF/WAVE file) fails at
    // the DECODE step with the named 400 — the Wave-1 catch-all 501 is
    // gone for audio too.
    let gemma = gemma_vision_config();
    match build_msg_entries(
        None,
        None,
        Some(&gemma),
        Some(&gemma_audio_config()),
        &[text(Role::User, "hi"), audio(Role::User)],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    ) {
        Ok(_) => panic!("expected 400, got Ok"),
        Err(resp) => assert_eq!(resp.status(), StatusCode::BAD_REQUEST),
    }
}

#[test]
fn gemma_text_only_request_builds_with_media_config_present() {
    // The must-not-break case: once gemma media config threads through
    // AppState, text-only requests on gemma-4-e2b keep working.
    let gemma = gemma_vision_config();
    let out = build_msg_entries(
        None,
        None,
        Some(&gemma),
        None,
        &[text(Role::User, "hello")],
        false,
        &crate::api::chat::levers::ChatLevers::OFF,
    )
    .expect("text-only gemma ok");
    assert_eq!(out.messages.len(), 1);
    assert_eq!(out.messages[0].image_count, 0);
    assert!(out.gemma_media.is_empty());
}
