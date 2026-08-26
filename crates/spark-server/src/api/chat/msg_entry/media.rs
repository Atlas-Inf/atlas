// SPDX-License-Identifier: AGPL-3.0-only
//
// Media collection + preprocessing for `build_msg_entries` (Wave 5A).
// Images, audio, and video clips (gemma-4 E2B video = N base64 image
// frames, no ffmpeg) are validated at collection time (URL rejection)
// and turned into encoder inputs per model family: the Qwen3-VL
// preprocessor for non-gemma models, the gemma image preprocessor for
// gemma models — where IMAGE and VIDEO frames share one vision tower
// and therefore ONE ordered input vec (the splice draws both slot kinds
// from the same buf_out in token-stream order).

use axum::http::StatusCode;
use axum::response::Response;

use atlas_core::config::{GemmaAudioConfig, GemmaVisionConfig, VisionConfig};

use crate::ir::{ContentPart, ImageData, MediaData, Message};

use crate::api::compact::openai_error_response;

/// One gemma-4 E2B media unit in token-stream encounter order: an image,
/// a video frame, or an audio clip. The gemma vision tower processes
/// images + video frames and the audio tower the clips; the splice
/// consumes all three slot kinds from their own buffers in stream order.
pub(super) enum GemmaFrame {
    Image(String),
    VideoFrame(String),
    Audio(String),
}

/// Encoder inputs produced by [`preprocess_media`]. `image_pixels` is
/// the Qwen3-VL shape; `gemma_media` the gemma shape — exactly one is
/// populated per request (a model has one vision tower or the other).
pub(super) struct MediaPreprocessOut {
    pub(super) image_pixels: Vec<(Vec<f32>, usize, usize)>,
    pub(super) image_pad_counts: Vec<usize>,
    pub(super) gemma_media: Vec<spark_model::media::gemma_vision::GemmaImageInput>,
    /// Gemma-4 E2B audio clips in encounter order (parallel to the
    /// `<|audio|>` markers in the token stream).
    pub(super) gemma_audios: Vec<spark_model::media::gemma_audio::GemmaAudioInput>,
    /// Frames per video clip, in clip order (parallel to the clip
    /// boundaries inside `all_videos`).
    pub(super) video_clip_frame_counts: Vec<usize>,
}

/// Append the encoder-input string for every media part on `m` to the
/// per-modality vectors, growing `image_pad_counts` in lockstep (each pad
/// count is filled in later by the vision preprocessor). Shared by the
/// tool-message branch and the normal branch so media ride every role
/// uniformly — including tool results, the motivating case for issue #165.
/// `gemma_frames` additionally records the encounter order the gemma
/// preprocessor must preserve (see [`GemmaFrame`]).
#[allow(clippy::result_large_err)]
pub(super) fn collect_media(
    m: &Message,
    all_images: &mut Vec<String>,
    image_pad_counts: &mut Vec<usize>,
    all_audios: &mut Vec<String>,
    all_videos: &mut Vec<String>,
    video_clip_frame_counts: &mut Vec<usize>,
    gemma_frames: &mut Vec<GemmaFrame>,
) -> Result<(), Response> {
    for part in &m.content {
        match part {
            ContentPart::Image(src) => {
                let uri = match &src.data {
                    ImageData::Base64(s) => s.clone(),
                    // The encoder does not fetch remote URLs. Fed onward, the
                    // URL string would hit the base64 decoder and fail with a
                    // confusing "base64 decode failed" — reject with the real
                    // reason instead (PCND: fail fast).
                    ImageData::Url(url) => {
                        let shown: String = url.chars().take(120).collect();
                        return Err(openai_error_response(
                            StatusCode::BAD_REQUEST,
                            format!(
                                "image URLs are not fetched by this server (got '{shown}'); \
                                 send the image as a base64 data: URI"
                            ),
                        ));
                    }
                };
                all_images.push(uri.clone());
                image_pad_counts.push(0);
                gemma_frames.push(GemmaFrame::Image(uri));
            }
            ContentPart::Audio(src) => {
                let uri = media_uri(&src.data, "audio")?;
                all_audios.push(uri.clone());
                gemma_frames.push(GemmaFrame::Audio(uri));
            }
            ContentPart::Video(src) => {
                let mut clip_frames = 0usize;
                for frame in &src.frames {
                    let uri = media_uri(frame, "video")?;
                    all_videos.push(uri.clone());
                    gemma_frames.push(GemmaFrame::VideoFrame(uri));
                    clip_frames += 1;
                }
                video_clip_frame_counts.push(clip_frames);
            }
            ContentPart::Text(_) => {}
        }
    }
    Ok(())
}

/// Extract the encoder-input string from a media part, rejecting remote
/// URLs with a named error (the preprocessing layer never fetches URLs).
#[allow(clippy::result_large_err)]
pub(super) fn media_uri(data: &MediaData, modality: &str) -> Result<String, Response> {
    match data {
        MediaData::Base64(s) => Ok(s.clone()),
        MediaData::Url(url) => {
            let shown: String = url.chars().take(120).collect();
            Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                format!(
                    "{modality} URLs are not fetched by this server (got '{shown}'); \
                     send the {modality} as a base64 data: URI"
                ),
            ))
        }
    }
}

/// One shared fail-fast point for the collected media: if media were
/// supplied but the model has no encoder for them, reject the request
/// (issue #165) instead of silently dropping the user's input with a
/// 200. Gemma-4 E2B (Wave 5A): the vision tower handles IMAGES and VIDEO
/// FRAMES (video frames are images at the `video_soft_tokens_per_frame`
/// budget); the audio tower handles WAV clips through the mel front end.
#[allow(clippy::result_large_err)]
pub(super) fn preprocess_media(
    vision_config: Option<&VisionConfig>,
    vision_max_pixels: Option<usize>,
    gemma_vision_config: Option<&GemmaVisionConfig>,
    gemma_audio_config: Option<&GemmaAudioConfig>,
    all_images: &[String],
    image_pad_counts: &mut Vec<usize>,
    all_audios: &[String],
    all_videos: &[String],
    video_clip_frame_counts: Vec<usize>,
    gemma_frames: &[GemmaFrame],
) -> Result<MediaPreprocessOut, Response> {
    let mut image_pixels: Vec<(Vec<f32>, usize, usize)> = Vec::new();
    let has_media = !all_images.is_empty() || !all_audios.is_empty() || !all_videos.is_empty();
    if !has_media {
        return Ok(MediaPreprocessOut {
            image_pixels,
            image_pad_counts: std::mem::take(image_pad_counts),
            gemma_media: Vec::new(),
            gemma_audios: Vec::new(),
            video_clip_frame_counts,
        });
    }
    // Gemma-4 E2B: images + video frames pass through the gemma image
    // preprocessor (one shared vision tower); audio clips through the
    // WAV→mel front end. The encounter order of `gemma_frames` is
    // preserved so the splice rows line up with the token stream.
    if let Some(gvcfg) = gemma_vision_config {
        let gacfg = gemma_audio_config;
        let mut gemma_media = Vec::with_capacity(gemma_frames.len());
        let mut gemma_audios = Vec::new();
        for frame in gemma_frames {
            match frame {
                GemmaFrame::Image(uri) | GemmaFrame::VideoFrame(uri) => {
                    let budget = match frame {
                        GemmaFrame::Image(_) => gvcfg.max_soft_tokens,
                        _ => gvcfg.video_soft_tokens_per_frame,
                    };
                    match spark_model::media::gemma_vision::preprocess_gemma_image(
                        uri, gvcfg, budget,
                    ) {
                        Ok(input) => gemma_media.push(input),
                        Err(e) => {
                            return Err(openai_error_response(
                                StatusCode::BAD_REQUEST,
                                format!("Image decode error: {e}"),
                            ));
                        }
                    }
                }
                GemmaFrame::Audio(uri) => {
                    let Some(acfg) = gacfg else {
                        return Err(openai_error_response(
                            StatusCode::BAD_REQUEST,
                            "audio supplied but this model has no audio encoder".to_string(),
                        ));
                    };
                    match spark_model::media::gemma_audio::gemma_audio_input_from_wav(uri, acfg) {
                        Ok(input) => gemma_audios.push(input),
                        Err(e) => {
                            return Err(openai_error_response(
                                StatusCode::BAD_REQUEST,
                                format!("Audio decode error: {e}"),
                            ));
                        }
                    }
                }
            }
        }
        return Ok(MediaPreprocessOut {
            image_pixels,
            image_pad_counts: std::mem::take(image_pad_counts),
            gemma_media,
            gemma_audios,
            video_clip_frame_counts,
        });
    }
    // Gemma audio-only config (no vision tower): run the same WAV→mel
    // front end; there is no image/video support on this config.
    if let Some(acfg) = gemma_audio_config {
        if !all_images.is_empty() || !all_videos.is_empty() {
            return Err(openai_error_response(
                StatusCode::BAD_REQUEST,
                "this gemma model has no vision tower (image/video not accepted)".to_string(),
            ));
        }
        let mut gemma_audios = Vec::new();
        for uri in all_audios {
            match spark_model::media::gemma_audio::gemma_audio_input_from_wav(uri, acfg) {
                Ok(input) => gemma_audios.push(input),
                Err(e) => {
                    return Err(openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Audio decode error: {e}"),
                    ));
                }
            }
        }
        return Ok(MediaPreprocessOut {
            image_pixels,
            image_pad_counts: std::mem::take(image_pad_counts),
            gemma_media: Vec::new(),
            gemma_audios,
            video_clip_frame_counts,
        });
    }
    let Some(vcfg) = vision_config else {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            "this model does not accept image input (no vision config)".to_string(),
        ));
    };
    for (idx, uri) in all_images.iter().enumerate() {
        match spark_model::vision_preprocess::preprocess_image_with_max_pixels(
            uri,
            vcfg,
            vision_max_pixels,
        ) {
            Ok((pixels, grid_h, grid_w)) => {
                image_pad_counts[idx] = spark_model::vision_preprocess::image_pad_count(
                    grid_h,
                    grid_w,
                    vcfg.spatial_merge_size,
                );
                image_pixels.push((pixels, grid_h, grid_w));
            }
            Err(e) => {
                return Err(openai_error_response(
                    StatusCode::BAD_REQUEST,
                    format!("Image decode error: {e}"),
                ));
            }
        }
    }
    // Audio/video encoders don't exist for any supported non-gemma model
    // family — reject instead of silently dropping (same rationale as
    // images above).
    if !all_audios.is_empty() {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            "this model does not accept audio input (no audio encoder)".to_string(),
        ));
    }
    if !all_videos.is_empty() {
        return Err(openai_error_response(
            StatusCode::BAD_REQUEST,
            "this model does not accept video input (no video encoder)".to_string(),
        ));
    }
    Ok(MediaPreprocessOut {
        image_pixels,
        image_pad_counts: std::mem::take(image_pad_counts),
        gemma_media: Vec::new(),
        gemma_audios: Vec::new(),
        video_clip_frame_counts,
    })
}

#[cfg(test)]
#[path = "media_tests.rs"]
mod media_tests;
