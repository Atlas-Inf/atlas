// SPDX-License-Identifier: AGPL-3.0-only

//! Media preprocessing under admission (`vision_admit`).
//!
//! Three passes, so a request is refused before it costs unaccounted memory
//! and holds the in-flight budget all at once or not at all:
//! 1. size every item (stills from their headers, without decoding; videos by
//!    decoding under a conservative Qwen reservation, as they have no header
//!    probe), checking the request's bounds as each is sized: a 400 stops it
//!    early;
//! 2. the whole-request bounds (400), then the request's rows from the
//!    in-flight budget in one acquire (503);
//! 3. decode the stills, and attach the request's permit to every item.
//!
//! ONE sequence over the collected media, so the flattened Qwen temporal
//! groups remain in media order while `pad_counts[i]` and the i-th rendered
//! marker continue to describe the original media part.

use axum::http::StatusCode;
use axum::response::Response;

use atlas_core::config::VisionConfig;
use spark_model::VisionItem;
use spark_model::vision_preprocess::{ProbedImage, probe_image};

use super::super::super::compact::openai_error_response;
use super::super::vision_admit::Admission;
use super::{MediaInput, VideoDecode, glm5_vision};
use crate::ir::MediaKind;
use crate::tokenizer::{Glm5VisionPlaceholder, QwenVisionKind, QwenVisionPlaceholder};

/// An item sized but (for a still) not yet decoded.
enum Sized {
    Still(ProbedImage),
    /// GLM keeps one item for the whole clip. Qwen splits the temporal groups
    /// into one item each so admission and chunked vision encoding can bound a
    /// long clip by the encoder window. The group buffers are moved out of
    /// `PreprocessedVideo`, never cloned.
    Video {
        items: Vec<VisionItem>,
        timestamps: Vec<f32>,
        pad_count: usize,
    },
}

/// Conservative Qwen row reservation while an opaque video is decoded.
///
/// Qwen's video processor bounds the whole `t_bar × height × width` volume.
/// One merged row consumes `temporal_patch_size × patch_size² ×
/// spatial_merge_size²` pixels of that volume. The frame cap limits the
/// alternative per-frame encoder capacity when a checkpoint's video bound is
/// larger. The result is an upper bound; `Admission::admit` shrinks it to the
/// actual groups after preprocessing.
fn qwen_video_rows_ceiling(
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    video: &VideoDecode<'_>,
) -> usize {
    let ps = vcfg.patch_size.max(1);
    let sms = vcfg.spatial_merge_size.max(1);
    let tp = vcfg.temporal_patch_size.max(1);
    let p_max = spark_model::layers::vision_encoder::derive_max_patches(max_pixels, ps).0;
    let per_frame_pixels = p_max.saturating_mul(ps).saturating_mul(ps);
    let max_frames = video.ffmpeg.max_frames.max(1);
    // Preprocessing repeats an odd final frame to fill its temporal group.
    let padded_frames = max_frames.div_ceil(tp).saturating_mul(tp);
    let frame_volume = padded_frames.saturating_mul(per_frame_pixels);
    let configured_volume = vcfg
        .video_pixels
        .map(|bounds| bounds.max_pixels)
        .filter(|&pixels| pixels > 0)
        .unwrap_or(usize::MAX);
    let volume = configured_volume.min(frame_volume);
    let row_pixels = tp
        .saturating_mul(ps)
        .saturating_mul(ps)
        .saturating_mul(sms)
        .saturating_mul(sms)
        .max(1);
    volume.div_ceil(row_pixels).max(1)
}

/// The items of `media`, in order; fills `pad_counts` and, for GLM,
/// `placeholders`.
#[allow(clippy::result_large_err, clippy::too_many_arguments)]
pub(super) fn preprocess_media(
    media: &[MediaInput],
    vcfg: &VisionConfig,
    vision_max_pixels: Option<usize>,
    video: &VideoDecode<'_>,
    mut admission: Admission<'_>,
    pad_counts: &mut [usize],
    placeholders: &mut Vec<Glm5VisionPlaceholder>,
    qwen_placeholders: &mut Vec<QwenVisionPlaceholder>,
) -> Result<Vec<VisionItem>, Response> {
    let sms = vcfg.spatial_merge_size;
    // Legacy short-video profiles retain their decode gate and final admission.
    // Only an explicit aggregate bound makes a pre-decode reservation useful.
    let reserve_decode = !vcfg.is_glm5_next && video.ffmpeg.max_total_decoded_pixels.is_some();
    let image_error =
        |e| openai_error_response(StatusCode::BAD_REQUEST, format!("Image decode error: {e}"));
    let mut sized = Vec::with_capacity(media.len());
    let mut actual_video_rows = 0usize;
    for input in media {
        sized.push(match input.kind {
            MediaKind::Image => {
                let probed =
                    probe_image(&input.uri, vcfg, vision_max_pixels).map_err(image_error)?;
                admission.note(probed.pad_count(sms))?;
                Sized::Still(probed)
            }
            MediaKind::Video => {
                // Decoded before admission (no header probe): under the
                // large-decode gate, so videos cannot add to the decode peak.
                if reserve_decode {
                    // Reserve only this video's worst case. After it is
                    // decoded, reconcile to its actual rows before moving to
                    // the next clip, so several short clips do not consume
                    // the full hour-video ceiling each.
                    let ceiling = qwen_video_rows_ceiling(vcfg, vision_max_pixels, video);
                    admission.reserve_video(ceiling)?;
                }
                let v = spark_model::vision_preprocess::decode_gated(|| {
                    spark_model::video_preprocess::preprocess_video(
                        &input.uri,
                        vcfg,
                        vision_max_pixels,
                        video.fps,
                        video.ffmpeg,
                    )
                })
                .map_err(|e| {
                    openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Video decode error: {e:#}"),
                    )
                })?;
                let timestamps = v.timestamps;
                let grid_h = v.grid_h;
                let grid_w = v.grid_w;
                if vcfg.is_glm5_next {
                    let item = VisionItem::video(v.groups, grid_h, grid_w);
                    let pad_count = item.pad_count(sms);
                    admission.note(pad_count)?;
                    Sized::Video {
                        items: vec![item],
                        timestamps,
                        pad_count,
                    }
                } else {
                    // Moving each group out of the owned preprocessing result
                    // keeps the large f32 allocation shared by ownership
                    // transfer; cloning the groups here would multiply a
                    // one-hour clip's host memory by thousands of items.
                    let items: Vec<VisionItem> = v
                        .groups
                        .into_iter()
                        .map(|group| VisionItem::video(vec![group], grid_h, grid_w))
                        .collect();
                    let mut pad_count = 0usize;
                    for item in &items {
                        let rows = item.pad_count(sms);
                        admission.note(rows)?;
                        pad_count = pad_count.saturating_add(rows);
                    }
                    actual_video_rows = actual_video_rows.saturating_add(pad_count);
                    if reserve_decode {
                        admission.reconcile_video(actual_video_rows)?;
                    }
                    Sized::Video {
                        items,
                        timestamps,
                        pad_count,
                    }
                }
            }
        });
    }
    let admitted = admission.admit()?;

    let mut items = Vec::with_capacity(sized.len());
    for (idx, (input, sized)) in media.iter().zip(sized).enumerate() {
        match sized {
            Sized::Still(probed) => {
                let (pixels, grid_h, grid_w) = probed.preprocess(vcfg).map_err(image_error)?;
                let item = admitted.attach(VisionItem::image(pixels, grid_h, grid_w));
                pad_counts[idx] = item.pad_count(sms);
                if vcfg.is_glm5_next {
                    placeholders.push(glm5_vision::placeholder(
                        input.kind,
                        &item,
                        pad_counts[idx],
                        Vec::new(),
                        false, // GLM path: qwen3vl=false
                    )?);
                } else {
                    qwen_placeholders.push(QwenVisionPlaceholder {
                        kind: QwenVisionKind::Image,
                        pad_count: pad_counts[idx],
                        per_frame_pad_count: pad_counts[idx],
                        timestamps: Vec::new(),
                    });
                }
                items.push(item);
            }
            Sized::Video {
                items: video_items,
                timestamps,
                pad_count,
            } => {
                let first = video_items
                    .first()
                    .expect("preprocess_video produces at least one temporal group");
                pad_counts[idx] = pad_count;
                if vcfg.is_glm5_next {
                    // GLM's existing single-item video contract is retained.
                    placeholders.push(glm5_vision::placeholder(
                        input.kind,
                        first,
                        pad_counts[idx],
                        timestamps,
                        false, // GLM path: qwen3vl=false
                    )?);
                } else {
                    let per_frame_pad_count = (first.grid_h / sms) * (first.grid_w / sms);
                    qwen_placeholders.push(QwenVisionPlaceholder {
                        kind: QwenVisionKind::Video,
                        pad_count: pad_counts[idx],
                        per_frame_pad_count,
                        timestamps,
                    });
                }
                if input.kind == MediaKind::Video {
                    // Logged at the media index, not a video ordinal: the
                    // index is what lines the clip up with its pad run and
                    // its encoder rows.
                    tracing::info!(
                        "Video (media item {}): {} temporal groups, {}x{} patches, {} vision tokens",
                        idx,
                        if vcfg.is_glm5_next {
                            first.t_len()
                        } else {
                            video_items.len()
                        },
                        first.grid_h,
                        first.grid_w,
                        pad_counts[idx],
                    );
                }
                items.extend(video_items.into_iter().map(|item| admitted.attach(item)));
            }
        }
    }
    Ok(items)
}

#[cfg(test)]
mod tests {
    use super::qwen_video_rows_ceiling;
    use crate::api::chat::msg_entry::VideoDecode;
    use atlas_core::config::PixelBounds;
    use spark_model::video_decode_ffmpeg::FfmpegPolicy;
    use spark_model::vision_preprocess::fixtures::qwen_config;

    #[test]
    fn hour_profile_reservation_matches_224k_qwen_rows() {
        let mut vcfg = qwen_config();
        vcfg.video_pixels = Some(PixelBounds {
            min_pixels: 4_096,
            max_pixels: 469_762_048,
        });
        let ffmpeg = FfmpegPolicy {
            max_frames: 7_200,
            ..Default::default()
        };
        let video = VideoDecode {
            ffmpeg: &ffmpeg,
            fps: 2.0,
        };
        assert_eq!(
            qwen_video_rows_ceiling(&vcfg, Some(16_777_216), &video),
            229_376
        );
    }

    #[test]
    fn odd_video_cap_reserves_the_padded_temporal_group() {
        let mut vcfg = qwen_config();
        vcfg.video_pixels = Some(PixelBounds {
            min_pixels: 4_096,
            max_pixels: 469_762_048,
        });
        let reservation = |max_frames| {
            let ffmpeg = FfmpegPolicy {
                max_frames,
                ..Default::default()
            };
            let video = VideoDecode {
                ffmpeg: &ffmpeg,
                fps: 2.0,
            };
            qwen_video_rows_ceiling(&vcfg, Some(16_777_216), &video)
        };
        assert_eq!(reservation(1), reservation(2));
        assert_eq!(reservation(3), reservation(4));
    }
}
