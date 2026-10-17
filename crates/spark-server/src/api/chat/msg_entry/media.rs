// SPDX-License-Identifier: AGPL-3.0-only

//! Media preprocessing under admission (`vision_admit`).
//!
//! Three passes, so a request is refused before it costs memory and holds
//! the in-flight budget all at once or not at all:
//! 1. size every item (stills from their headers, without decoding; videos by
//!    decoding, as they have no header probe), checking the request's
//!    bounds as each is sized: a 400 stops it early;
//! 2. the whole-request bounds (400), then the request's rows from the
//!    in-flight budget in one acquire (503);
//! 3. decode the stills, and attach the request's permit to every item.
//!
//! ONE sequence over the collected media, so `items[i]`, `pad_counts[i]` and
//! the i-th rendered marker are the same item by construction.

use axum::http::StatusCode;
use axum::response::Response;

use atlas_core::config::VisionConfig;
use spark_model::VisionItem;
use spark_model::vision_preprocess::{ProbedImage, probe_image};

use super::super::super::compact::openai_error_response;
use super::super::vision_admit::Admission;
use super::{MediaInput, VideoDecode, glm5_vision};
use crate::ir::MediaKind;
use crate::tokenizer::Glm5VisionPlaceholder;

/// An item sized but (for a still) not yet decoded.
enum Sized {
    Still(ProbedImage),
    Video(VisionItem, Vec<f32>),
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
) -> Result<Vec<VisionItem>, Response> {
    let sms = vcfg.spatial_merge_size;
    let image_error =
        |e| openai_error_response(StatusCode::BAD_REQUEST, format!("Image decode error: {e}"));
    let mut sized = Vec::with_capacity(media.len());
    for input in media {
        sized.push(match input.kind {
            MediaKind::Image => {
                let probed =
                    probe_image(&input.uri, vcfg, vision_max_pixels).map_err(image_error)?;
                admission.note(probed.pad_count(sms))?;
                Sized::Still(probed)
            }
            MediaKind::Video => {
                let v = spark_model::video_preprocess::preprocess_video(
                    &input.uri,
                    vcfg,
                    vision_max_pixels,
                    video.fps,
                    video.ffmpeg,
                )
                .map_err(|e| {
                    openai_error_response(
                        StatusCode::BAD_REQUEST,
                        format!("Video decode error: {e:#}"),
                    )
                })?;
                let item = VisionItem::video(v.groups, v.grid_h, v.grid_w);
                admission.note(item.pad_count(sms))?;
                Sized::Video(item, v.timestamps)
            }
        });
    }
    let admitted = admission.admit()?;

    let mut items = Vec::with_capacity(sized.len());
    for (idx, (input, sized)) in media.iter().zip(sized).enumerate() {
        let (item, video_timestamps) = match sized {
            Sized::Still(probed) => {
                let (pixels, grid_h, grid_w) = probed.preprocess(vcfg).map_err(image_error)?;
                (VisionItem::image(pixels, grid_h, grid_w), Vec::new())
            }
            Sized::Video(item, timestamps) => (item, timestamps),
        };
        let item = admitted.attach(item);
        pad_counts[idx] = item.pad_count(sms);
        if vcfg.is_glm5_next {
            placeholders.push(glm5_vision::placeholder(
                input.kind,
                &item,
                pad_counts[idx],
                video_timestamps,
            )?);
        }
        if input.kind == MediaKind::Video {
            // Logged at the media index, not a video ordinal: the index is
            // what lines the clip up with its pad run and its encoder rows.
            tracing::info!(
                "Video (media item {}): {} temporal groups, {}x{} patches, {} vision tokens",
                idx,
                item.t_len(),
                item.grid_h,
                item.grid_w,
                pad_counts[idx],
            );
        }
        items.push(item);
    }
    Ok(items)
}
