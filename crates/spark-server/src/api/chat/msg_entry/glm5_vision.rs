// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 image/video placeholder geometry for the template renderer.

use axum::http::StatusCode;
use axum::response::Response;

use crate::ir::MediaKind;
use crate::tokenizer::{Glm5VisionKind, Glm5VisionPlaceholder};

use super::super::super::compact::openai_error_response;

pub(super) fn checked_glm5_video_timestamps(
    timestamps: Vec<f32>,
    frame_count: usize,
) -> Result<Vec<f32>, String> {
    if timestamps.len() != frame_count {
        return Err(format!(
            "GLM-5 video metadata has {} timestamps for {} temporal groups",
            timestamps.len(),
            frame_count
        ));
    }
    Ok(timestamps)
}

/// The GLM-5 placeholder for one preprocessed media `item` whose rendered pad
/// run is `pad_count` tokens.
#[allow(clippy::result_large_err)]
pub(super) fn placeholder(
    kind: MediaKind,
    item: &spark_model::VisionItem,
    pad_count: usize,
    video_timestamps: Vec<f32>,
) -> Result<Glm5VisionPlaceholder, Response> {
    let frame_count = item.t_len();
    let frame_pad_count = pad_count / frame_count.max(1);
    let timestamps = if kind == MediaKind::Video {
        match checked_glm5_video_timestamps(video_timestamps, frame_count) {
            Ok(timestamps) => timestamps,
            Err(message) => {
                return Err(openai_error_response(StatusCode::BAD_REQUEST, message));
            }
        }
    } else {
        Vec::new()
    };
    Ok(Glm5VisionPlaceholder {
        kind: match kind {
            MediaKind::Image => Glm5VisionKind::Image,
            MediaKind::Video => Glm5VisionKind::Video,
        },
        pad_count,
        per_frame_pad_count: frame_pad_count,
        timestamps,
    })
}
