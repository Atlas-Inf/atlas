// SPDX-License-Identifier: AGPL-3.0-only

//! Decode-budget and PNG-stream helpers (extracted from
//! `video_decode_ffmpeg.rs` under the 500-LoC cap; no logic edits).

use super::*;

/// Per-frame pixel bound under the aggregate decode budget: the smaller of
/// the per-frame cap and `max_total_decoded_pixels / frames` when the
/// operator set the total bound (Enntity/atlas 8fe99bfe).
pub(super) fn decoded_frame_budget(policy: &FfmpegPolicy, frames: usize) -> Result<usize> {
    let area = policy
        .max_total_decoded_pixels
        .map_or(policy.max_decoded_pixels, |total| {
            policy.max_decoded_pixels.min(total / frames.max(1))
        });
    ensure!(
        area > 0,
        "video decode pixel budget cannot hold {frames} frames"
    );
    Ok(area)
}

/// Split a concatenated PNG stream into images.
///
/// Splitting on the signature rather than parsing IEND chunks: a PNG's
/// payload can legitimately contain the IEND byte pattern, whereas the
/// 8-byte signature only appears at a file start in a well-formed stream
/// produced by `image2pipe`.
pub(super) fn split_png_stream(buf: &[u8]) -> Result<Vec<RgbImage>> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + PNG_MAGIC.len() <= buf.len() {
        if buf[i..i + PNG_MAGIC.len()] == PNG_MAGIC {
            starts.push(i);
            i += PNG_MAGIC.len();
        } else {
            i += 1;
        }
    }
    let mut frames = Vec::with_capacity(starts.len());
    for (n, &s) in starts.iter().enumerate() {
        let e = starts.get(n + 1).copied().unwrap_or(buf.len());
        let img = image::load_from_memory_with_format(&buf[s..e], image::ImageFormat::Png)
            .with_context(|| format!("frame {n} did not decode as PNG"))?;
        frames.push(img.to_rgb8());
    }
    Ok(frames)
}
