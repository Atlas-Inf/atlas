// SPDX-License-Identifier: AGPL-3.0-only

//! FFmpeg scaling for the bounded image-pipe path.

/// Scale a frame down only when its area exceeds `max_pixels`, keeping its
/// aspect ratio and never enlarging a smaller frame.
pub(super) fn decoded_scale_filter(max_pixels: usize) -> String {
    let area = max_pixels.max(1);
    // First bound the height, then derive the width from that integer height.
    // This makes the area bound explicit even for extreme aspect ratios and
    // avoids relying on FFmpeg's rounding of two independent square roots.
    let height = format!("min(ih,min({area},max(1,floor(sqrt({area}*ih/iw)))))");
    let width = format!("min(iw,max(1,floor({area}/({height}))))");
    format!("scale='{width}':'{height}':force_original_aspect_ratio=decrease")
}
