// SPDX-License-Identifier: AGPL-3.0-only

//! In-process GIF frame access shared by the planned and the GLM samplers.

use anyhow::{Context, Result, ensure};
use image::RgbImage;

/// Pass one over a GIF: replay the decoder's frame ITERATOR without
/// materialising the pixel buffers, to count frames, accumulate the declared
/// duration in ms and find the largest frame area. The decoder still
/// decompresses each frame, but drops it as it arrives, so peak memory is one
/// frame.
pub(super) fn gif_timeline(bytes: &[u8]) -> Result<(usize, f64, usize)> {
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;

    let header =
        GifDecoder::new(std::io::Cursor::new(bytes.to_vec())).context("not a decodable GIF")?;
    let mut count = 0usize;
    let mut total_ms = 0f64;
    let mut max_area = 0usize;
    for frame in header.into_frames() {
        let frame = frame.context("failed to decode animation frames")?;
        let buf = frame.buffer();
        max_area = max_area.max(buf.width() as usize * buf.height() as usize);
        let (num, den) = frame.delay().numer_denom_ms();
        if den != 0 {
            total_ms += num as f64 / den as f64;
        }
        count += 1;
    }
    ensure!(count > 0, "the container decoded to zero frames");
    Ok((count, total_ms, max_area))
}

/// Decode the GIF frames numbered `keep` (ascending, distinct), handing each
/// to `on_frame` in order. Returns how many were handed over.
pub(super) fn gif_frames(
    bytes: &[u8],
    keep: &[usize],
    on_frame: &mut dyn FnMut(RgbImage) -> Result<()>,
) -> Result<usize> {
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;

    let mut wanted = keep.iter().peekable();
    let mut handed = 0usize;
    for (i, frame) in GifDecoder::new(std::io::Cursor::new(bytes.to_vec()))
        .context("not a decodable GIF")?
        .into_frames()
        .enumerate()
    {
        if wanted.peek().is_none() {
            break;
        }
        if wanted.peek() != Some(&&i) {
            continue;
        }
        wanted.next();
        let frame = frame.context("failed to decode animation frames")?;
        on_frame(image::DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8())?;
        handed += 1;
    }
    Ok(handed)
}
