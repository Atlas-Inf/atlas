// SPDX-License-Identifier: AGPL-3.0-only

//! Decode named source frames, streamed: the subprocess side of a sampler
//! that chooses frames by index (GLM-5.3's) rather than by a time plan.
//!
//! The frames come back at their native size, because the processor resizes
//! from the source frame; each is handed to the caller as it arrives, so a
//! long clip never holds more than one native frame. They travel as binary
//! PPM, whose header states each frame's size: no compression on either side
//! and no signature scanning to split the stream.

use super::*;
use crate::vision_preprocess::{DECODE_MAX_ALLOC, DECODE_MAX_SIDE};

/// A clip's frame count and rate, as the reference's OpenCV capture reports
/// them: the stream's declared `nb_frames`, else `duration × fps` rounded
/// (OpenCV's own estimate), and the stream's average rate.
pub fn probe_frames(bytes: &[u8], policy: &FfmpegPolicy) -> Result<(usize, f64)> {
    ensure_decoder(policy)?;
    let timing = probe_duration(bytes, policy)?;
    let fps = timing.native_fps as f64;
    let frames = timing.frame_count.map_or_else(
        || (timing.duration_secs as f64 * fps + 0.5).floor() as usize,
        |frames| frames as usize,
    );
    ensure!(frames > 0, "the clip declares no video frames");
    Ok((frames, fps))
}

/// Decode the source frames numbered `indices` (ascending, distinct), handing
/// each to `on_frame` in order. Returns how many arrived: a container whose
/// declared count overstates its frames returns fewer.
pub fn decode_frame_indices(
    bytes: &[u8],
    indices: &[usize],
    policy: &FfmpegPolicy,
    on_frame: &mut dyn FnMut(RgbImage) -> Result<()>,
) -> Result<usize> {
    ensure_decoder(policy)?;
    ensure!(
        !indices.is_empty() && indices.len() <= crate::video_sample::MAX_SAMPLE_PLAN,
        "video sampling selected {} frames; the decoder takes 1..={}",
        indices.len(),
        crate::video_sample::MAX_SAMPLE_PLAN
    );
    let filter = crate::video_sample::select_frames(indices.iter().copied());
    // `passthrough`: one output frame per selected input frame, never a
    // duplicate to fill a constant-rate timeline.
    let output = ["-fps_mode", "passthrough", "-vcodec", "ppm"];
    let (count, _) = spawn_ffmpeg(
        bytes,
        &filter,
        indices.len(),
        &output,
        policy,
        "error",
        |stdout| {
            read_ppm_frames(
                &mut std::io::BufReader::new(stdout),
                indices.len(),
                on_frame,
            )
        },
    )?;
    Ok(count)
}

/// Read concatenated binary PPM (`P6`, maxval 255) frames, at most `limit`,
/// each bounded like a still image's decode before its pixels are allocated.
pub(super) fn read_ppm_frames(
    stream: &mut impl std::io::BufRead,
    limit: usize,
    on_frame: &mut dyn FnMut(RgbImage) -> Result<()>,
) -> Result<usize> {
    let mut count = 0usize;
    while !stream
        .fill_buf()
        .context("reading decoded frames")?
        .is_empty()
    {
        ensure!(
            count < limit,
            "the decoder returned more than {limit} frames"
        );
        let header = [
            ppm_token(stream)?,
            ppm_token(stream)?,
            ppm_token(stream)?,
            ppm_token(stream)?,
        ];
        ensure!(header[0] == "P6", "frame {count} is not a binary PPM");
        let [width, height, maxval] = [&header[1], &header[2], &header[3]].map(|field| {
            field
                .parse::<u32>()
                .with_context(|| format!("frame {count} has a malformed PPM header"))
        });
        let (width, height) = (width?, height?);
        ensure!(maxval? == 255, "frame {count} is not 8-bit RGB");
        let bytes = width as u64 * height as u64 * 3;
        ensure!(
            width > 0
                && height > 0
                && width.max(height) <= DECODE_MAX_SIDE
                && bytes <= DECODE_MAX_ALLOC,
            "frame {count} is {width}x{height}, above the decoded-image limit"
        );
        let mut pixels = vec![0u8; bytes as usize];
        stream
            .read_exact(&mut pixels)
            .with_context(|| format!("frame {count} ended early"))?;
        let frame = RgbImage::from_raw(width, height, pixels).context("frame buffer size")?;
        on_frame(frame)?;
        count += 1;
    }
    Ok(count)
}

/// One whitespace-delimited header token; the single whitespace byte after
/// it is consumed, which after the maxval is the one before the pixels.
fn ppm_token(stream: &mut impl std::io::BufRead) -> Result<String> {
    let mut token = String::new();
    loop {
        let mut byte = [0u8; 1];
        stream
            .read_exact(&mut byte)
            .context("a PPM header ended early")?;
        if byte[0].is_ascii_whitespace() {
            if token.is_empty() {
                continue;
            }
            return Ok(token);
        }
        ensure!(token.len() < 16, "a PPM header token is too long");
        token.push(byte[0] as char);
    }
}
