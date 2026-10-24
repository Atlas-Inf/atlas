// SPDX-License-Identifier: AGPL-3.0-only

//! Bounded in-process GIF sampling.
use super::*;

/// In-process GIF decode, bounded the same way the ffmpeg path is.
///
/// The GIF's own timeline is walked once to learn its duration, then the
/// shared planner picks `wanted` positions spanning it — so a GIF and an MP4
/// of the same clip sample the same way, and a GIF's timestamps are source
/// times rather than an assumed constant rate.
///
/// The declared dimensions and frame count come from a cheap header pass
/// BEFORE any pixels are allocated: a small file can declare an enormous GIF
/// canvas, and the HTTP body cap does not bound what decoding it costs. The
/// same pass refuses a clip whose decoded RGB would exceed the policy's
/// `max_frames × max_decoded_pixels` budget, which is what `ffmpeg` is
/// bounded by too.
pub(super) fn decode_gif(
    bytes: &[u8],
    target_fps: f32,
    policy: &crate::video_decode_ffmpeg::FfmpegPolicy,
) -> Result<(Vec<RgbImage>, Vec<f32>)> {
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;

    // Pass one: replay the decoder's frame ITERATOR without materialising the
    // pixel buffers, to count frames and accumulate the declared duration. The
    // decoder still decompresses each frame here, but the buffers are dropped
    // as they arrive rather than collected, so peak memory is one frame.
    let header =
        GifDecoder::new(std::io::Cursor::new(bytes.to_vec())).context("not a decodable GIF")?;
    let mut count = 0usize;
    let mut total_ms = 0f64;
    // The largest declared frame area seen, for the memory bound below.
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

    // The decoded-pixel budget, shared with the ffmpeg path's `scale` bound:
    // `max_frames` frames at `max_decoded_pixels` each. `no_truncate` classes
    // of omission are refused here rather than sampled around, because a GIF
    // that cannot fit the budget must say so, not quietly decode a prefix.
    let budget = policy
        .max_frames
        .max(1)
        .saturating_mul(policy.max_decoded_pixels.max(1));
    ensure!(
        count.saturating_mul(max_area) <= budget,
        "animated GIF would decode to {} frames of up to {max_area} pixels each, above the \
         {budget}-pixel decoded-image budget; shorten or shrink the clip",
        count
    );

    let native_fps = if total_ms > 0.0 {
        (count as f64 * 1000.0 / total_ms) as f32
    } else {
        DEFAULT_FPS
    };
    let fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        DEFAULT_FPS
    };
    let want = crate::video_sample::wanted_frames(
        (total_ms / 1000.0) as f32,
        native_fps,
        fps,
        policy.max_frames,
        fps,
    );
    let plan = crate::video_sample::sample_plan(want, (total_ms / 1000.0) as f32)?;

    // The plan's times are source seconds; a GIF has one frame per delay, so
    // the frame for time `t` is the one whose cumulative delay covers `t`.
    let mut cumulative = Vec::with_capacity(count);
    {
        let mut acc = 0f64;
        for frame in GifDecoder::new(std::io::Cursor::new(bytes.to_vec()))
            .context("not a decodable GIF")?
            .into_frames()
        {
            let frame = frame.context("failed to decode animation frames")?;
            cumulative.push(acc);
            let (num, den) = frame.delay().numer_denom_ms();
            if den != 0 {
                acc += num as f64 / den as f64;
            }
        }
    }
    let index_for = |t: f32| -> usize {
        let target = t as f64 * 1000.0;
        // Last frame whose start time is at or before `t`; clamped so the
        // first sample is frame 0 and nothing runs off the end.
        cumulative
            .partition_point(|&start| start <= target + 1e-6)
            .saturating_sub(1)
            .min(count - 1)
    };

    // Decode once, keeping only the frames the plan selected, in plan order.
    let wanted_indices: Vec<usize> = plan.times.iter().map(|&t| index_for(t)).collect();
    // A plan target can land inside a long single-frame delay. Do not retain
    // an extra full RGB clone for each repeated target; the source frame is
    // already the actual information available at that time.
    let mut selected_indices = Vec::with_capacity(wanted_indices.len());
    for index in wanted_indices {
        if selected_indices.last().copied() != Some(index) {
            selected_indices.push(index);
        }
    }
    let keep: std::collections::BTreeSet<usize> = selected_indices.iter().copied().collect();
    let mut chosen: Vec<Option<RgbImage>> = vec![None; count];
    for (i, frame) in GifDecoder::new(std::io::Cursor::new(bytes.to_vec()))
        .context("not a decodable GIF")?
        .into_frames()
        .enumerate()
    {
        if !keep.contains(&i) {
            continue;
        }
        let frame = frame.context("failed to decode animation frames")?;
        chosen[i] = Some(image::DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8());
    }

    let mut frames = Vec::with_capacity(selected_indices.len());
    let mut times = Vec::with_capacity(selected_indices.len());
    for i in selected_indices {
        if let Some(img) = chosen[i].take() {
            frames.push(img);
            times.push((cumulative[i] / 1000.0) as f32);
        }
    }
    ensure!(!frames.is_empty(), "the GIF sampled to zero frames");
    Ok((frames, times))
}
