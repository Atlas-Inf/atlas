// SPDX-License-Identifier: AGPL-3.0-only

//! Ling 3.2 (Bailing MM2) video sampling and frame size.
//!
//! The checkpoint's processor (`bailingmm_utils_video.v1_fetch_video`, the
//! default `VIDEO_FETCH_VERSION`) samples at most `fps` frames per second and
//! at most 32 frames in all, evenly spaced over the clip, and times them at
//! `i / sample_fps` from the start. Every frame gets one size, chosen so the
//! sampled clip stays near 4,718,592 pixels: the per-frame bound is
//! `max(total / frames × 2, 1.05 × 43,008)` with a 43,008-pixel floor, on a
//! 32-pixel grid. An odd count repeats the last frame into the final temporal
//! group, which [`super::preprocess_video`] already does.

use image::RgbImage;

/// The processor's `FPS_MAX_FRAMES`, the default sampled-frame cap.
pub const DEFAULT_MAX_FRAMES: usize = 32;
/// `VIDEO_TOTAL_PIXELS`: the pixel volume of all sampled frames.
pub const TOTAL_PIXELS: usize = 4_718_592;
/// `VIDEO_MIN_PIXELS`: the per-frame floor.
const MIN_PIXELS: f64 = 43_008.0;
/// `FRAME_FACTOR`: frames per temporal group, and the frame-count multiple.
const FRAME_FACTOR: usize = 2;
/// `IMAGE_FACTOR`: patch 16 × merge 2.
const GRID: f64 = 32.0;

/// Keep the frames `v1_smart_nframes` and `v1_sample_frames` would keep from
/// a decoder that sampled `frames` at the source times `times`, and return
/// them with the reference's timestamps.
///
/// The decoder hands over every source frame of a short clip and an even
/// spread of a long one, so the source duration is the span of `times` plus
/// one frame interval.
pub fn select(
    frames: Vec<RgbImage>,
    times: &[f32],
    fps: f32,
    max_frames: usize,
) -> (Vec<RgbImage>, Vec<f32>) {
    let total = frames.len();
    if total == 0 {
        return (frames, Vec::new());
    }
    let duration = clip_duration(times);
    let n = frame_count(duration, total, fps as f64, max_frames);
    let picks = linspace_indices(total, n);
    let sample_fps = n as f64 / duration;
    let timestamps = (0..n).map(|i| (i as f64 / sample_fps) as f32).collect();
    let mut frames: Vec<Option<RgbImage>> = frames.into_iter().map(Some).collect();
    let mut kept = Vec::with_capacity(n);
    let mut last = None;
    for &idx in &picks {
        // `linspace` can repeat an index when the clip is shorter than `n`;
        // the repeat reuses the frame already taken.
        if last == Some(idx) {
            let previous: &RgbImage = kept.last().expect("a repeat follows a kept frame");
            kept.push(previous.clone());
        } else {
            kept.push(frames[idx].take().expect("indices are non-decreasing"));
        }
        last = Some(idx);
    }
    (kept, timestamps)
}

fn clip_duration(times: &[f32]) -> f64 {
    match times {
        [] => 1.0,
        [_] => 1.0,
        [first, .., last] => {
            let span = (*last - *first) as f64;
            let step = span / (times.len() - 1) as f64;
            if span > 0.0 { span + step } else { 1.0 }
        }
    }
}

/// `v1_smart_nframes` for the `fps` branch.
pub fn frame_count(duration: f64, total: usize, fps: f64, max_frames: usize) -> usize {
    let cap = ((max_frames.min(total) / FRAME_FACTOR) * FRAME_FACTOR).max(1);
    // The duration comes back from f32 frame times: a whole second of 30 fps
    // reads 0.99999994 s, and the reference's exact 2.0 frames must not
    // truncate to 1.
    let wanted = (duration * fps + 1e-4).max(1.0);
    (wanted.min(cap as f64).min(total as f64) as usize).max(1)
}

/// `np.linspace(0, total - 1, n, dtype=int)`.
fn linspace_indices(total: usize, n: usize) -> Vec<usize> {
    if n <= 1 {
        return vec![0];
    }
    let last = (total - 1) as f64;
    (0..n)
        .map(|i| ((i as f64) * last / (n - 1) as f64) as usize)
        .collect()
}

/// The frame size `smart_resize` picks for a clip of `frames` sampled frames.
/// `total_pixels` is the operator's `--video-max-pixels`, else
/// [`TOTAL_PIXELS`].
pub fn frame_size(height: u32, width: u32, frames: usize, total_pixels: usize) -> (u32, u32) {
    let max_pixels =
        (total_pixels as f64 / frames.max(1) as f64 * FRAME_FACTOR as f64).max(MIN_PIXELS * 1.05);
    let (h, w) = (height as f64, width as f64);
    let round = |x: f64| ((x / GRID).round_ties_even() * GRID).max(GRID);
    let (mut h_bar, mut w_bar) = (round(h), round(w));
    if h_bar * w_bar > max_pixels {
        let beta = (h * w / max_pixels).sqrt();
        h_bar = ((h / beta / GRID).floor() * GRID).max(GRID);
        w_bar = ((w / beta / GRID).floor() * GRID).max(GRID);
    } else if h_bar * w_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS / (h * w)).sqrt();
        h_bar = (h * beta / GRID).ceil() * GRID;
        w_bar = (w * beta / GRID).ceil() * GRID;
    }
    (h_bar as u32, w_bar as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(n: usize) -> Vec<RgbImage> {
        (0..n)
            .map(|i| RgbImage::from_pixel(4, 4, image::Rgb([i as u8, 0, 0])))
            .collect()
    }

    #[test]
    fn a_short_clip_at_the_source_rate_is_sampled_at_two_per_second() {
        // One second of 30 fps: the reference keeps 2 frames, not 30.
        let times: Vec<f32> = (0..30).map(|i| i as f32 / 30.0).collect();
        let (kept, ts) = select(frames(30), &times, 2.0, DEFAULT_MAX_FRAMES);
        assert_eq!(kept.len(), 2);
        assert_eq!(kept[0].get_pixel(0, 0)[0], 0);
        assert_eq!(kept[1].get_pixel(0, 0)[0], 29);
        assert_eq!(ts, vec![0.0, 0.5]);
    }

    #[test]
    fn a_long_clip_is_capped_at_thirty_two_even_frames() {
        // 60 s at 30 fps, as every source frame.
        let times: Vec<f32> = (0..1800).map(|i| i as f32 / 30.0).collect();
        let (kept, ts) = select(frames(1800), &times, 2.0, DEFAULT_MAX_FRAMES);
        assert_eq!(kept.len(), 32);
        assert_eq!(ts.len(), 32);
        // i / (32 / 60 s)
        assert!((ts[1] - 1.875).abs() < 1e-4, "{}", ts[1]);
        assert!((ts[31] - 58.125).abs() < 1e-3, "{}", ts[31]);
    }

    #[test]
    fn frame_count_matches_v1_smart_nframes() {
        // min(duration × fps, floor2(min(cap, total)), total), truncated.
        assert_eq!(frame_count(10.0, 300, 2.0, 32), 20);
        assert_eq!(frame_count(3.3, 99, 2.0, 32), 6);
        assert_eq!(frame_count(100.0, 3000, 2.0, 32), 32);
        assert_eq!(frame_count(0.2, 6, 2.0, 32), 1);
        assert_eq!(frame_count(10.0, 7, 2.0, 32), 6);
        assert_eq!(frame_count(100.0, 3000, 2.0, 129), 128);
    }

    #[test]
    fn frame_size_matches_smart_resize() {
        // 32 frames of 1280×720: bound 294,912 px per frame.
        assert_eq!(frame_size(720, 1280, 32, TOTAL_PIXELS), (384, 704));
        // 2 frames of 640×480 fit unchanged on the 32-pixel grid.
        assert_eq!(frame_size(480, 640, 2, TOTAL_PIXELS), (480, 640));
        // A tiny frame is raised to the 43,008-pixel floor.
        assert_eq!(frame_size(64, 64, 2, TOTAL_PIXELS), (224, 224));
    }
}
