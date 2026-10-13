// SPDX-License-Identifier: AGPL-3.0-only

//! Video → patch tensor, the temporal sibling of [`crate::vision_preprocess`].
//!
//! # What a video is, to this encoder
//!
//! GLM's visual tower has no temporal attention between groups. Frames fuse
//! inside a patch: the flattened patch dimension is
//! `C × temporal_patch_size × patch²`, so a patch spans `tp` consecutive
//! frames. A still image fills that axis by replicating itself `tp` times.
//!
//! A video of `n` sampled frames becomes `grid_t = n / tp` temporal groups,
//! each a full `grid_h × grid_w` patch plane. The native encoder runs each
//! group as its own attention sequence, then packs the resulting merger rows
//! into one contiguous video pad run. Keeping the grouping in the item is
//! still necessary for token insertion and MRoPE: T advances once per group.
//!
//! # Container support
//!
//! Two backends, chosen by MAGIC BYTES rather than the declared MIME:
//!
//! - **GIF** decodes in-process, pure Rust, always available, no dependency.
//! - **Everything else** (MP4/MOV, WebM/Matroska, AVI — H.264, H.265, VP9,
//!   AV1) goes to ffmpeg as a subprocess, which is opt-in.
//!
//! Sniffing the bytes rather than trusting the label means a client that
//! sends an mp4 as `video/gif`, or as `application/octet-stream`, still gets
//! the right decoder. See `video_decode_ffmpeg` for why a subprocess rather
//! than a linked decoder, and issue #515.

use anyhow::{Context, Result, ensure};
use atlas_core::config::VisionConfig;
use image::RgbImage;

use crate::vision_preprocess::{
    MEAN, STD, decode_data_uri_bytes, glm_resize_or_pad, glm_target_size, normalize_channel,
    patch_coordinates, target_size_for,
};

/// Frames per second to sample at, when the caller has no better idea.
/// Matches the `fps: 2` every Qwen3-VL `video_processor` block declares.
pub const DEFAULT_FPS: f32 = 2.0;

/// Sampling floor and ceiling, also from the checkpoints' own video processor
/// (`min_frames: 4`, `max_frames: 768`). The floor matters more than it looks:
/// with `temporal_patch_size = 2`, fewer than 2 frames cannot fill a single
/// temporal group, and a 1-frame "video" would silently become a still.
pub const DEFAULT_MIN_FRAMES: usize = 4;
pub const DEFAULT_MAX_FRAMES: usize = 768;

/// A decoded, ready-to-encode video.
pub struct PreprocessedVideo {
    /// One entry per temporal group, each shaped exactly like a preprocessed
    /// still: `[grid_h * grid_w, C * tp * patch * patch]`.
    pub groups: Vec<Vec<f32>>,
    pub grid_t: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    /// One timestamp per temporal group, in source-video seconds. GLM's
    /// prompt expansion places this text after each per-frame image marker;
    /// retaining it here keeps the prompt and encoder group order coupled.
    pub timestamps: Vec<f32>,
}

/// Summarised rather than derived: the payload is megabytes of f32 and a
/// derived `Debug` would dump all of it into any test failure or log line
/// that happens to format one.
impl std::fmt::Debug for PreprocessedVideo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PreprocessedVideo {{ grid_t: {}, grid_h: {}, grid_w: {}, groups: {} x {} f32, timestamps: {} }}",
            self.grid_t,
            self.grid_h,
            self.grid_w,
            self.groups.len(),
            self.groups.first().map_or(0, Vec::len),
            self.timestamps.len()
        )
    }
}

impl PreprocessedVideo {
    /// Merged tokens this video contributes: one per `merge × merge` block of
    /// patches, per temporal group.
    pub fn pad_count(&self, spatial_merge_size: usize) -> usize {
        let sms = spatial_merge_size.max(1);
        self.grid_t * (self.grid_h / sms) * (self.grid_w / sms)
    }
}

/// Pick which frame indices to keep so the clip plays at `fps`.
///
/// `native_fps` is what the container says it runs at. Sampling is by
/// NEAREST-INDEX over a uniform grid rather than by dropping every Nth frame:
/// the latter quantises badly when the ratio is not an integer (a 30fps clip
/// sampled at 2fps by "keep every 15th" is fine, at 2.5fps it is not).
///
/// The result is clamped into `[min_frames, max_frames]` and then to a
/// multiple of `temporal_patch_size`, because a partial group cannot be
/// encoded. Returns indices into the decoded frame list.
pub fn sample_indices(
    n_frames: usize,
    native_fps: f32,
    target_fps: f32,
    min_frames: usize,
    max_frames: usize,
    temporal_patch_size: usize,
) -> Vec<usize> {
    if n_frames == 0 {
        return Vec::new();
    }
    let tp = temporal_patch_size.max(1);
    let native_fps = if native_fps.is_finite() && native_fps > 0.0 {
        native_fps
    } else {
        DEFAULT_FPS
    };
    let target_fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        DEFAULT_FPS
    };

    let duration = n_frames as f32 / native_fps;
    let wanted = (duration * target_fps).round().max(1.0) as usize;

    // Clamp to the checkpoint's band, but never ask for more frames than
    // exist — upsampling a short clip by repeating frames would inflate the
    // token count with no new information.
    let wanted = wanted
        .clamp(min_frames.max(1), max_frames.max(1))
        .min(n_frames);

    // Round DOWN to a whole number of temporal groups; a partial group has no
    // representation. Never below one group, or there is nothing to encode.
    let wanted = (wanted / tp).max(1) * tp;
    let wanted = wanted.min((n_frames / tp).max(1) * tp).min(n_frames);

    if wanted >= n_frames {
        return (0..n_frames).collect();
    }
    // Uniform positions across the clip, nearest index, deduplicated in order.
    let mut out = Vec::with_capacity(wanted);
    for i in 0..wanted {
        let pos = if wanted == 1 {
            0.0
        } else {
            (i as f32) * ((n_frames - 1) as f32) / ((wanted - 1) as f32)
        };
        out.push((pos.round() as usize).min(n_frames - 1));
    }
    out
}

/// GLM's processor uses an `fps_interval` timestamp walk followed by a
/// linspace repair when the greedy walk misses its requested frame count.
///
/// Kept for the grouping contract's arithmetic above; the decode path now
/// samples through [`crate::video_sample`] and only rounds to groups here.
fn _legacy_sampling_reference() {}

/// The only re-sampling left after the decoder has done the coverage work:
/// drop any trailing partial temporal group.
///
/// The decoder ([`crate::video_sample`]) already chose which source frames to
/// return, at the operator's rate and spanning the clip. A partial group
/// cannot be encoded as video — the patch tensor's temporal axis is exactly
/// `temporal_patch_size` frames — so a remainder is dropped here rather than
/// padded. Frames are never duplicated to fill a group, which would charge
/// the model for motion it never saw.
pub fn group_indices(n_frames: usize, temporal_patch_size: usize) -> Vec<usize> {
    let tp = temporal_patch_size.max(1);
    let whole = n_frames / tp * tp;
    // A clip with fewer frames than one group keeps them all: refusing it is
    // the caller's job (the `temporal group` error in `preprocess_video`),
    // and reporting nothing here would hide why it was refused.
    if whole == 0 {
        return (0..n_frames).collect();
    }
    (0..whole).collect()
}

/// Decode every frame of a container, choosing a backend by what the bytes
/// actually are.
///
/// Returns the frames, with one SOURCE time per frame. Both backends now go
/// through the same coverage rule: the plan spans the whole clip at no more
/// than `target_fps`, and the times it returns say where each returned frame
/// actually sits. GIF decodes in-process (pure Rust, no dependency); every
/// other container goes to ffmpeg.
///
/// Dispatch is on MAGIC BYTES, not the declared MIME. A client that labels an
/// mp4 `video/gif`, or sends `application/octet-stream`, still gets the right
/// decoder; and a GIF mislabelled as mp4 does not needlessly spawn a process.
pub fn decode_frames(
    data_uri: &str,
    target_fps: f32,
    ffmpeg: &crate::video_decode_ffmpeg::FfmpegPolicy,
) -> Result<(Vec<RgbImage>, Vec<f32>)> {
    let (mime, bytes) = decode_data_uri_bytes(data_uri)?;
    ensure!(!bytes.is_empty(), "the video payload is empty");

    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        return decode_gif(&bytes, target_fps, ffmpeg);
    }

    // Everything else goes to the subprocess backend. If it is disabled the
    // error names the flag AND the container, so the operator is not left
    // guessing which of the two problems they have.
    let kind = sniff_container(&bytes, &mime);
    crate::video_decode_ffmpeg::decode_with_plan(&bytes, target_fps, ffmpeg)
        .with_context(|| format!("decoding {kind}"))
}

/// Best-effort container name for error messages. Cosmetic only — nothing
/// branches on it — so an unrecognized blob is described as such rather than
/// guessed at.
fn sniff_container(bytes: &[u8], mime: &str) -> String {
    let by_magic = if bytes.len() > 12 && &bytes[4..8] == b"ftyp" {
        Some("an MP4/MOV container")
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Some("a Matroska/WebM container")
    } else if bytes.starts_with(b"RIFF") {
        Some("an AVI container")
    } else {
        None
    };
    match (by_magic, mime.is_empty()) {
        (Some(k), _) => k.to_string(),
        (None, false) => format!("a {mime} payload"),
        (None, true) => "an unrecognized container".to_string(),
    }
}

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
fn decode_gif(
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
    let keep: std::collections::BTreeSet<usize> = wanted_indices.iter().copied().collect();
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

    let mut frames = Vec::with_capacity(wanted_indices.len());
    let mut times = Vec::with_capacity(wanted_indices.len());
    for (t, &i) in plan.times.iter().zip(&wanted_indices) {
        if let Some(img) = chosen[i].take() {
            frames.push(img);
            times.push(*t);
        }
    }
    ensure!(!frames.is_empty(), "the GIF sampled to zero frames");
    Ok((frames, times))
}

/// Full pipeline: a base64 `data:` URI holding an animated container becomes
/// temporal groups of patches.
pub fn preprocess_video(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    target_fps: f32,
    ffmpeg: &crate::video_decode_ffmpeg::FfmpegPolicy,
) -> Result<PreprocessedVideo> {
    ensure!(
        vcfg.patch_size > 0 && vcfg.spatial_merge_size > 0 && vcfg.temporal_patch_size > 0,
        "vision_config geometry is invalid (patch/merge/temporal size is 0)"
    );
    let (frames, frame_times) = decode_frames(data_uri, target_fps, ffmpeg)?;
    let tp = vcfg.temporal_patch_size;

    // The decoder has already sampled the clip: its frames ARE the plan, at
    // the operator's rate and spanning the whole clip. Re-sampling here would
    // be a second, lossy pass over positions the decoder chose deliberately
    // (and, for a GIF, would need a rate the container may not declare).
    // The one thing still owed is the group shape: a partial temporal group
    // has no representation, so the count is rounded down to whole groups.
    let keep = group_indices(frames.len(), tp);
    ensure!(!keep.is_empty(), "frame sampling selected no frames");

    // A clip shorter than one temporal group cannot be encoded as video.
    // Saying so beats silently padding it into a still, which would report a
    // plausible token count for something the model never saw as motion.
    ensure!(
        keep.len() >= tp,
        "video has {} usable frame(s) but temporal_patch_size is {tp}; a clip must carry at \
         least one full temporal group",
        keep.len()
    );

    // Geometry is decided ONCE, from the first kept frame, and applied to all
    // of them. Per-frame sizing would be a correctness bug rather than a
    // refinement: the groups are concatenated into one pad run whose token
    // count assumes a single grid.
    let first = &frames[keep[0]];
    let grid_unit = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (th, tw) = if vcfg.is_glm5_next {
        glm_target_size(
            vcfg,
            first.height(),
            first.width(),
            keep.len(),
            max_pixels,
            true,
        )?
    } else {
        target_size_for(first.height(), first.width(), grid_unit, max_pixels)
    };
    let glm_min_pixels = 16usize
        * tp
        * (vcfg.patch_size * vcfg.spatial_merge_size)
            .saturating_mul(vcfg.patch_size * vcfg.spatial_merge_size);
    let glm_allow_upscale = vcfg.is_glm5_next
        && keep.len() * first.height() as usize * (first.width() as usize) < glm_min_pixels;

    let ps = vcfg.patch_size;
    let grid_h = (th as usize) / ps;
    let grid_w = (tw as usize) / ps;
    let grid_t = keep.len() / tp;
    let patch_dim = 3 * tp * ps * ps;
    let plane = grid_h * grid_w;

    let mut groups = Vec::with_capacity(grid_t);
    for g in 0..grid_t {
        // Resize this group's `tp` frames once each, up front: the patch loop
        // reads every pixel of every frame, so resizing inside it would redo
        // the work `patch²` times.
        let resized: Vec<RgbImage> = (0..tp)
            .map(|k| {
                let f = &frames[keep[g * tp + k]];
                if vcfg.is_glm5_next {
                    glm_resize_or_pad(f, th, tw, glm_allow_upscale)
                } else {
                    image::imageops::resize(f, tw, th, image::imageops::FilterType::CatmullRom)
                }
            })
            .collect();

        let mut pixels = vec![0.0f32; plane * patch_dim];
        for (patch_idx, (ph, pw)) in patch_coordinates(vcfg, grid_h, grid_w)
            .into_iter()
            .enumerate()
        {
            for c in 0..3usize {
                for (t, frame) in resized.iter().enumerate() {
                    for py in 0..ps {
                        for px in 0..ps {
                            let raw = frame.get_pixel((pw * ps + px) as u32, (ph * ps + py) as u32)
                                [c] as f32
                                / 255.0;
                            let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                            pixels[patch_idx * patch_dim + off] = if vcfg.is_glm5_next {
                                normalize_channel(
                                    vcfg,
                                    c,
                                    frame.get_pixel((pw * ps + px) as u32, (ph * ps + py) as u32)
                                        [c],
                                )
                            } else {
                                (raw - MEAN[c]) / STD[c]
                            };
                        }
                    }
                }
            }
        }
        groups.push(pixels);
    }

    Ok(PreprocessedVideo {
        groups,
        grid_t,
        grid_h,
        grid_w,
        // The pinned GLM processor's non-Glm4v path formats full-second
        // timestamps (`"{:.1f} seconds"`) and takes one timestamp per
        // temporal pair. The decoder's per-frame source times are the SSOT
        // for where each group sits — an assumed constant rate would drift
        // from the positions actually sampled.
        timestamps: (0..grid_t)
            .map(|g| frame_times[keep[g * tp]].floor())
            .collect(),
    })
}

#[cfg(test)]
#[path = "video_preprocess_tests.rs"]
mod tests;
