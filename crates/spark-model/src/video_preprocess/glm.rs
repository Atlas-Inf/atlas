// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 video as the reference serves it: vLLM's `glm5next` loader picks
//! source frames with `glm_sample_frame_indices` and reads them at their
//! native size, then `Glm5NextVideoProcessor` sizes one canvas for all of
//! them with `smart_resize`, pads each frame onto it, and repeats the last
//! frame to fill the final temporal pair.

use anyhow::{Result, ensure};
use atlas_core::config::VisionConfig;
use image::RgbImage;

use super::PreprocessedVideo;
use super::gif_access::{gif_frames, gif_timeline};
use super::patchify::patchify_group;
use crate::video_decode_ffmpeg::{FfmpegPolicy, decode_frame_indices, probe_frames};
use crate::video_sample::{finite_positive, glm_sample_frame_indices};
use crate::vision_preprocess::{
    DECODE_MAX_ALLOC, decode_data_uri_bytes, glm_pixel_budget, glm_resize_or_pad, glm_target_size,
};

pub(super) fn preprocess_glm_video(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    target_fps: f32,
    ffmpeg: &FfmpegPolicy,
) -> Result<PreprocessedVideo> {
    let (_, bytes) = decode_data_uri_bytes(data_uri)?;
    ensure!(!bytes.is_empty(), "the video payload is empty");
    let gif = bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a");
    let (total, fps) = if gif {
        let (count, total_ms, max_area) = gif_timeline(&bytes)?;
        ensure!(
            max_area as u64 * 3 <= DECODE_MAX_ALLOC,
            "animated GIF frames of {max_area} pixels exceed the decoded-image limit"
        );
        let fps = if total_ms > 0.0 {
            count as f64 * 1000.0 / total_ms
        } else {
            f64::from(super::DEFAULT_FPS)
        };
        (count, fps)
    } else {
        probe_frames(&bytes, ffmpeg)?
    };
    ensure!(
        fps.is_finite() && fps > 0.0,
        "the clip declares no frame rate to sample by"
    );
    let tp = vcfg.temporal_patch_size;
    // The loader decodes each sampled frame once; the pair-filling repeat of
    // the last index is restored after decoding, as the processor pads.
    let mut indices = glm_sample_frame_indices(
        total,
        fps,
        total as f64 / fps,
        finite_positive(target_fps).map(f64::from),
        Some(ffmpeg.max_frames),
        tp,
    );
    indices.dedup();
    ensure!(
        !indices.is_empty(),
        "a {:.2} s clip is too short to sample at {target_fps} fps",
        total as f64 / fps
    );

    // One canvas for every frame, from the first frame's size and the sampled
    // count: the processor resizes the clip as one `t × h × w` volume.
    let canvas = |(h, w): (u32, u32), frames: usize| -> Result<(u32, u32, bool)> {
        let (th, tw) = glm_target_size(vcfg, h, w, frames, max_pixels, true)?;
        let min_pixels = glm_pixel_budget(vcfg, max_pixels, true).0;
        if let Some(budget) = ffmpeg.max_total_decoded_pixels {
            let volume = frames
                .div_ceil(tp)
                .saturating_mul(tp)
                .saturating_mul(th as usize)
                .saturating_mul(tw as usize);
            ensure!(
                volume <= budget,
                "video grid needs {volume} pixels, above the {budget}-pixel volume bound"
            );
        }
        Ok((th, tw, frames * h as usize * (w as usize) < min_pixels))
    };
    let wanted = indices.len();
    let mut first = None;
    let mut frames = Vec::with_capacity(wanted.div_ceil(tp) * tp);
    let mut on_frame = |frame: RgbImage| -> Result<()> {
        let size = (frame.height(), frame.width());
        if first.is_none() {
            first = Some((size, canvas(size, wanted)?));
        }
        let (_, (th, tw, upscale)) = first.expect("set above");
        frames.push(glm_resize_or_pad(&frame, th, tw, upscale));
        Ok(())
    };
    let got = if gif {
        gif_frames(&bytes, &indices, &mut on_frame)?
    } else {
        decode_frame_indices(&bytes, &indices, ffmpeg, &mut on_frame)?
    };
    let Some((size, geometry)) = first else {
        anyhow::bail!("the clip decoded to zero frames");
    };
    // A container that overstates its frames returns fewer; the reference
    // then sizes the canvas for the frames it got. Keep them only if that is
    // the canvas they were already resized onto.
    if got < wanted {
        ensure!(
            canvas(size, got)? == geometry,
            "the decoder returned {got} of {wanted} sampled frames"
        );
        indices.truncate(got);
    }
    while !frames.len().is_multiple_of(tp) {
        frames.push(frames[frames.len() - 1].clone());
        indices.push(indices[indices.len() - 1]);
    }

    let (th, tw, _) = geometry;
    let ps = vcfg.patch_size;
    let (grid_h, grid_w) = (th as usize / ps, tw as usize / ps);
    let groups: Vec<Vec<f32>> = frames
        .chunks(tp)
        .map(|group| patchify_group(vcfg, group, grid_h, grid_w))
        .collect();
    Ok(PreprocessedVideo {
        grid_t: groups.len(),
        groups,
        grid_h,
        grid_w,
        // Each group's first frame, in whole source seconds: `int(idx / fps)`.
        timestamps_hf: {
            // GLM's rule is the floor of each group's first frame; the HF
            // mean-of-pair rule is not read for GLM, so mirror it.
            indices
                .iter()
                .step_by(tp)
                .map(|&index| (index as f64 / fps).trunc() as f32)
                .collect()
        },
        timestamps: indices
            .iter()
            .step_by(tp)
            .map(|&index| (index as f64 / fps).trunc() as f32)
            .collect(),
    })
}
