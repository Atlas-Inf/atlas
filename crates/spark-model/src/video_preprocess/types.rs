// SPDX-License-Identifier: AGPL-3.0-only

//! The video payload type and the frame-index arithmetic — pure data and
//! index math, no decode.

use super::DEFAULT_FPS;

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
