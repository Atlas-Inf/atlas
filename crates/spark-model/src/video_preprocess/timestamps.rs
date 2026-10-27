// SPDX-License-Identifier: AGPL-3.0-only

//! Per-temporal-group timestamps for a sampled video, two rules kept apart.
//!
//! The decoder's per-frame SOURCE times are the SSOT for where each group
//! sits — an assumed constant rate would drift from the positions actually
//! sampled. `frame_times[i]` is the source-video second of decoded frame `i`;
//! `keep` maps each kept position onto that list.

/// GLM's rule: the floor of the first kept frame's time in each group. The
/// pinned GLM processor formats these `"{:.1f} seconds"` and takes one per
/// temporal pair.
pub(super) fn glm_group_timestamps(
    grid_t: usize,
    temporal_patch_size: usize,
    keep: &[usize],
    frame_times: &[f32],
) -> Vec<f32> {
    let tp = temporal_patch_size.max(1);
    (0..grid_t)
        .map(|g| frame_times[keep[g * tp]].floor())
        .collect()
}

/// HF Qwen3VL's rule (`replace_video_token`/`_calculate_timestamps`): the
/// mean of the first and last kept frame's source time within each group. A
/// short final group repeats the last kept frame's time, as HF extends the
/// index list.
pub(super) fn hf_group_timestamps(
    grid_t: usize,
    temporal_patch_size: usize,
    keep: &[usize],
    frame_times: &[f32],
) -> Vec<f32> {
    let tp = temporal_patch_size.max(1);
    (0..grid_t)
        .map(|g| {
            let first = keep[g * tp];
            let last = keep[(g * tp + tp - 1).min(keep.len() - 1)];
            (frame_times[first] + frame_times[last]) / 2.0
        })
        .collect()
}
