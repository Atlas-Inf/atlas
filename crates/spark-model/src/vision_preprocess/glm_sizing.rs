// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 vision geometry: the serving processor's token budgets, the
//! native encoder capacity they imply, the token-budget `smart_resize` port
//! and resize-or-pad.

use super::*;

/// The upstream serving processor caps the checkpoint's 240k video tokens at
/// 30k (`_MAX_VIDEO_TOKENS`) to keep encoder profiling and KV-cache admission
/// bounded; stills keep the checkpoint's budget verbatim.
pub(crate) const GLM_MAX_VIDEO_TOKENS: usize = 30_000;

/// Pixels of the processor's `t × h × w` volume one vision token covers: one
/// temporal group of one `merge × merge` patch block.
fn glm_token_pixels(vcfg: &VisionConfig) -> usize {
    let cell = vcfg.patch_size * vcfg.spatial_merge_size;
    vcfg.temporal_patch_size.saturating_mul(cell * cell)
}

/// `(min_pixels, max_pixels)` of the processor's `t × h × w` volume: the
/// checkpoint's token budgets (`_pixel_budget`), the video one under the
/// serving cap. An operator `max_pixels` replaces the upper bound of both.
pub(crate) fn glm_pixel_budget(
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    video: bool,
) -> (usize, usize) {
    let (min_tokens, max_tokens) = if video {
        let (min, max) = vcfg.token_budgets.video;
        (min, max.min(GLM_MAX_VIDEO_TOKENS))
    } else {
        vcfg.token_budgets.image
    };
    let unit = glm_token_pixels(vcfg);
    let max_pixels = max_pixels
        .filter(|&value| value > 0)
        .unwrap_or_else(|| max_tokens.saturating_mul(unit));
    (min_tokens.saturating_mul(unit), max_pixels)
}

/// Pre-merge patches in the largest attention sequence the processor can
/// emit: one still, or one temporal group of a video. `smart_resize` bounds
/// `t_bar × h × w` with `t_bar >= temporal_patch_size`, so no sequence holds
/// more than `max_pixels / (temporal_patch_size × patch²)` patches; the native
/// tower sizes its per-sequence buffers from this.
#[cfg(test)]
pub(crate) fn glm_max_sequence_patches(vcfg: &VisionConfig) -> usize {
    let patch_volume = vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size;
    let image = glm_pixel_budget(vcfg, vcfg.max_pixels, false).1;
    let video = glm_pixel_budget(vcfg, vcfg.max_pixels, true).1;
    (image.max(video) / patch_volume.max(1)).max(1)
}

/// Vision tokens in the largest single item. A still holds at most its
/// budget. A video's budget is charged against `t_bar`, which rounds the
/// frame count to whole temporal groups half-to-even, while the frames are
/// padded up to `grid_t` groups: five frames budget two groups and encode
/// three, so a video can carry half again its token budget.
#[cfg(test)]
pub(crate) fn glm_max_item_tokens(vcfg: &VisionConfig) -> usize {
    let tp = vcfg.temporal_patch_size.max(1);
    let cell = vcfg.patch_size * vcfg.spatial_merge_size;
    let image = glm_pixel_budget(vcfg, vcfg.max_pixels, false).1 / glm_token_pixels(vcfg).max(1);
    let video_pixels = glm_pixel_budget(vcfg, vcfg.max_pixels, true).1;
    let video = (1..=crate::video_sample::MAX_SAMPLE_PLAN)
        .map(|frames| {
            let per_group = video_pixels / (glm_temporal_extent(frames, tp) * cell * cell).max(1);
            frames.div_ceil(tp) * per_group
        })
        .max()
        .unwrap_or(0);
    image.max(video)
}

/// `smart_resize`'s `t_bar`: the frame count rounded to whole temporal
/// groups, never below one. Python's `round` ties to even, so five frames
/// budget two groups, not three.
fn glm_temporal_extent(frames: usize, tp: usize) -> usize {
    tp.max((frames as f64 / tp as f64).round_ties_even() as usize * tp)
}

/// Port of GLM-5.3's token-budget `smart_resize`: aligned dimensions are
/// rounded upward, then proportionally refit by binary search when the
/// temporal pixel budget is exceeded. The generic Qwen path intentionally
/// keeps its historical nearest-grid behavior.
pub(crate) fn glm_target_size(
    vcfg: &VisionConfig,
    orig_h: u32,
    orig_w: u32,
    temporal_len: usize,
    requested_max_pixels: Option<usize>,
    video: bool,
) -> Result<(u32, u32)> {
    let factor = vcfg.patch_size * vcfg.spatial_merge_size;
    let (min_pixels, max_pixels) = glm_pixel_budget(vcfg, requested_max_pixels, video);
    ensure!(
        temporal_len > 0 && orig_h > 0 && orig_w > 0,
        "GLM vision input has an empty dimension"
    );
    ensure!(
        min_pixels <= max_pixels,
        "GLM vision max_pixels {max_pixels} is below the {min_pixels}-pixel minimum budget"
    );
    let t_bar = glm_temporal_extent(temporal_len, vcfg.temporal_patch_size.max(1));
    let (h, w) = (orig_h as usize, orig_w as usize);
    let volume = |hh: usize, ww: usize| t_bar.saturating_mul(hh).saturating_mul(ww);
    let mut h_bar = h.div_ceil(factor) * factor;
    let mut w_bar = w.div_ceil(factor) * factor;
    if volume(h_bar, w_bar) > max_pixels {
        (h_bar, w_bar) = glm_fit_within_budget(t_bar, h, w, factor, max_pixels)?;
    } else if volume(h_bar, w_bar) < min_pixels {
        // `t`, not `t_bar`: the reference scales by the frames it was given.
        let beta = (min_pixels as f64 / (temporal_len * h * w) as f64).sqrt();
        h_bar = ((h as f64 * beta).ceil() as usize).max(1).div_ceil(factor) * factor;
        w_bar = ((w as f64 * beta).ceil() as usize).max(1).div_ceil(factor) * factor;
        if volume(h_bar, w_bar) > max_pixels {
            (h_bar, w_bar) = glm_fit_within_budget(t_bar, h, w, factor, max_pixels)?;
        }
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// `_fit_aligned_size_within_budget`: the largest proportional size whose
/// upward-aligned canvas fits `t × h × w <= max_pixels`, by binary search on
/// the unaligned content height.
fn glm_fit_within_budget(
    t: usize,
    h: usize,
    w: usize,
    factor: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    ensure!(
        max_pixels >= t * factor * factor,
        "GLM vision max_pixels {max_pixels} cannot hold one aligned {factor}x{factor} cell"
    );
    let (mut low, mut high) = (1usize, h);
    let mut best = (factor, factor);
    while low <= high {
        let content_h = (low + high) / 2;
        let content_w = (w * content_h / h).max(1);
        let aligned = (
            content_h.div_ceil(factor) * factor,
            content_w.div_ceil(factor) * factor,
        );
        if t * aligned.0 * aligned.1 <= max_pixels {
            best = aligned;
            low = content_h + 1;
        } else {
            high = content_h - 1;
        }
    }
    Ok(best)
}

/// Preserve GLM's default `resize_mode="pad"`: resize the image to fit the
/// aligned canvas and fill the right/bottom remainder with black pixels.
pub(crate) fn glm_resize_or_pad(
    img: &RgbImage,
    target_h: u32,
    target_w: u32,
    allow_upscale: bool,
) -> RgbImage {
    let scale = (target_h as f32 / img.height() as f32)
        .min(target_w as f32 / img.width() as f32)
        .min(if allow_upscale { f32::INFINITY } else { 1.0 });
    let content_h = ((img.height() as f32 * scale).floor() as u32).clamp(1, target_h);
    let content_w = ((img.width() as f32 * scale).floor() as u32).clamp(1, target_w);
    let resized = image::imageops::resize(
        img,
        content_w,
        content_h,
        image::imageops::FilterType::CatmullRom,
    );
    let mut canvas = RgbImage::from_pixel(target_w, target_h, Rgb([0, 0, 0]));
    image::imageops::replace(&mut canvas, &resized, 0, 0);
    canvas
}

/// preprocessing must use the same bounded capacity instead of the processor's
/// larger checkpoint token budget.
pub(crate) const GLM_CEILING_MAX_PATCHES: usize = 16_384;

/// Convert the resolved GLM `max_pixels` setting into the capacity used by the
/// native tower. For GLM this setting follows the processor's `t*h*w` pixel
/// volume: still images use `t = temporal_patch_size`, while videos use the
/// sampled frame count. The native buffers remain bounded even when the
/// checkpoint advertises a larger processor budget.
const GLM_FALLBACK_MAX_PATCHES: usize = 6_400;

pub(crate) fn derive_glm_max_patches(
    max_pixels: Option<usize>,
    patch_size: usize,
    temporal_patch_size: usize,
) -> (usize, Option<usize>) {
    let Some(volume) = max_pixels.filter(|&value| value > 0) else {
        return (GLM_FALLBACK_MAX_PATCHES, None);
    };
    let patch_volume = temporal_patch_size
        .max(1)
        .saturating_mul(patch_size.max(1).saturating_mul(patch_size.max(1)));
    let wanted = (volume / patch_volume).max(1);
    if wanted > GLM_CEILING_MAX_PATCHES {
        (GLM_CEILING_MAX_PATCHES, Some(wanted))
    } else {
        (wanted, None)
    }
}
