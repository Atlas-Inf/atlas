// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 vision geometry: token budgets, native encoder patch capacity,
//! the token-budget `smart_resize` port and resize-or-pad.

use super::*;

/// The pinned GLM processor expresses its geometry as token budgets.  A
/// vision token covers one temporal group and one 2×2 patch merge.
const GLM_MIN_IMAGE_TOKENS: usize = 16;
const GLM_MAX_IMAGE_TOKENS: usize = 8_000;
/// The upstream serving processor caps its checkpoint-declared 240k video
/// tokens at 30k to keep encoder profiling and KV-cache admission bounded.
const GLM_MAX_VIDEO_TOKENS: usize = 30_000;

/// Native GLM encoder capacity, in pre-merge spatial patches or merged output
/// rows. The quadratic attention scratch is allocated once at construction, so
/// preprocessing must use the same bounded capacity instead of the processor's
/// larger checkpoint token budget.
pub(crate) const GLM_FALLBACK_MAX_PATCHES: usize = 6_400;
pub(crate) const GLM_CEILING_MAX_PATCHES: usize = 16_384;

/// Convert the resolved GLM `max_pixels` setting into the capacity used by the
/// native tower. For GLM this setting follows the processor's `t*h*w` pixel
/// volume: still images use `t = temporal_patch_size`, while videos use the
/// sampled frame count. The native buffers remain bounded even when the
/// checkpoint advertises a larger processor budget.
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

/// Effective GLM pixel-volume budget after applying the fixed native capacity.
/// A video has one merged output row per 2×2 spatial patch for every temporal
/// group, so it gets the extra `merge²` factor; each individual sequence still
/// fits the same `p_max` input-patch allocation.
pub(crate) fn glm_runtime_max_pixels(
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    video: bool,
) -> usize {
    let (p_max, _) = derive_glm_max_patches(max_pixels, vcfg.patch_size, vcfg.temporal_patch_size);
    let patch_volume = vcfg.temporal_patch_size.max(1).saturating_mul(
        vcfg.patch_size
            .max(1)
            .saturating_mul(vcfg.patch_size.max(1)),
    );
    let merge_factor = if video {
        vcfg.spatial_merge_size
            .max(1)
            .saturating_mul(vcfg.spatial_merge_size.max(1))
    } else {
        1
    };
    let capacity = p_max
        .saturating_mul(patch_volume)
        .saturating_mul(merge_factor);
    max_pixels
        .filter(|&value| value > 0)
        .map_or(capacity, |value| value.min(capacity))
}

pub(super) fn glm_pixel_budget(
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
    video: bool,
) -> (usize, usize) {
    let factor = vcfg.temporal_patch_size
        * (vcfg.patch_size * vcfg.spatial_merge_size)
            .saturating_mul(vcfg.patch_size * vcfg.spatial_merge_size);
    let min_pixels = GLM_MIN_IMAGE_TOKENS.saturating_mul(factor);
    let default_max_tokens = if video {
        GLM_MAX_VIDEO_TOKENS
    } else {
        GLM_MAX_IMAGE_TOKENS
    };
    // Keep the canonical token budget as the requested upper bound, but cap
    // it to the actual native encoder capacity. The default is also bounded;
    // `None` means no operator/checkpoint override, not an unbounded GLM run.
    let processor_max = max_pixels.unwrap_or_else(|| default_max_tokens.saturating_mul(factor));
    let capacity_max = glm_runtime_max_pixels(vcfg, max_pixels, video);
    let max_pixels = processor_max.min(capacity_max);
    (min_pixels, max_pixels)
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
    let factor = (vcfg.patch_size * vcfg.spatial_merge_size) as u32;
    let (min_pixels, mut max_pixels) = glm_pixel_budget(vcfg, requested_max_pixels, video);
    ensure!(
        max_pixels >= min_pixels,
        "GLM vision max_pixels is below one image token"
    );
    let t_factor = vcfg.temporal_patch_size.max(1);
    let t_bar = t_factor
        .max(((temporal_len as f64 / t_factor as f64).round() as usize).saturating_mul(t_factor));
    if video {
        // The aggregate output-row bound is not sufficient for a short video:
        // one temporal group still runs the full spatial grid through the
        // per-sequence input buffers. Bound that grid independently, using the
        // actual sampled temporal length now that `t_bar` is known.
        let (p_max, _) = derive_glm_max_patches(
            requested_max_pixels,
            vcfg.patch_size,
            vcfg.temporal_patch_size,
        );
        let per_group_volume = t_bar
            .saturating_mul(p_max)
            .saturating_mul(vcfg.patch_size.saturating_mul(vcfg.patch_size));
        max_pixels = max_pixels.min(per_group_volume);
    }
    let h = orig_h as usize;
    let w = orig_w as usize;
    let mut h_bar = h.div_ceil(factor as usize) * factor as usize;
    let mut w_bar = w.div_ceil(factor as usize) * factor as usize;
    let aligned_pixels = |hh: usize, ww: usize| t_bar.saturating_mul(hh).saturating_mul(ww);
    if aligned_pixels(h_bar, w_bar) > max_pixels {
        let mut low = 1usize;
        let mut high = h.max(1);
        h_bar = factor as usize;
        w_bar = factor as usize;
        while low <= high {
            let content_h = (low + high) / 2;
            let content_w = (w.saturating_mul(content_h) / h.max(1)).max(1);
            let candidate_h = content_h.div_ceil(factor as usize) * factor as usize;
            let candidate_w = content_w.div_ceil(factor as usize) * factor as usize;
            if aligned_pixels(candidate_h, candidate_w) <= max_pixels {
                h_bar = candidate_h;
                w_bar = candidate_w;
                low = content_h + 1;
            } else {
                high = content_h.saturating_sub(1);
            }
        }
    } else if aligned_pixels(h_bar, w_bar) < min_pixels {
        let scale = (min_pixels as f64 / (temporal_len.max(t_factor) * h * w) as f64).sqrt();
        let content_h = ((h as f64 * scale).ceil() as usize).max(1);
        let content_w = ((w as f64 * scale).ceil() as usize).max(1);
        h_bar = content_h.div_ceil(factor as usize) * factor as usize;
        w_bar = content_w.div_ceil(factor as usize) * factor as usize;
        if aligned_pixels(h_bar, w_bar) > max_pixels {
            let mut low = 1usize;
            let mut high = h.max(1);
            h_bar = factor as usize;
            w_bar = factor as usize;
            while low <= high {
                let content_h = (low + high) / 2;
                let content_w = (w.saturating_mul(content_h) / h.max(1)).max(1);
                let candidate_h = content_h.div_ceil(factor as usize) * factor as usize;
                let candidate_w = content_w.div_ceil(factor as usize) * factor as usize;
                if aligned_pixels(candidate_h, candidate_w) <= max_pixels {
                    h_bar = candidate_h;
                    w_bar = candidate_w;
                    low = content_h + 1;
                } else {
                    high = content_h.saturating_sub(1);
                }
            }
        }
    }
    Ok((h_bar as u32, w_bar as u32))
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
