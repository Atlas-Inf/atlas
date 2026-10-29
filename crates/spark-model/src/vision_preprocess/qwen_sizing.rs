// SPDX-License-Identifier: AGPL-3.0-only

//! Qwen vision geometry: HF's `smart_resize` within the declared pixel
//! bounds, fitted to the encoder's patch capacity, or the historical
//! long-side clamp where nothing is declared. Split out of
//! `vision_preprocess` like `glm_sizing`.

use super::*;

/// HF's widest aspect ratio: `smart_resize` refuses a more extreme image.
const MAX_ASPECT_RATIO: u64 = 200;

/// The Qwen target (H, W) of a still (`frames = None`), or of every frame of
/// a clip of `frames` sampled frames.
///
/// Under a declared bound this is HF's `smart_resize` exactly
/// ([`smart_resize`]): the checkpoint's `min_pixels` scales a small still UP
/// (Qwen3.8: 65536, so a 32x32 icon is 256x256, 64 tokens), and `max_pixels`
/// is the operator's or the checkpoint's, lowered to the encoder's patch
/// capacity (`derive_max_patches`: 2048^2 at patch 16 against Qwen3.8's
/// declared 4096^2), which is what HF gives at that bound. A clip is sized
/// over all its frames within the checkpoint's video bounds, as Qwen3-VL's
/// video processor does, when it declares them, and like a still otherwise.
///
/// With no bound declared (no processor config, no `--vision-max-pixels`),
/// the historical sizing: [`fallback_target`].
pub(crate) fn qwen_fit(
    vcfg: &VisionConfig,
    orig_h: u32,
    orig_w: u32,
    max_pixels: Option<usize>,
    frames: Option<usize>,
) -> Result<(u32, u32)> {
    let ps = vcfg.patch_size.max(1);
    let factor = (ps * vcfg.spatial_merge_size) as u32;
    let p_max = crate::layers::vision_encoder::enc_impl::init::derive_max_patches(max_pixels, ps).0;
    let capacity = p_max * ps * ps;
    let fit = match (frames, vcfg.video_pixels, max_pixels.filter(|&p| p > 0)) {
        (Some(n), Some(video), _) => {
            ensure!(
                orig_h >= factor && orig_w >= factor,
                "video frames of {orig_w}x{orig_h} are smaller than the {factor}px grid unit"
            );
            // Each temporal group is one encoder sequence: at most
            // `capacity` per frame.
            let max = video.max_pixels.min(n.saturating_mul(capacity));
            let min = video.min_pixels.min(max);
            smart_resize(
                orig_h,
                orig_w,
                factor,
                n,
                vcfg.temporal_patch_size,
                min,
                max,
            )?
        }
        (_, _, Some(max)) => {
            let max = max.min(capacity);
            // Undeclared, the least a grid holds: one unit.
            let min = vcfg.min_pixels.unwrap_or(0).max((factor * factor) as usize);
            smart_resize(orig_h, orig_w, factor, 1, 1, min.min(max), max)?
        }
        _ => fallback_target(orig_h, orig_w, factor, ps, p_max),
    };
    Ok(fit_strip(fit, factor as usize, ps, p_max))
}

/// HF's Qwen `smart_resize`, exactly: transformers'
/// `qwen2_vl/image_processing_qwen2_vl.py` for a still (`frames = 1`,
/// `temporal_factor = 1`) and `qwen3_vl/video_processing_qwen3_vl.py` for a
/// clip. f64 in HF's operation order, and Python's `round` (half to even):
/// an 80px side snaps to 64 at factor 32, not 96.
fn smart_resize(
    h: u32,
    w: u32,
    factor: u32,
    frames: usize,
    temporal_factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(u32, u32)> {
    ensure!(
        h.max(w) as u64 <= MAX_ASPECT_RATIO * h.min(w) as u64,
        "image of {w}x{h} has an aspect ratio past {MAX_ASPECT_RATIO}:1"
    );
    let (hf, wf, f) = (h as f64, w as f64, factor as f64);
    let t_bar = frames.div_ceil(temporal_factor.max(1)) * temporal_factor.max(1);
    let volume = |hb: f64, wb: f64| t_bar as f64 * hb * wb;
    let content = frames as f64 * hf * wf;
    let mut h_bar = (hf / f).round_ties_even() * f;
    let mut w_bar = (wf / f).round_ties_even() * f;
    if volume(h_bar, w_bar) > max_pixels as f64 {
        let beta = (content / max_pixels as f64).sqrt();
        h_bar = f.max((hf / beta / f).floor() * f);
        w_bar = f.max((wf / beta / f).floor() * f);
    } else if volume(h_bar, w_bar) < min_pixels as f64 {
        let beta = (min_pixels as f64 / content).sqrt();
        h_bar = (hf * beta / f).ceil() * f;
        w_bar = (wf * beta / f).ceil() * f;
    }
    Ok((h_bar as u32, w_bar as u32))
}

/// The historical sizing, kept for a checkpoint that declares no bound: the
/// long side clamped to [`FALLBACK_MAX_DIM`], never upscaled, each side to
/// the nearest grid unit; past the encoder's capacity (the fallback at
/// patch 14), resized to the capacity's area instead, floored when rounding
/// lands above it.
fn fallback_target(h: u32, w: u32, unit: u32, ps: usize, p_max: usize) -> (u32, u32) {
    let scaled = |scale: f32, snap: fn(f32) -> f32| {
        let side = |orig: u32| (snap(orig as f32 * scale / unit as f32) as u32).max(1) * unit;
        (side(h), side(w))
    };
    let patches = |(h, w): (u32, u32)| (h as usize / ps) * (w as usize / ps);
    let base = scaled(
        (FALLBACK_MAX_DIM as f32 / h.max(w) as f32).min(1.0),
        f32::round,
    );
    if patches(base) <= p_max {
        return base;
    }
    let capacity = p_max * ps * ps;
    let scale = (capacity as f32 / (h as f32 * w as f32)).sqrt().min(1.0);
    let fit = scaled(scale, f32::round);
    if fit.0 as usize * fit.1 as usize <= capacity {
        fit
    } else {
        scaled(scale, f32::floor)
    }
}

/// A strip's short side cannot go below one grid unit, so its area can
/// exceed the capacity (HF's `max(factor, ...)` does the same): shorten the
/// long side until the grid fits. A no-op for every size that fits.
fn fit_strip(fit: (u32, u32), unit: usize, ps: usize, p_max: usize) -> (u32, u32) {
    let unit_patches = (unit / ps) * (unit / ps);
    let max_units = (p_max / unit_patches.max(1)).max(1);
    let (uh, uw) = (fit.0 as usize / unit, fit.1 as usize / unit);
    if uh * uw <= max_units {
        return fit;
    }
    let short = uh.min(uw);
    let long = (max_units / short).max(1);
    let (uh, uw) = if uh <= uw {
        (short, long)
    } else {
        (long, short)
    };
    ((uh * unit) as u32, (uw * unit) as u32)
}

#[cfg(test)]
#[path = "qwen_sizing_tests.rs"]
mod tests;
