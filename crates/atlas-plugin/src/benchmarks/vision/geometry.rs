// SPDX-License-Identifier: AGPL-3.0-only

//! How many vision tokens a fixture should become, and why that is the leg
//! worth having.
//!
//! A capability probe ("what colour is this?") answers whether the model saw
//! *an* image. It cannot tell you whether it saw the RIGHT one. The
//! 2026-08-14 resolution-cap defect is the proof: every image above 1280px on
//! the long side was silently downscaled to about a tenth of the area its
//! checkpoint permitted, and every capability probe still passed — a
//! downscaled picture is still recognisably red.
//!
//! Vision-token count is the observable that moves when preprocessing
//! changes, and the server reports it for free in `usage.prompt_tokens`. This
//! module is the model those assertions are checked against: the reference
//! processor's `smart_resize` (`Qwen2VLImageProcessorFast`, which Qwen3-VL
//! checkpoints declare), not any one engine's approximation of it.

/// The reference processor's lower area bound for patch 16 / merge 2
/// checkpoints: `size.shortest_edge` in their `preprocessor_config.json`
/// (Qwen3.8-Flash-Next-FP4-Local declares 65536, i.e. 256×256). An image
/// smaller than this is scaled UP until it covers it. Two ladder rungs sit
/// below it (224×224, which is also the calibration fixture, and the 8×8
/// tiny), so their expectations depend on it.
pub const MIN_PIXELS: u64 = 65_536;
/// The matching upper bound (`size.longest_edge`, 16777216 = 4096×4096). The
/// ladder sits inside it on purpose, so no rung depends on it.
pub const MAX_PIXELS: u64 = 16_777_216;

/// Snap a side to the patch grid, the way the reference preprocessor does
/// when no rescale is required: Python's `round(side / grid) * grid`.
///
/// Python's `round` is half-to-EVEN, which is what makes 336 → 320 and
/// 720 → 704 (10.5 → 10, 22.5 → 22). Pinned by `the_rounding_mode_is_pinned`.
/// Before 2026-10-29 this model used half-away-from-zero (336 → 352), which
/// matched the engine of the time but not the processor the checkpoint was
/// trained with. The engine moved to the reference rounding with #237, and
/// this model moved with it.
pub fn snap(side: u32, grid: u32) -> u32 {
    ((((side as f64) / (grid as f64)).round_ties_even() as u32).max(1)) * grid
}

/// The reference `smart_resize` target for a `w × h` image: both sides snapped
/// to `grid`, then, if the snapped area is outside `[MIN_PIXELS, MAX_PIXELS]`,
/// rescaled by the area ratio (floored to the grid going down, ceiled going
/// up), exactly as `transformers`' `smart_resize` does.
pub fn reference_target(w: u32, h: u32, grid: u32) -> (u32, u32) {
    let (wf, hf, g) = (w as f64, h as f64, grid as f64);
    let (mut sw, mut sh) = (snap(w, grid) as f64, snap(h, grid) as f64);
    if sw * sh > MAX_PIXELS as f64 {
        let beta = (wf * hf / MAX_PIXELS as f64).sqrt();
        sw = g.max((wf / beta / g).floor() * g);
        sh = g.max((hf / beta / g).floor() * g);
    } else if sw * sh < MIN_PIXELS as f64 {
        let beta = (MIN_PIXELS as f64 / (wf * hf)).sqrt();
        sw = (wf * beta / g).ceil() * g;
        sh = (hf * beta / g).ceil() * g;
    }
    (sw as u32, sh as u32)
}

/// Merged vision tokens for a `w × h` image at the given geometry.
///
/// The image goes to its reference target first, then each `patch × patch`
/// square is one patch and each `merge × merge` block of patches becomes one
/// token. For Qwen3-VL geometry (patch 16, merge 2) a 448×448 image is 784
/// patches and **196 tokens**: the figure measured against a live server on
/// 2026-08-14, which anchors this whole model.
pub fn expected_vision_tokens(w: u32, h: u32, patch: u32, merge: u32) -> u32 {
    let (tw, th) = reference_target(w, h, patch * merge);
    (tw / patch) * (th / patch) / (merge * merge)
}

// A `within_bound(w, h, max_pixels)` helper used to live here, documented as
// letting the geometry leg report UNMEASURED when the served bound could not
// be determined. Nothing ever called it — the driver asserts every rung
// unconditionally — so the doc described behaviour the benchmark did not have.
// Removed rather than wired up: the ladder is deliberately built so that every
// fixture is inside any plausible bound, which makes the prediction independent
// of the bound and the unconditional assertion correct. The one rung that
// straddles the retired 1280px clamp is the point of the exercise (see
// `provision::FIXTURES`), and it must be ASSERTED rather than skipped, or the
// regression it exists to catch would be reported as "unmeasured" and pass.

#[cfg(test)]
#[path = "geometry_tests.rs"]
mod geometry_tests;
