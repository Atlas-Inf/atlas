// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B CPU image preprocessing (Wave 1 of the multimodal bring-up).
//!
//! Pipeline, matching the Gemma-4 E2B processor semantics:
//! 1. Decode the base64 data URI through the shared [`super::decode`] helper.
//! 2. Aspect-ratio-preserving resize to a target pixel AREA of
//!    `max_soft_tokens × pooling_kernel_size² × patch_size²`, both dimensions
//!    rounded DOWN to multiples of `unit = pooling_kernel_size × patch_size`.
//! 3. Rescale pixels by 1/255. NO mean/std normalization.
//! 4. Patchify into 16×16 patches, each flattened to `3 × 16 × 16` floats.
//! 5. Per-patch (x, y) grid position ids; slots not used (fixed-size padding
//!    in a later wave) carry (-1, -1).
//!
//! The dynamic soft-token count is `P / pooling_kernel_size²` where `P` is
//! the patch count — that is what the model's token expansion needs.

use anyhow::{Result, bail};
use atlas_core::config::GemmaVisionConfig;

use super::decode::decode_image;

/// The result of preprocessing one image for the Gemma-4 E2B vision tower.
#[derive(Debug, Clone)]
pub struct GemmaImageInput {
    /// Flat pixel tensor, shape `[P, 3 × patch_size × patch_size]` — one
    /// row per patch, channels-major within the patch. Values in [0, 1].
    pub pixels: Vec<f32>,
    /// Patches along the resized height (`resized_h / patch_size`).
    pub grid_h: usize,
    /// Patches along the resized width (`resized_w / patch_size`).
    pub grid_w: usize,
    /// Per-patch position ids as `(x, y)` grid coordinates, one per patch in
    /// the same row-major order as `pixels`. Unused (padding) slots in a
    /// fixed-size tensor are (-1, -1).
    pub pos_ids: Vec<(i32, i32)>,
    /// Dynamic soft-token count: `P / pooling_kernel_size²`.
    pub soft_token_count: usize,
}

/// Reject a Gemma vision config whose geometry cannot drive the preprocessor.
///
/// Missing keys in the checkpoint's `config.json` are parsed as `0`; an absent
/// `patch_size` or `pooling_kernel_size` would otherwise reach the resize and
/// patchify steps as a zero divisor. Deliberately no fallback default.
fn validate_geometry(cfg: &GemmaVisionConfig, max_soft_tokens: usize) -> Result<()> {
    if cfg.patch_size == 0 {
        bail!(
            "gemma vision_config.patch_size is 0 (missing or invalid in the checkpoint's config.json)"
        );
    }
    if cfg.pooling_kernel_size == 0 {
        bail!("gemma vision_config.pooling_kernel_size is 0 (missing or invalid in config.json)");
    }
    if max_soft_tokens == 0 {
        bail!("gemma max_soft_tokens is 0 (missing or invalid in config.json)");
    }
    Ok(())
}

/// Compute the Gemma-4 E2B resize/geometry for a decoded `w × h` image.
///
/// Returns `(resized_w, resized_h, grid_h, grid_w)`:
/// - both resized dims are multiples of `unit = pooling_kernel_size × patch_size`
///   (48 for the defaults 3 × 16), each at least one unit, with an
///   aspect-ratio-preserving scale targeting
///   `max_soft_tokens × unit²` pixels;
/// - `grid = resized / patch_size` (the patch grid that drives pixel and
///   position-id layouts); the soft-token count derived from the grid is
///   `grid_h × grid_w / pooling_kernel_size²`.
pub fn gemma_grid_for(
    w: u32,
    h: u32,
    cfg: &GemmaVisionConfig,
    max_soft_tokens: usize,
) -> (usize, usize, usize, usize) {
    let unit = (cfg.pooling_kernel_size * cfg.patch_size) as u32;
    // Target pixel AREA = max_soft_tokens × unit² (e.g. 280 × 2304 = 645,120).
    let target_area = (max_soft_tokens as f32) * (unit as f32) * (unit as f32);
    let scale = (target_area / ((w as f32) * (h as f32))).sqrt();
    // Round DOWN to a multiple of `unit`; never below one unit per side.
    let rw = (((w as f32) * scale / unit as f32).floor() as u32).max(1) * unit;
    let rh = (((h as f32) * scale / unit as f32).floor() as u32).max(1) * unit;
    let gh = (rh as usize) / cfg.patch_size;
    let gw = (rw as usize) / cfg.patch_size;
    (rw as usize, rh as usize, gh, gw)
}

/// Per-patch `(x, y)` position ids in row-major patch order.
fn gemma_pos_ids(grid_h: usize, grid_w: usize) -> Vec<(i32, i32)> {
    let mut ids = Vec::with_capacity(grid_h * grid_w);
    for y in 0..grid_h {
        for x in 0..grid_w {
            ids.push((x as i32, y as i32));
        }
    }
    ids
}

/// Preprocess one base64 image with an explicit soft-token budget.
pub fn preprocess_gemma_image(
    data_uri: &str,
    cfg: &GemmaVisionConfig,
    max_soft_tokens: usize,
) -> Result<GemmaImageInput> {
    validate_geometry(cfg, max_soft_tokens)?;
    let img = decode_image(data_uri)?.to_rgb8();
    let (rw, rh, gh, gw) = gemma_grid_for(img.width(), img.height(), cfg, max_soft_tokens);
    // CatmullRom — closest BICUBIC match in the `image` crate, mirroring the
    // Qwen preprocessor and HF's PIL resample=3 (BICUBIC) processors.
    let img = image::imageops::resize(
        &img,
        rw as u32,
        rh as u32,
        image::imageops::FilterType::CatmullRom,
    );

    // Patchify: each patch is C × patch_size × patch_size floats, channels
    // first, in row-major patch order (matching `pos_ids`).
    let ps = cfg.patch_size;
    let patch_dim = 3 * ps * ps;
    let num_patches = gh * gw;
    let mut pixels = vec![0.0f32; num_patches * patch_dim];
    for ph in 0..gh {
        for pw in 0..gw {
            let patch_idx = ph * gw + pw;
            for c in 0..3usize {
                for py in 0..ps {
                    for px in 0..ps {
                        let raw = img.get_pixel((pw * ps + px) as u32, (ph * ps + py) as u32)[c]
                            as f32
                            / 255.0;
                        let off = c * ps * ps + py * ps + px;
                        pixels[patch_idx * patch_dim + off] = raw;
                    }
                }
            }
        }
    }

    let pks = cfg.pooling_kernel_size;
    Ok(GemmaImageInput {
        pixels,
        grid_h: gh,
        grid_w: gw,
        pos_ids: gemma_pos_ids(gh, gw),
        soft_token_count: num_patches / (pks * pks),
    })
}

/// Preprocess one base64 image using `cfg.max_soft_tokens` as the budget.
pub fn preprocess_gemma_image_default(
    data_uri: &str,
    cfg: &GemmaVisionConfig,
) -> Result<GemmaImageInput> {
    preprocess_gemma_image(data_uri, cfg, cfg.max_soft_tokens)
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::Engine;
    use image::{DynamicImage, Rgb, RgbImage};
    use std::io::Cursor;

    /// A `GemmaVisionConfig` with the shipped Gemma-4 E2B image geometry:
    /// 16 px patches, 3×3 pooling, 280 soft tokens (→ 645,120 px target area).
    fn default_cfg() -> GemmaVisionConfig {
        GemmaVisionConfig {
            hidden_size: 1152,
            intermediate_size: 4304,
            num_hidden_layers: 27,
            num_attention_heads: 16,
            head_dim: 72,
            patch_size: 16,
            pooling_kernel_size: 3,
            position_embedding_size: 4096,
            use_clipped_linears: true,
            image_token_id: 261_022,
            rope_theta: 10_000.0,
            max_patches: 2_520,
            max_soft_tokens: 280,
            position_table_shape: (1, 256, 1152),
            norm_eps: 1e-6,
            video_frames: 16,
            video_soft_tokens_per_frame: 128,
            video_token_id: 261_023,
            boi_token_id: 261_024,
            eoi_token_id: 261_025,
        }
    }

    /// 768×768 is already 48-aligned and near the 645,120 px area target, so
    /// the golden geometry is the identity: 48×48 grid → 2304 patches → 256
    /// soft tokens (2304 / 3²).
    #[test]
    fn golden_768_square_grid() {
        let (rw, rh, gh, gw) = gemma_grid_for(768, 768, &default_cfg(), 280);
        assert_eq!((rw, rh), (768, 768));
        assert_eq!((gh, gw), (48, 48));
        assert_eq!(gh * gw, 2_304);
        assert_eq!(gh * gw / 9, 256);
    }

    /// 800×600 must be scaled to the target area, then both dims snapped DOWN
    /// to multiples of 48, keeping the area under the 645,120 px budget.
    #[test]
    fn non_multiple_dims_snap_down_to_48() {
        let cfg = default_cfg();
        let (rw, rh, gh, gw) = gemma_grid_for(800, 600, &cfg, 280);
        assert_eq!((rw, rh), (912, 672));
        assert_eq!(rw % 48, 0);
        assert_eq!(rh % 48, 0);
        assert!(rw * rh <= 280 * 9 * 256, "area {} over budget", rw * rh);
        assert_eq!((gh, gw), (42, 57)); // 672/16, 912/16
    }

    /// A 1×1 image must never collapse to a 0-size grid: each resized dim is
    /// at least one 48 px unit.
    #[test]
    fn tiny_image_keeps_minimum_geometry() {
        let (rw, rh, gh, gw) = gemma_grid_for(1, 1, &default_cfg(), 280);
        assert!(rw >= 48 && rh >= 48);
        assert!(gh >= 1 && gw >= 1);
        assert_eq!(rw % 48, 0);
        assert_eq!(rh % 48, 0);
    }

    /// Position ids are (x, y) grid coordinates in row-major patch order.
    #[test]
    fn pos_ids_2x2_grid() {
        assert_eq!(gemma_pos_ids(2, 2), vec![(0, 0), (1, 0), (0, 1), (1, 1)]);
    }

    /// Row-major order: x (column) varies fastest; last patch is (gw-1, gh-1).
    #[test]
    fn pos_ids_3x4_grid() {
        let ids = gemma_pos_ids(3, 4);
        assert_eq!(ids.len(), 12);
        assert_eq!(ids[0], (0, 0));
        assert_eq!(ids[4], (0, 1)); // second row starts
        assert_eq!(ids[11], (3, 2)); // bottom-right
    }

    #[test]
    fn zero_patch_size_is_an_error() {
        let mut cfg = default_cfg();
        cfg.patch_size = 0;
        let err = preprocess_gemma_image("", &cfg, 280)
            .unwrap_err()
            .to_string();
        assert!(err.contains("patch_size"), "{err}");
    }

    #[test]
    fn zero_pooling_kernel_is_an_error() {
        let mut cfg = default_cfg();
        cfg.pooling_kernel_size = 0;
        let err = preprocess_gemma_image("", &cfg, 280)
            .unwrap_err()
            .to_string();
        assert!(err.contains("pooling_kernel_size"), "{err}");
    }

    #[test]
    fn zero_soft_token_budget_is_an_error() {
        let err = preprocess_gemma_image("", &default_cfg(), 0)
            .unwrap_err()
            .to_string();
        assert!(err.contains("max_soft_tokens"), "{err}");
    }

    /// Encode `w × h` solid-grey PNG bytes into a base64 data URI.
    fn png_data_uri(w: u32, h: u32) -> String {
        let img = RgbImage::from_fn(w, h, |x, y| {
            Rgb([(x * 7 % 256) as u8, (y * 11 % 256) as u8, 128])
        });
        let mut bytes = Vec::new();
        DynamicImage::ImageRgb8(img)
            .write_to(&mut Cursor::new(&mut bytes), image::ImageFormat::Png)
            .unwrap();
        let b64 = base64::engine::general_purpose::STANDARD.encode(bytes);
        format!("data:image/png;base64,{b64}")
    }

    /// Full pipeline on a generated PNG with the default 280-soft-token
    /// budget: the 64×64 input scales up to the golden 768×768 grid, and the
    /// pixel tensor is exactly P × 768 floats in [0, 1].
    #[test]
    fn full_pipeline_default_budget() {
        let cfg = default_cfg();
        let out = preprocess_gemma_image(&png_data_uri(64, 64), &cfg, 280).unwrap();
        assert_eq!((out.grid_h, out.grid_w), (48, 48));
        assert_eq!(out.soft_token_count, 256);
        assert_eq!(out.pixels.len(), 2_304 * 768);
        assert_eq!(out.pos_ids.len(), 2_304);
        assert!(out.pixels.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    /// `preprocess_gemma_image_default` honors `cfg.max_soft_tokens`: a 4-token
    /// budget gives a 96×96 target (area 4 × 2304), a 6×6 grid and 36 patches.
    #[test]
    fn default_entry_point_uses_cfg_budget() {
        let mut cfg = default_cfg();
        cfg.max_soft_tokens = 4;
        let out = preprocess_gemma_image_default(&png_data_uri(64, 64), &cfg).unwrap();
        assert_eq!((out.grid_h, out.grid_w), (6, 6));
        assert_eq!(out.soft_token_count, 4);
        assert_eq!(out.pixels.len(), 36 * 768);
    }
}
