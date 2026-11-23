// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2's image processor (`Inkling2ImageProcessor`): no resize. The
//! image is padded on its right and bottom to a grid of
//! `patch_size * spatial_merge_size` pixel tiles (40 for the shipped
//! checkpoint), one image token per tile; padding pixels are -1 on the 0-255
//! scale, and every pixel is then rescaled and normalised with mean = std =
//! 0.5. Transparent pixels are composited onto an 8-pixel 204/153
//! checkerboard first.
//!
//! One deviation, for the encoder's capacity: an image of more tiles than the
//! encoder holds (`derive_max_patches`) is first scaled down, keeping its
//! aspect, until it fits. The reference processor has no such bound.

use anyhow::{Result, ensure};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, RgbImage};

/// A padding pixel after rescale and normalisation: `(-1/255 - 0.5) / 0.5`.
const PAD: f32 = (-1.0 / 255.0 - 0.5) / 0.5;

fn tile(vcfg: &VisionConfig) -> u32 {
    (vcfg.patch_size * vcfg.spatial_merge_size) as u32
}

fn tiles(vcfg: &VisionConfig, h: u32, w: u32) -> usize {
    let t = tile(vcfg);
    h.div_ceil(t) as usize * w.div_ceil(t) as usize
}

/// The size the image content is drawn at: its own, or the largest scaled
/// size of the same aspect whose tiles fit the encoder.
pub(crate) fn content_size(
    vcfg: &VisionConfig,
    h: u32,
    w: u32,
    max_pixels: Option<usize>,
) -> Result<(u32, u32)> {
    ensure!(h > 0 && w > 0, "empty image");
    let p_max = crate::layers::vision_encoder::enc_impl::init::derive_max_patches(
        max_pixels,
        vcfg.patch_size,
    )
    .0;
    let cap = (p_max / (vcfg.spatial_merge_size * vcfg.spatial_merge_size)).max(1);
    if tiles(vcfg, h, w) <= cap {
        return Ok((h, w));
    }
    let area = (cap * (tile(vcfg) as usize).pow(2)) as f64;
    let mut scale = (area / (h as f64 * w as f64)).sqrt().min(1.0);
    loop {
        let (sh, sw) = (
            ((h as f64 * scale) as u32).max(1),
            ((w as f64 * scale) as u32).max(1),
        );
        if tiles(vcfg, sh, sw) <= cap {
            return Ok((sh, sw));
        }
        ensure!(
            scale > 1e-4,
            "image {w}x{h} cannot be fitted to {cap} tiles"
        );
        scale *= 0.98;
    }
}

/// The padded canvas `(height, width)`: the content rounded up to whole tiles.
pub(crate) fn canvas(
    vcfg: &VisionConfig,
    h: u32,
    w: u32,
    max_pixels: Option<usize>,
) -> Result<(u32, u32)> {
    let (sh, sw) = content_size(vcfg, h, w, max_pixels)?;
    let t = tile(vcfg);
    Ok((sh.div_ceil(t) * t, sw.div_ceil(t) * t))
}

/// RGB, with alpha composited onto the reference checkerboard.
pub(crate) fn to_rgb(img: DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.into_rgb8();
    }
    let rgba = img.into_rgba8();
    RgbImage::from_fn(rgba.width(), rgba.height(), |x, y| {
        let p = rgba.get_pixel(x, y);
        let background = if (x / 8 + y / 8) % 2 == 0 {
            204u32
        } else {
            153
        };
        let a = p[3] as u32;
        let mix = |c: u8| ((c as u32 * a + background * (255 - a) + 127) / 255) as u8;
        image::Rgb([mix(p[0]), mix(p[1]), mix(p[2])])
    })
}

/// `(pixels, grid_h, grid_w)`: one row of `3 * patch_size²` values per
/// patch (rows, columns, channels), the patches of a tile together (its
/// merge rows, then merge columns) and the tiles in raster order, which is
/// the order the tower's merger and position tables read.
pub(crate) fn patchify(
    vcfg: &VisionConfig,
    img: &RgbImage,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    let (sh, sw) = content_size(vcfg, img.height(), img.width(), max_pixels)?;
    let scaled;
    let img = if (sh, sw) == (img.height(), img.width()) {
        img
    } else {
        scaled = image::imageops::resize(img, sw, sh, image::imageops::FilterType::CatmullRom);
        &scaled
    };
    let (ps, merge) = (vcfg.patch_size, vcfg.spatial_merge_size);
    let t = (ps * merge) as u32;
    let (rows, cols) = (sh.div_ceil(t) as usize, sw.div_ceil(t) as usize);
    let patch_dim = 3 * ps * ps;
    let mut pixels = vec![PAD; rows * cols * merge * merge * patch_dim];
    let mut at = 0usize;
    for tr in 0..rows {
        for tc in 0..cols {
            for mr in 0..merge {
                for mc in 0..merge {
                    let (y0, x0) = ((tr * merge + mr) * ps, (tc * merge + mc) * ps);
                    for py in 0..ps {
                        let y = (y0 + py) as u32;
                        if y >= sh {
                            at += ps * 3;
                            continue;
                        }
                        for px in 0..ps {
                            let x = (x0 + px) as u32;
                            if x < sw {
                                let p = img.get_pixel(x, y);
                                for c in 0..3 {
                                    pixels[at + c] = (p[c] as f32 / 255.0 - 0.5) / 0.5;
                                }
                            }
                            at += 3;
                        }
                    }
                }
            }
        }
    }
    Ok((pixels, rows * merge, cols * merge))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg() -> VisionConfig {
        VisionConfig {
            is_inkling2: true,
            patch_size: 20,
            spatial_merge_size: 2,
            temporal_patch_size: 1,
            ..crate::vision_preprocess::fixtures::qwen_config()
        }
    }

    #[test]
    fn one_token_per_40_pixel_tile_without_resizing() {
        let v = cfg();
        // 200x120 (w x h): 3 rows of 5 tiles, as the reference processor.
        assert_eq!(canvas(&v, 120, 200, None).unwrap(), (120, 200));
        assert_eq!(canvas(&v, 121, 201, None).unwrap(), (160, 240));
        assert_eq!(canvas(&v, 1, 1, None).unwrap(), (40, 40));
    }

    #[test]
    fn padding_is_minus_one_on_the_byte_scale_and_patches_follow_their_tile() {
        let v = cfg();
        // 50 wide, 30 high: one row of two tiles; white except a black
        // pixel at (x 45, y 3), which is tile 1, merge (0, 0), patch (3, 5).
        let mut img = RgbImage::from_pixel(50, 30, image::Rgb([255, 255, 255]));
        img.put_pixel(45, 3, image::Rgb([0, 0, 0]));
        let (px, gh, gw) = patchify(&v, &img, None).unwrap();
        assert_eq!((gh, gw), (2, 4));
        assert_eq!(px.len(), 8 * 1200);
        let at =
            |patch: usize, y: usize, x: usize, c: usize| px[patch * 1200 + (y * 20 + x) * 3 + c];
        assert_eq!(at(0, 0, 0, 0), 1.0);
        assert_eq!(at(4, 3, 5, 1), -1.0, "the black pixel");
        // tile 0, merge (1, 0): rows 20..40, of which 30..40 are padding
        assert_eq!(at(2, 9, 0, 2), 1.0);
        assert_eq!(at(2, 10, 0, 2), PAD);
        // tile 1, merge (0, 1): columns 60..80, all padding
        assert_eq!(at(5, 0, 0, 0), PAD);
        assert_eq!(PAD, -257.0 / 255.0);
    }

    #[test]
    fn an_image_past_the_encoder_is_scaled_down_to_fit() {
        let v = cfg();
        // derive_max_patches(None) = 6400 patches = 1600 tiles.
        let (h, w) = content_size(&v, 4000, 3000, None).unwrap();
        assert!(tiles(&v, h, w) <= 1600 && tiles(&v, h, w) > 1400, "{h}x{w}");
        assert!((h as f64 / w as f64 - 4.0 / 3.0).abs() < 0.01);
        assert_eq!(content_size(&v, 1600, 1600, None).unwrap(), (1600, 1600));
    }

    #[test]
    fn transparency_lands_on_the_checkerboard() {
        let mut rgba = image::RgbaImage::from_pixel(16, 8, image::Rgba([10, 20, 30, 0]));
        rgba.put_pixel(0, 0, image::Rgba([10, 20, 30, 255]));
        let rgb = to_rgb(DynamicImage::ImageRgba8(rgba));
        assert_eq!(rgb.get_pixel(0, 0).0, [10, 20, 30]);
        assert_eq!(rgb.get_pixel(1, 0).0, [204, 204, 204]);
        assert_eq!(rgb.get_pixel(8, 0).0, [153, 153, 153]);
    }
}
