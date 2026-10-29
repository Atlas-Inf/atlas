// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `vision_preprocess.rs`.
//!
//! A sibling file rather than an inline `mod tests`: adding the video path's
//! shared helpers carried the parent past the 500-line cap, and it is not on
//! the allow-list.

use super::glm_sizing::{glm_max_item_tokens, glm_max_sequence_patches};
use super::*;

#[test]
fn test_image_pad_count_2x2_merge() {
    // Standard Qwen3-VL: 2×2 spatial merger folds a patch block
    // into one embedding token.
    assert_eq!(image_pad_count(64, 64, 2), 32 * 32);
    assert_eq!(image_pad_count(40, 80, 2), 20 * 40);
}

#[test]
fn test_image_pad_count_no_merge() {
    // spatial_merge_size=1 → identity (each patch → one token).
    assert_eq!(image_pad_count(64, 64, 1), 64 * 64);
    assert_eq!(image_pad_count(8, 12, 1), 96);
}

#[test]
fn test_image_pad_count_zero_sms_clamps_to_one() {
    // sms=0 is invalid; clamps to 1 so we never divide by zero.
    assert_eq!(image_pad_count(64, 64, 0), 64 * 64);
}

#[test]
fn glm_patch_order_keeps_each_spatial_merge_block_contiguous() {
    let mut cfg = ok_cfg();
    cfg.is_glm5_next = true;
    let got = patch_coordinates(&cfg, 4, 4);
    assert_eq!(
        got,
        vec![
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
        ]
    );
}

#[test]
fn glm_processor_geometry_uses_token_budget_and_clip_normalization() {
    let mut cfg = ok_cfg();
    cfg.is_glm5_next = true;
    cfg.patch_size = 14;
    cfg.spatial_merge_size = 2;
    cfg.temporal_patch_size = 2;
    let (h, w) = glm_target_size(&cfg, 1, 1, 2, None, false).unwrap();
    assert_eq!((h, w), (112, 112));
    let red = normalize_channel(&cfg, 0, 255);
    let blue = normalize_channel(&cfg, 2, 0);
    assert!((red - (1.0 - GLM_MEAN[0]) / GLM_STD[0]).abs() < 1e-6);
    assert!((blue - (0.0 - GLM_MEAN[2]) / GLM_STD[2]).abs() < 1e-6);
}

#[test]
fn glm_resize_or_pad_keeps_content_in_the_upper_left() {
    let source = image::RgbImage::from_pixel(2, 4, image::Rgb([255, 0, 0]));
    let padded = glm_resize_or_pad(&source, 4, 8, false);
    assert_eq!(padded.dimensions(), (8, 4));
    assert_eq!(padded.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
    assert_eq!(padded.get_pixel(7, 3), &image::Rgb([0, 0, 0]));
}

fn glm_budget_cfg() -> VisionConfig {
    let mut cfg = ok_cfg();
    cfg.is_glm5_next = true;
    cfg.patch_size = 14;
    cfg.temporal_patch_size = 2;
    cfg.spatial_merge_size = 2;
    cfg
}

#[test]
fn glm_large_photo_fits_the_bounded_encoder_capacity() {
    let cfg = glm_budget_cfg();
    let (h, w) = glm_target_size(&cfg, 1080, 1920, 2, None, false).unwrap();
    assert_eq!(h % 28, 0);
    assert_eq!(w % 28, 0);
    assert!(h as usize / 14 * (w as usize / 14) <= glm_max_sequence_patches(&cfg));
}

#[test]
fn glm_panorama_fits_without_unbounded_encoder_growth() {
    let cfg = glm_budget_cfg();
    let (h, w) = glm_target_size(&cfg, 512, 8192, 2, None, false).unwrap();
    assert_eq!(h % 28, 0);
    assert_eq!(w % 28, 0);
    assert!(h as usize / 14 * (w as usize / 14) <= glm_max_sequence_patches(&cfg));
}

#[test]
fn glm_video_total_rows_fit_the_bounded_encoder_capacity() {
    let cfg = glm_budget_cfg();
    for frames in [2, 4, 8, 256] {
        let (h, w) = glm_target_size(&cfg, 4096, 4096, frames, None, true).unwrap();
        let groups = frames / cfg.temporal_patch_size;
        let per_group_patches = h as usize / 14 * (w as usize / 14);
        let output_rows = groups * (h as usize / 14 / 2) * (w as usize / 14 / 2);
        assert!(
            per_group_patches <= glm_max_sequence_patches(&cfg),
            "frames={frames}: per-group patches={per_group_patches}"
        );
        assert!(
            output_rows <= glm_max_item_tokens(&cfg),
            "frames={frames}: output rows={output_rows}"
        );
    }
}

#[test]
fn glm_explicit_budget_counts_temporal_volume() {
    let cfg = glm_budget_cfg();
    let volume = 6_400 * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
    let (h, w) = glm_target_size(&cfg, 4096, 4096, 2, Some(volume), false).unwrap();
    assert_eq!(glm_pixel_budget(&cfg, Some(volume), false).1, volume);
    assert!(2 * h as usize * w as usize <= volume);
}

/// A `vision_config` for the shipped Qwen3-VL geometry.
fn ok_cfg() -> VisionConfig {
    super::fixtures::qwen_config()
}

/// A 1×1 PNG, the smallest decodable input.
const TINY_PNG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";

/// `parse_vision_config` reports a MISSING key as 0, so a checkpoint whose
/// `vision_config` omits `patch_size` reaches here as a zero divisor. The
/// old code divided `th / ps` at line 98 and panicked — from an HTTP
/// request body, i.e. a remote crash of the request task.
#[test]
fn zero_patch_size_is_an_error_not_a_divide_by_zero_panic() {
    let mut vcfg = ok_cfg();
    vcfg.patch_size = 0;
    let err = preprocess_image(TINY_PNG_B64, &vcfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("patch_size"), "{err}");
}

/// `grid_unit = patch_size * spatial_merge_size` is the other divisor, and
/// a zero here made it 0.0 in the f32 scale computation → a 0×0 target.
#[test]
fn zero_spatial_merge_size_is_an_error() {
    let mut vcfg = ok_cfg();
    vcfg.spatial_merge_size = 0;
    let err = preprocess_image(TINY_PNG_B64, &vcfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("spatial_merge_size"), "{err}");
}

/// `temporal_patch_size = 0` collapses `patch_dim` to 0, yielding an empty
/// pixel buffer that the encoder would then have to catch.
#[test]
fn zero_temporal_patch_size_is_an_error() {
    let mut vcfg = ok_cfg();
    vcfg.temporal_patch_size = 0;
    let err = preprocess_image(TINY_PNG_B64, &vcfg)
        .unwrap_err()
        .to_string();
    assert!(err.contains("temporal_patch_size"), "{err}");
}

/// The geometry check must not have broken the working path: a valid
/// config still decodes and produces `num_patches × patch_dim` floats.
#[test]
fn valid_config_still_preprocesses() {
    let vcfg = ok_cfg();
    let (pixels, gh, gw) = preprocess_image(TINY_PNG_B64, &vcfg).unwrap();
    let patch_dim = 3 * vcfg.temporal_patch_size * vcfg.patch_size * vcfg.patch_size;
    assert_eq!(pixels.len(), gh * gw * patch_dim);
    assert!(gh > 0 && gw > 0);
}

/// Garbage that is not an image must be a clean error, not a panic.
#[test]
fn undecodable_input_is_an_error() {
    assert!(preprocess_image("bm90LWFuLWltYWdl", &ok_cfg()).is_err());
    assert!(preprocess_image("!!! not base64 !!!", &ok_cfg()).is_err());
}

/// A PNG whose IHDR declares 65535×65535 (~12.9 GB of RGB) but whose file
/// is 70 bytes. The dimension limit rejects it from the header, before any
/// pixel buffer is reserved.
#[test]
fn decode_bomb_dimensions_are_rejected_before_allocation() {
    let mut png: Vec<u8> = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
    let mut ihdr: Vec<u8> = b"IHDR".to_vec();
    ihdr.extend_from_slice(&65535u32.to_be_bytes()); // width
    ihdr.extend_from_slice(&65535u32.to_be_bytes()); // height
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
    let crc = {
        // CRC-32 over the chunk type + data, as PNG requires.
        let mut c: u32 = 0xFFFF_FFFF;
        for &b in &ihdr {
            c ^= u32::from(b);
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    (c >> 1) ^ 0xEDB8_8320
                } else {
                    c >> 1
                };
            }
        }
        !c
    };
    png.extend_from_slice(&13u32.to_be_bytes());
    png.extend_from_slice(&ihdr);
    png.extend_from_slice(&crc.to_be_bytes());

    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png);
    // The fixture must exceed the limit under test, or it proves nothing.
    const { assert!(65535 > DECODE_MAX_SIDE) };
    // Assert on the LIMIT error specifically, not just "some error": the
    // fixture is also a truncated PNG, so a decoder without limits would
    // still fail — just later, after reserving the buffer. Only this exact
    // message proves the header check fired first.
    let err = format!("{:#}", preprocess_image(&b64, &ok_cfg()).unwrap_err());
    assert!(err.contains("Image size exceeds limit"), "{err}");
}

#[test]
fn test_image_pad_count_non_divisible_floors() {
    // Integer division truncates: 65/2 = 32 (not 33).
    assert_eq!(image_pad_count(65, 64, 2), 32 * 32);
}

// ── EXIF orientation ─────────────────────────────────────────────────────

/// The committed benchmark fixtures, reused so the unit test and the
/// benchmark cannot disagree about what the bytes contain: identical pixels
/// (red top half, blue bottom half), one tagged `Orientation = 6`, one
/// untagged.
const EXIF_ROT90: &[u8] = include_bytes!("../../../tests/fixtures/images/15_exif_rot90_224.jpg");
const EXIF_NONE: &[u8] = include_bytes!("../../../tests/fixtures/images/16_exif_none_224.jpg");

fn as_data_uri(bytes: &[u8]) -> String {
    use base64::Engine as _;
    format!(
        "data:image/jpeg;base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

/// Mean red channel of a quadrant, as a fraction of the image.
fn region_red(img: &image::RgbImage, x0: f32, y0: f32, x1: f32, y1: f32) -> f32 {
    let (w, h) = (img.width() as f32, img.height() as f32);
    let (xs, ys) = ((x0 * w) as u32, (y0 * h) as u32);
    let (xe, ye) = ((x1 * w) as u32, (y1 * h) as u32);
    let mut sum = 0f32;
    let mut n = 0f32;
    for y in ys..ye {
        for x in xs..xe {
            sum += img.get_pixel(x, y)[0] as f32;
            n += 1.0;
        }
    }
    sum / n.max(1.0)
}

/// The untagged control: nothing to apply, so red stays on TOP. This is the
/// non-regression half — every PNG and every untagged JPEG must decode
/// exactly as it did before orientation handling existed.
#[test]
fn an_untagged_image_is_not_rotated() {
    let img = decode_image(&as_data_uri(EXIF_NONE))
        .expect("decode")
        .to_rgb8();
    let top = region_red(&img, 0.1, 0.05, 0.9, 0.35);
    let bottom = region_red(&img, 0.1, 0.65, 0.9, 0.95);
    assert!(
        top > 200.0 && bottom < 60.0,
        "expected red on top, blue below; got top={top:.0} bottom={bottom:.0}"
    );
}

/// ★ The tagged image MUST come out rotated. `Orientation = 6` means "rotate
/// 90° clockwise to display", which moves the stored top edge to the right
/// side — so the red half ends up on the RIGHT.
///
/// Before 2026-08-14 this assertion would have failed: the tag was ignored and
/// red stayed on top, meaning every rotated phone photo reached the model a
/// quarter turn from how its owner saw it.
#[test]
fn an_exif_tagged_image_is_rotated_upright() {
    let img = decode_image(&as_data_uri(EXIF_ROT90))
        .expect("decode")
        .to_rgb8();
    let left = region_red(&img, 0.05, 0.1, 0.35, 0.9);
    let right = region_red(&img, 0.65, 0.1, 0.95, 0.9);
    assert!(
        right > 200.0 && left < 60.0,
        "Orientation=6 should put the red half on the RIGHT; got left={left:.0} right={right:.0}"
    );
}

/// Rotation happens BEFORE the grid snap, so a portrait photo tagged as
/// landscape is measured at its displayed shape rather than its stored one.
/// A 90° turn swaps the axes; on a square fixture the token count is
/// unchanged, which is what makes this safe to apply universally.
#[test]
fn rotation_does_not_change_the_token_count_of_a_square() {
    let (_, gh_a, gw_a) = preprocess_image(&as_data_uri(EXIF_ROT90), &ok_cfg()).expect("tagged");
    let (_, gh_b, gw_b) = preprocess_image(&as_data_uri(EXIF_NONE), &ok_cfg()).expect("untagged");
    assert_eq!((gh_a, gw_a), (gh_b, gw_b));
}
