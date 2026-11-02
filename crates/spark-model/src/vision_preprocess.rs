// SPDX-License-Identifier: AGPL-3.0-only

//! CPU-side image preprocessing for Qwen3-VL vision inputs.
//!
//! Decodes base64 JPEG/PNG images, resizes to a grid snapped to
//! `patch_size × spatial_merge_size`, normalizes with ImageNet stats,
//! and produces a flat `f32` tensor ready for the GPU vision encoder.

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, ImageDecoder, ImageReader, Limits, Rgb, RgbImage};

mod glm_sizing;
mod qwen_sizing;
pub(crate) use glm_sizing::glm_pixel_budget;
pub(crate) use glm_sizing::{derive_glm_max_patches, glm_resize_or_pad, glm_target_size};
pub(crate) use qwen_sizing::qwen_fit;

/// SigLIP normalization — matches HF's Qwen2VLImageProcessor
/// (`image_mean = image_std = (0.5, 0.5, 0.5)` → pixels mapped to [-1, 1]).
/// `pub(crate)` because the video path normalizes with the identical stats —
/// a video frame is not a different kind of pixel, and two copies of these
/// numbers is two places for them to drift apart.
pub(crate) const MEAN: [f32; 3] = [0.5, 0.5, 0.5];
pub(crate) const STD: [f32; 3] = [0.5, 0.5, 0.5];

/// GLM-5.3 ships the OpenAI CLIP processor constants in its nested
/// `processor_config.json`, rather than the Qwen/SigLIP defaults above.
pub(crate) const GLM_MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
pub(crate) const GLM_STD: [f32; 3] = [0.26862954, 0.261_302_6, 0.275_777_1];

/// Long-side cap used ONLY when nothing else bounds the image — i.e. the
/// caller passed no `max_pixels` because the checkpoint shipped no
/// `preprocessor_config.json` and the operator set no `--vision-max-pixels`.
///
/// This was an UNCONDITIONAL ceiling until 2026-08-14, which silently threw
/// away most of the resolution such checkpoints allow. Qwen3.8-27B declares
/// `size = {longest_edge: 16777216, shortest_edge: 65536}` — pixel AREAS, so
/// up to 4096² — while this constant clamped every image to 1280 on the long
/// side, roughly a tenth of the permitted area. Measured before the change:
/// a 1344×1344 input came back as 1600 merged tokens (1280×1280), and
/// 1920×1080 as ~900 (1280×720). Detail-bearing inputs — documents, charts,
/// dense screenshots — paid for that directly, and nothing logged it.
const FALLBACK_MAX_DIM: u32 = 1280;

/// Decoder limit: reject a header declaring more than this on either side
/// before a single pixel is allocated. Everything is resized down to the
/// encoder's capacity anyway, so this only has to be above any real camera
/// image; 16384 is ~4× the long side of a 50 MP photo.
pub(crate) const DECODE_MAX_SIDE: u32 = 16_384;

/// Decoder limit: bytes one image's decoded buffer (its native color type)
/// may take. 640 MiB admits a 200 MP RGB phone photo (16320x12240, 571 MiB)
/// and refuses absurd declarations (16384^2 RGBA16, 2 GiB) from a few-KB
/// body. On GB10's unified memory this competes with the KV cache, so
/// decodes past [`DECODE_LARGE`] are gated to two at once (`probe`).
pub(crate) const DECODE_MAX_ALLOC: u64 = 640 * 1024 * 1024;

/// A decode past this many bytes is "large" and takes the large-decode gate
/// too: 192 MiB is an 8000x8000 RGB photo.
const DECODE_LARGE: u64 = 192 * 1024 * 1024;

/// Split a base64 `data:` URI (or a bare base64 string) into its declared
/// MIME type and decoded bytes.
///
/// Shared with the video path, which needs the SAME unwrapping but a
/// different decoder — and needs the MIME string, because "this is an mp4"
/// is worth saying by name rather than discovering as a parse failure.
/// The MIME is empty when the input carried no `data:` header.
pub(crate) fn decode_data_uri_bytes(data_uri: &str) -> Result<(String, Vec<u8>)> {
    // Strip optional "data:<mime>;base64," prefix.
    let (mime, b64) = if let Some(pos) = data_uri.find(",base64,") {
        (
            data_uri[..pos].trim_start_matches("data:").to_string(),
            &data_uri[pos + 8..],
        )
    } else if let Some(rest) = data_uri.strip_prefix("data:") {
        // "data:image/jpeg;base64,..." — the common, well-formed shape.
        match rest.find(',') {
            Some(p) => (
                rest[..p].trim_end_matches(";base64").to_string(),
                &rest[p + 1..],
            ),
            None => (String::new(), data_uri),
        }
    } else {
        (String::new(), data_uri)
    };

    let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64.trim())
        .context("base64 decode failed")?;
    Ok((mime, bytes))
}

/// The image formats `decode_bytes` reads (the `image` crate features enabled
/// in this crate's Cargo.toml), as the client is told when it sends another.
const SUPPORTED_IMAGE_FORMATS: &str = "send JPEG, PNG, GIF, WebP, BMP, TIFF or ICO";

/// The format of `bytes`, from their magic bytes. Anything we cannot read is
/// refused by name, so the client learns what to send instead of seeing a
/// decoder error about a JPEG it never sent.
fn sniff_format(bytes: &[u8]) -> Result<image::ImageFormat> {
    let fmt = image::guess_format(bytes)
        .map_err(|_| anyhow::anyhow!("unrecognized image format; {SUPPORTED_IMAGE_FORMATS}"))?;
    if !fmt.reading_enabled() {
        bail!("{fmt:?} images are not supported; {SUPPORTED_IMAGE_FORMATS}");
    }
    Ok(fmt)
}

/// A reader over `bytes` with our decoder limits.
fn limited_reader(bytes: &[u8], fmt: image::ImageFormat) -> ImageReader<std::io::Cursor<&[u8]>> {
    let mut reader = ImageReader::new(std::io::Cursor::new(bytes));
    reader.set_format(fmt);
    let mut limits = Limits::default();
    limits.max_image_width = Some(DECODE_MAX_SIDE);
    limits.max_image_height = Some(DECODE_MAX_SIDE);
    limits.max_alloc = Some(DECODE_MAX_ALLOC);
    reader.limits(limits);
    reader
}

/// Decode a base64 data URI or raw base64 string into a `DynamicImage` (the
/// probe and the decode do this in two steps outside tests).
pub(crate) fn decode_image(data_uri: &str) -> Result<DynamicImage> {
    let (_mime, bytes) = decode_data_uri_bytes(data_uri)?;
    let fmt = sniff_format(&bytes)?;
    decode_bytes(&bytes, fmt)
}

/// Refuse an image whose decoded buffer (native color type) would exceed
/// [`DECODE_MAX_ALLOC`]: a ~1 MB PNG may declare 16384^2 RGBA16 (2 GiB).
fn check_decode_size(total_bytes: u64) -> Result<()> {
    ensure!(
        total_bytes <= DECODE_MAX_ALLOC,
        "image decodes to {} MiB, more than the {} MiB limit; send a smaller image",
        total_bytes >> 20,
        DECODE_MAX_ALLOC >> 20
    );
    Ok(())
}

/// Whether an EXIF orientation turns the image a quarter turn (swapping the
/// sides its header declares).
fn orientation_swaps_sides(o: image::metadata::Orientation) -> bool {
    use image::metadata::Orientation::*;
    matches!(o, Rotate90 | Rotate270 | Rotate90FlipH | Rotate270FlipH)
}

/// Decode already-sniffed image bytes into a `DynamicImage`.
pub(crate) fn decode_bytes(bytes: &[u8], fmt: image::ImageFormat) -> Result<DynamicImage> {
    // Decode through `ImageReader` rather than `load_from_memory_with_format`
    // so the limits are ours. (The free function is not unlimited — it applies
    // `Limits::default()`, i.e. 512 MiB alloc — but it sets NO dimension cap,
    // and the alloc cap is documented as non-strict.) A 40-byte PNG header can
    // declare 65535×65535; the dimension limit rejects that from the header,
    // before any buffer is reserved.
    let reader = limited_reader(bytes, fmt);

    // ★ EXIF ORIENTATION IS APPLIED. A camera writes the sensor's raw pixels
    // and records how to turn them upright in an EXIF tag rather than rotating
    // the data, so a phone photo is very often stored sideways with
    // `Orientation = 6` ("rotate 90° CW to display"). Decoding without that
    // tag hands the model an image rotated a quarter turn — and it does not
    // error or look broken, it simply answers about a sideways picture, which
    // is the failure mode this whole benchmark family exists to catch.
    //
    // Measured on 2026-08-14: Atlas ignored the tag entirely. Every viewer the
    // user compares against — their phone, their browser, their file manager —
    // honours it, so "what the model saw" and "what the user saw" silently
    // disagreed on a large fraction of real photographs.
    //
    // `into_decoder` rather than `decode`, because the tag lives on the
    // DECODER and is gone once the pixels are out. A format that carries no
    // orientation reports `NoTransforms`, so this is a no-op for PNG and for
    // any JPEG without the tag — the earlier behaviour, preserved exactly
    // where there is nothing to apply.
    let mut decoder = reader.into_decoder().context("image decode failed")?;
    // `from_decoder` does not apply `max_alloc` (only `ImageReader::decode`
    // reserves against it), so check the decoded size here.
    check_decode_size(decoder.total_bytes())?;
    let orientation = decoder
        .orientation()
        .unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder).context("image decode failed")?;
    if orientation != image::metadata::Orientation::NoTransforms {
        tracing::debug!("applying EXIF orientation {orientation:?}");
        img.apply_orientation(orientation);
    }
    Ok(img)
}

/// Reject a vision config whose geometry cannot drive the preprocessor.
///
/// Every field here comes from a third-party `config.json` via
/// `parse_vision_config`, which reports a MISSING key as `0` — so an absent or
/// malformed `patch_size` reaches `preprocess_image` as a divisor of zero, and
/// `grid_unit = patch_size * spatial_merge_size` reaches the scale computation
/// as `0.0`, producing a 0×0 target and then a division by zero. Fail with a
/// named error instead. Deliberately no fallback default: silently assuming
/// `patch_size = 16` would let a mismatched checkpoint produce a wrongly-shaped
/// pixel buffer, which is the hazard the encoder's own length check exists for.
fn validate_geometry(vcfg: &VisionConfig) -> Result<()> {
    if vcfg.patch_size == 0 {
        bail!("vision_config.patch_size is 0 (missing or invalid in the checkpoint's config.json)");
    }
    if vcfg.spatial_merge_size == 0 {
        bail!("vision_config.spatial_merge_size is 0 (missing or invalid in config.json)");
    }
    if vcfg.temporal_patch_size == 0 {
        bail!("vision_config.temporal_patch_size is 0 (missing or invalid in config.json)");
    }
    Ok(())
}

/// Return patch coordinates in the order consumed by the native GLM tower.
/// GLM's `view(-1, merge, merge, hidden)` downsample assumes each spatial
/// merge block is contiguous; Qwen keeps the historical row-major ordering.
pub(crate) fn patch_coordinates(
    vcfg: &VisionConfig,
    grid_h: usize,
    grid_w: usize,
) -> Vec<(usize, usize)> {
    if !vcfg.is_glm5_next {
        return (0..grid_h)
            .flat_map(|ph| (0..grid_w).map(move |pw| (ph, pw)))
            .collect();
    }
    let merge = vcfg.spatial_merge_size;
    (0..grid_h / merge)
        .flat_map(|bh| {
            (0..grid_w / merge).flat_map(move |bw| {
                (0..merge).flat_map(move |ih| {
                    (0..merge).map(move |iw| (bh * merge + ih, bw * merge + iw))
                })
            })
        })
        .collect()
}

pub(crate) fn normalize_channel(vcfg: &VisionConfig, channel: usize, raw: u8) -> f32 {
    let (mean, std) = if vcfg.is_glm5_next {
        (GLM_MEAN[channel], GLM_STD[channel])
    } else {
        (MEAN[channel], STD[channel])
    };
    (raw as f32 / 255.0 - mean) / std
}

/// Target (H, W) in pixels of a still of `orig_h x orig_w`, for either tower.
/// One function for the header probe and the decode, so the two agree.
/// Refuses a Qwen grid past the encoder's patch capacity (a backstop:
/// [`qwen_fit`] already keeps it inside).
fn still_target(
    vcfg: &VisionConfig,
    orig_h: u32,
    orig_w: u32,
    max_pixels: Option<usize>,
) -> Result<(u32, u32)> {
    if vcfg.is_glm5_next {
        return glm_target_size(
            vcfg,
            orig_h,
            orig_w,
            vcfg.temporal_patch_size,
            max_pixels,
            false,
        );
    }
    let (th, tw) = qwen_fit(vcfg, orig_h, orig_w, max_pixels, None)?;
    let p_max = crate::layers::vision_encoder::enc_impl::init::derive_max_patches(
        max_pixels,
        vcfg.patch_size,
    )
    .0;
    let patches = (th as usize / vcfg.patch_size) * (tw as usize / vcfg.patch_size);
    ensure!(
        patches <= p_max,
        "image resizes to {patches} patches, more than the encoder's {p_max}"
    );
    Ok((th, tw))
}

/// Preprocess a single base64-encoded image for the Qwen3-VL encoder.
///
/// Returns:
/// - `pixels`: flat `f32` tensor shaped `[P, C × T × H_p × W_p]` where:
///   - `P = (H/patch_size) × (W/patch_size)` — number of patches
///   - `C = 3` channels, `T = temporal_patch_size` (image duplicated), `H_p = W_p = patch_size`
/// - `grid_h`: number of patches along height
/// - `grid_w`: number of patches along width
pub fn preprocess_image(data_uri: &str, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
    preprocess_image_with_max_pixels(data_uri, vcfg, None)
}

/// Preprocess an image with an optional max-pixels cap, matching vLLM-style
/// multimodal processor controls. `None` preserves Atlas' historical 1280px
/// long-side cap.
pub fn preprocess_image_with_max_pixels(
    data_uri: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    probe_image(data_uri, vcfg, max_pixels)?.preprocess(vcfg)
}

/// Decode and patch an image whose geometry [`probe_image`] already fixed.
fn preprocess_probed(probed: ProbedImage, vcfg: &VisionConfig) -> Result<(Vec<f32>, usize, usize)> {
    let max_pixels = probed.max_pixels;
    // Decodes at once are few: each holds its decoded image and its f32.
    let _slot = probe::decode_slot(probed.decoded_bytes);
    let img = decode_bytes(&probed.bytes, probed.fmt)?;
    drop(probed.bytes);
    let img = img.into_rgb8();
    let (orig_w, orig_h) = (img.width(), img.height());
    let (th, tw) = still_target(vcfg, orig_h, orig_w, max_pixels)?;
    // The probe sized the oriented image with the same function: the grid
    // admission charged is the grid decoded, exactly.
    ensure!(
        (th, tw) == probed.target,
        "decoded image geometry {th}x{tw} disagrees with its header's {}x{}",
        probed.target.0,
        probed.target.1
    );

    let img = if vcfg.is_glm5_next {
        let allow_upscale = vcfg.temporal_patch_size * orig_h as usize * (orig_w as usize)
            < glm_pixel_budget(vcfg, max_pixels, false).0;
        glm_resize_or_pad(&img, th, tw, allow_upscale)
    } else {
        // Resize with CatmullRom — closest BICUBIC match in the `image` crate,
        // matching HF's `Qwen2VLImageProcessor` which uses PIL resample=3 (BICUBIC).
        image::imageops::resize(&img, tw, th, image::imageops::FilterType::CatmullRom)
    };

    let ps = vcfg.patch_size;
    let tp = vcfg.temporal_patch_size;
    let grid_h = (th as usize) / ps;
    let grid_w = (tw as usize) / ps;
    let num_patches = grid_h * grid_w;
    // Flattened patch dim: C × temporal_patch_size × patch_size × patch_size
    let patch_dim = 3 * tp * ps * ps;
    let mut pixels = vec![0.0f32; num_patches * patch_dim];

    // Build patches. The temporal dimension is handled by duplicating the image `tp` times.
    // Layout: [P, C, T, Hp, Wp] → stored as [P, C*T*Hp*Wp] in row-major order.
    for (patch_idx, (ph, pw)) in patch_coordinates(vcfg, grid_h, grid_w)
        .into_iter()
        .enumerate()
    {
        for c in 0..3usize {
            for t in 0..tp {
                for py in 0..ps {
                    for px in 0..ps {
                        let pixel_y = ph * ps + py;
                        let pixel_x = pw * ps + px;
                        let norm = normalize_channel(
                            vcfg,
                            c,
                            img.get_pixel(pixel_x as u32, pixel_y as u32)[c],
                        );
                        // Offset into patch_dim: c*(T*Hp*Wp) + t*(Hp*Wp) + py*Wp + px
                        let off = c * (tp * ps * ps) + t * (ps * ps) + py * ps + px;
                        pixels[patch_idx * patch_dim + off] = norm;
                    }
                }
            }
        }
    }

    Ok((pixels, grid_h, grid_w))
}

/// Number of image pad tokens produced per image after the vision
/// encoder's spatial merger. Qwen3-VL / Qwen3.6 fold a 2×2 patch block
/// into a single token, so the embedding stream has
/// `(grid_h / sms) * (grid_w / sms)` rows — not `grid_h * grid_w`.
pub fn image_pad_count(grid_h: usize, grid_w: usize, spatial_merge_size: usize) -> usize {
    let sms = spatial_merge_size.max(1);
    (grid_h / sms) * (grid_w / sms)
}

mod probe;
pub use probe::{ProbedImage, decode_gated, probe_image};
pub mod fixtures;

#[cfg(test)]
#[path = "vision_preprocess_format_tests.rs"]
mod format_tests;
#[cfg(test)]
#[path = "vision_preprocess_tests.rs"]
mod tests;

/// Absolute long-side ceiling even under a `max_pixels` area bound (a 1xN
/// strip satisfies any area bound at an unbounded long side). Kept from the
/// pre-revision file for `target_size_for`'s callers.
pub(crate) const ABS_MAX_DIM: u32 = 4096;

/// - The long side never exceeds [`ABS_MAX_DIM`], bound or not.
/// - Both sides are multiples of `grid_unit = patch_size × spatial_merge_size`.
/// - Aspect ratio is preserved (rounded to nearest grid_unit).
/// - The image is never upscaled.
///
/// `max_pixels` is an area, matching the `size.longest_edge` /
/// `shortest_edge` convention HF's Qwen2VL/Qwen3VL processors use (both are
/// pixel counts, not edge lengths, despite the names). It comes from the
/// checkpoint's `preprocessor_config.json` or the operator's
/// `--vision-max-pixels`; the operator's value wins.
///
/// ★ `max_pixels` REPLACES the long-side clamp rather than combining with it.
/// Combining was the bug: `dim_scale.min(pixel_scale)` meant a checkpoint
/// permitting 4096² could never exceed 1280 on the long side, so the model's
/// own declared bound could only ever lower the resolution, never raise it.
/// `pub(crate)` alias name used by the video path — see [`target_size_for`].
pub(crate) fn target_size_for(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    target_size_with_max_pixels(orig_h, orig_w, grid_unit, max_pixels)
}

fn target_size_with_max_pixels(
    orig_h: u32,
    orig_w: u32,
    grid_unit: u32,
    max_pixels: Option<usize>,
) -> (u32, u32) {
    let long_side = orig_h.max(orig_w) as f32;
    let area = (orig_h as f32) * (orig_w as f32);
    let bound_scale = match max_pixels.filter(|&p| p > 0) {
        // Model- or operator-declared AREA bound governs.
        Some(p) => ((p as f32) / area).sqrt(),
        // Nothing declared: fall back to the historical long-side clamp.
        None => (FALLBACK_MAX_DIM as f32) / long_side,
    };
    // Safety net, always applied.
    let abs_scale = (ABS_MAX_DIM as f32) / long_side;
    let scale = bound_scale.min(abs_scale).min(1.0); // never upscale
    let target_h = ((orig_h as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;
    let target_w = ((orig_w as f32 * scale / grid_unit as f32).round() as u32).max(1) * grid_unit;
    (target_h, target_w)
}
