// SPDX-License-Identifier: AGPL-3.0-only

//! Synthetic images and vision configs for the preprocessing and admission
//! tests of this crate and `spark-server`. Pure functions, no state.

use atlas_core::config::VisionConfig;

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = (crc >> 1) ^ (0xEDB8_8320 & (crc & 1).wrapping_neg());
        }
    }
    !crc
}

fn data_uri(mime: &str, bytes: &[u8]) -> String {
    let b64 = base64::Engine::encode(&base64::engine::general_purpose::STANDARD, bytes);
    format!("data:{mime};base64,{b64}")
}

/// A ~60-byte PNG that DECLARES `w x h` 8-bit RGB but carries a truncated
/// IDAT: its header reads fine, and any attempt to decode it fails.
pub fn header_only_png(w: u32, h: u32) -> String {
    header_only_png_typed(w, h, 8, 2)
}

/// [`header_only_png`] with the IHDR's bit depth and color type (6 = RGBA).
pub fn header_only_png_typed(w: u32, h: u32, bit_depth: u8, color_type: u8) -> String {
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        png.extend((data.len() as u32).to_be_bytes());
        let start = png.len();
        png.extend(kind);
        png.extend(data);
        let crc = crc32(&png[start..]);
        png.extend(crc.to_be_bytes());
    };
    let mut ihdr = [w.to_be_bytes(), h.to_be_bytes()].concat();
    ihdr.extend([bit_depth, color_type, 0, 0, 0]);
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &[0x78, 0x9c]);
    chunk(b"IEND", &[]);
    data_uri("image/png", &png)
}

fn solid(w: u32, h: u32) -> image::DynamicImage {
    image::DynamicImage::ImageRgb8(image::RgbImage::from_fn(w, h, |x, y| {
        image::Rgb([(x % 251) as u8, (y % 241) as u8, 90])
    }))
}

/// A real, decodable PNG of `w x h`.
pub fn png(w: u32, h: u32) -> String {
    let mut buf = std::io::Cursor::new(Vec::new());
    solid(w, h)
        .write_to(&mut buf, image::ImageFormat::Png)
        .expect("encode png");
    data_uri("image/png", buf.get_ref())
}

/// A real JPEG of stored size `w x h` tagged with EXIF `orientation`
/// (1..=8; 6 = "rotate 90 degrees clockwise to display").
pub fn jpeg_with_orientation(w: u32, h: u32, orientation: u16) -> String {
    let mut buf = std::io::Cursor::new(Vec::new());
    solid(w, h)
        .write_to(&mut buf, image::ImageFormat::Jpeg)
        .expect("encode jpeg");
    let jpeg = buf.into_inner();
    // APP1 "Exif": little-endian TIFF, IFD0 with one entry 0x0112 SHORT.
    let mut exif = b"Exif\0\0II*\0\x08\0\0\0\x01\0\x12\x01\x03\0\x01\0\0\0".to_vec();
    exif.extend(orientation.to_le_bytes());
    exif.extend([0, 0, 0, 0, 0, 0]);
    let mut out = jpeg[..2].to_vec(); // SOI
    out.extend([0xFF, 0xE1]);
    out.extend(((exif.len() + 2) as u16).to_be_bytes());
    out.extend(exif);
    out.extend(&jpeg[2..]);
    data_uri("image/jpeg", &out)
}

/// The Qwen3.8 tower's geometry (patch 16, merge 2), no pixel bound.
pub fn qwen_config() -> VisionConfig {
    VisionConfig {
        is_glm5_next: false,
        in_channels: 3,
        depth: 27,
        hidden_size: 1152,
        num_heads: 16,
        patch_size: 16,
        temporal_patch_size: 2,
        spatial_merge_size: 2,
        intermediate_size: 4304,
        out_hidden_size: 2048,
        projection_intermediate_size: 0,
        rms_norm_eps: 1e-6,
        swiglu_limit: 0.0,
        deepstack_visual_indexes: vec![8, 16, 24],
        image_pad_token_id: 151_655,
        video_pad_token_id: 151_656,
        image_start_token_id: 0,
        image_end_token_id: 0,
        video_start_token_id: 0,
        video_end_token_id: 0,
        max_pixels: None,
        token_budgets: Default::default(),
        min_pixels: None,
        video_pixels: None,
        is_qwen3vl_video: false,
        omni: None,
    }
}

/// The native GLM-5.3 tower's preprocessing geometry (patch 14, merge 2).
pub fn glm_config() -> VisionConfig {
    VisionConfig {
        is_glm5_next: true,
        patch_size: 14,
        ..qwen_config()
    }
}

/// Four distinct 64x64 GIF frames, each held for half a second.
pub fn short_gif() -> String {
    use image::codecs::gif::GifEncoder;
    use image::{Delay, Frame, Rgba, RgbaImage};
    let mut bytes = Vec::new();
    {
        let mut encoder = GifEncoder::new(&mut bytes);
        for i in 0..4u8 {
            let color = 20 + 37 * i;
            let frame = RgbaImage::from_pixel(64, 64, Rgba([color, 40, 255 - color, 255]));
            encoder
                .encode_frame(Frame::from_parts(
                    frame,
                    0,
                    0,
                    Delay::from_numer_denom_ms(500, 1),
                ))
                .expect("encode GIF fixture");
        }
    }
    data_uri("image/gif", &bytes)
}
