// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result, bail, ensure};
use atlas_core::config::VisionConfig;
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits, Rgb, RgbImage};

use super::{DECODE_MAX_ALLOC, patchify_rgb, qwen_fit, validate_geometry};

pub const MAX_SCORE_IMAGE_BYTES: usize = 8_000_000;
pub const MAX_SCORE_IMAGE_PIXELS: u64 = 16_000_000;

pub fn preprocess_score_image(
    bytes: &[u8],
    format: &str,
    vcfg: &VisionConfig,
    max_pixels: Option<usize>,
) -> Result<(Vec<f32>, usize, usize)> {
    validate_geometry(vcfg)?;
    let (img, width, height) = decode_score_image_rgb(bytes, format)?;
    let (target_h, target_w) = qwen_fit(vcfg, height, width, max_pixels, None)?;
    patchify_rgb(img, vcfg, target_h, target_w, max_pixels)
}

fn decode_score_image_rgb(bytes: &[u8], format: &str) -> Result<(RgbImage, u32, u32)> {
    check_encoded_size(bytes.len())?;
    let declared = match format {
        "png" => ImageFormat::Png,
        "jpeg" => ImageFormat::Jpeg,
        "webp" => ImageFormat::WebP,
        other => bail!("unsupported image format {other:?}; expected png, jpeg or webp"),
    };
    let actual = image::guess_format(bytes).context("image format could not be identified")?;
    ensure!(
        actual == declared,
        "image format is {actual:?}, but the request declares {format:?}"
    );

    let mut reader = ImageReader::new(std::io::Cursor::new(bytes));
    reader.set_format(actual);
    let mut limits = Limits::default();
    limits.max_alloc = Some(DECODE_MAX_ALLOC);
    reader.limits(limits);
    let decoder = reader
        .into_decoder()
        .context("image could not be decoded")?;
    let (width, height) = decoder.dimensions();
    check_source_pixels(width, height)?;
    ensure!(
        decoder.total_bytes() <= DECODE_MAX_ALLOC,
        "decoded image allocation exceeds the decoder limit"
    );
    let img = match actual {
        #[cfg(feature = "pil-jpeg")]
        ImageFormat::Jpeg => decode_jpeg_pil(bytes, width, height)?,
        _ => pil_rgb(DynamicImage::from_decoder(decoder).context("image could not be decoded")?),
    };
    Ok((img, width, height))
}

/// A JPEG through libjpeg-turbo, as PIL decodes it.
#[cfg(feature = "pil-jpeg")]
fn decode_jpeg_pil(bytes: &[u8], width: u32, height: u32) -> Result<RgbImage> {
    ensure!(
        u64::from(width) * u64::from(height) * 3 <= DECODE_MAX_ALLOC,
        "decoded image allocation exceeds the decoder limit"
    );
    let decoded = turbojpeg::decompress(bytes, turbojpeg::PixelFormat::RGB)
        .context("image could not be decoded")?;
    ensure!(
        decoded.width == width as usize && decoded.height == height as usize,
        "JPEG decoder dimensions changed during decode"
    );
    RgbImage::from_raw(width, height, decoded.pixels)
        .context("JPEG decoder returned an invalid RGB buffer")
}

fn pil_rgb(image: DynamicImage) -> RgbImage {
    match image {
        DynamicImage::ImageLuma16(image) => {
            RgbImage::from_fn(image.width(), image.height(), |x, y| {
                let value = image.get_pixel(x, y)[0].min(255) as u8;
                Rgb([value; 3])
            })
        }
        // PIL opens 16-bit LA / RGB / RGBA PNGs with the high byte of each
        // sample (raw modes "LA;16B", "RGB;16B", "RGBA;16B"); only 16-bit grey
        // ("I;16") clips at 255 when converted to RGB.
        DynamicImage::ImageLumaA16(image) => {
            RgbImage::from_fn(image.width(), image.height(), |x, y| {
                Rgb([(image.get_pixel(x, y)[0] >> 8) as u8; 3])
            })
        }
        DynamicImage::ImageRgb16(image) => {
            RgbImage::from_fn(image.width(), image.height(), |x, y| {
                let pixel = image.get_pixel(x, y);
                Rgb([
                    (pixel[0] >> 8) as u8,
                    (pixel[1] >> 8) as u8,
                    (pixel[2] >> 8) as u8,
                ])
            })
        }
        DynamicImage::ImageRgba16(image) => {
            RgbImage::from_fn(image.width(), image.height(), |x, y| {
                let pixel = image.get_pixel(x, y);
                Rgb([
                    (pixel[0] >> 8) as u8,
                    (pixel[1] >> 8) as u8,
                    (pixel[2] >> 8) as u8,
                ])
            })
        }
        other => other.into_rgb8(),
    }
}

fn check_encoded_size(size: usize) -> Result<()> {
    ensure!(
        size <= MAX_SCORE_IMAGE_BYTES,
        "image decodes to {size} bytes, over the {MAX_SCORE_IMAGE_BYTES} byte limit"
    );
    Ok(())
}

fn check_source_pixels(width: u32, height: u32) -> Result<()> {
    let pixels = u64::from(width) * u64::from(height);
    ensure!(
        width > 0 && height > 0 && pixels <= MAX_SCORE_IMAGE_PIXELS,
        "image has {pixels} source pixels, over the {MAX_SCORE_IMAGE_PIXELS} pixel limit"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use sha2::{Digest, Sha256};

    use super::*;

    /// PIL 12.1 on 1x1 16-bit PNGs: LA / RGB keep each sample's high byte
    /// (0x1234 -> 18, 0x8001 -> 128, 0xff00 -> 255); grey clips at 255.
    #[test]
    fn sixteen_bit_pngs_decode_like_pil() {
        fn png<P: image::PixelWithColorType<Subpixel = u16>>(
            img: image::ImageBuffer<P, Vec<u16>>,
        ) -> Vec<u8>
        where
            [P::Subpixel]: image::EncodableLayout,
        {
            let mut out = std::io::Cursor::new(Vec::new());
            img.write_to(&mut out, ImageFormat::Png).unwrap();
            out.into_inner()
        }
        let rgb = png(image::ImageBuffer::from_pixel(
            1,
            1,
            image::Rgb([0x1234u16, 0x8001, 0xff00]),
        ));
        let (img, ..) = decode_score_image_rgb(&rgb, "png").unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [18, 128, 255]);
        let la = png(image::ImageBuffer::from_pixel(
            1,
            1,
            image::LumaA([0x1234u16, 0xffff]),
        ));
        let (img, ..) = decode_score_image_rgb(&la, "png").unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [18, 18, 18]);
        let grey = png(image::ImageBuffer::from_pixel(
            1,
            1,
            image::Luma([0x1234u16]),
        ));
        let (img, ..) = decode_score_image_rgb(&grey, "png").unwrap();
        assert_eq!(img.get_pixel(0, 0).0, [255, 255, 255]);
    }

    const FIXTURES: [(&str, &str, &str); 4] = [
        (
            "12_rgba_224.png",
            "png",
            "8293568f44664462cc22639658068c9e760e31778a5031e7bf351fc2b353e587",
        ),
        (
            "13_gray_224.jpg",
            "jpeg",
            "fcf9b498da01d01fd95d9f69292f776883cc639ea1f7b16f9b8d5c8aa18a9e9a",
        ),
        (
            "14_png16_224.png",
            "png",
            "c650c424173f1ff80e76451edbd4adf1678f6663c2d5ad8e0ba72f6b7c9eab1a",
        ),
        (
            "15_exif_rot90_224.jpg",
            "jpeg",
            "2bf78d7d164bca11a30c72f41e20fc2f007f95d4e22a024892fa010302b7e5b5",
        ),
    ];

    #[test]
    fn rgb_decode_matches_pil_convert_without_exif_transpose() {
        for (name, format, expected) in FIXTURES {
            let path = format!(
                "{}/../atlas-plugin/assets/vision/{name}",
                env!("CARGO_MANIFEST_DIR")
            );
            let bytes = std::fs::read(path).unwrap();
            let (rgb, width, height) = decode_score_image_rgb(&bytes, format).unwrap();
            assert_eq!((width, height), (224, 224), "{name}");
            assert_eq!(rgb.dimensions(), (224, 224), "{name}");
            // The JPEG digests are libjpeg-turbo's pixels (PIL's).
            if format == "jpeg" && !cfg!(feature = "pil-jpeg") {
                continue;
            }
            let digest = Sha256::digest(rgb.as_raw());
            let hex = digest
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>();
            assert_eq!(hex, expected, "{name}");
        }
    }

    #[test]
    fn source_pixel_limit_is_checked_before_decode() {
        assert!(check_source_pixels(4_001, 4_000).is_err());
        assert!(check_source_pixels(4_000, 4_000).is_ok());
        assert!(check_source_pixels(0, 1).is_err());
    }

    #[test]
    fn decoded_image_byte_limit_is_enforced() {
        assert!(check_encoded_size(MAX_SCORE_IMAGE_BYTES).is_ok());
        assert!(check_encoded_size(MAX_SCORE_IMAGE_BYTES + 1).is_err());
    }

    #[test]
    fn decode_rejects_format_mismatch_and_corrupt_bytes() {
        assert!(decode_score_image_rgb(b"not an image", "png").is_err());
        let path = format!(
            "{}/../atlas-plugin/assets/vision/12_rgba_224.png",
            env!("CARGO_MANIFEST_DIR")
        );
        let bytes = std::fs::read(path).unwrap();
        assert!(decode_score_image_rgb(&bytes, "jpeg").is_err());
    }
}
