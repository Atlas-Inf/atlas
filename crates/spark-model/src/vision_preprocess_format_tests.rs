// SPDX-License-Identifier: AGPL-3.0-only

//! Image formats `decode_image` reads, and the 400 for the ones it does not.

use super::*;
use image::{ImageEncoder, ImageFormat};

/// 32×32, red top half and blue bottom half.
fn halves() -> RgbImage {
    RgbImage::from_fn(32, 32, |_, y| {
        if y < 16 {
            Rgb([220, 20, 20])
        } else {
            Rgb([20, 40, 220])
        }
    })
}

fn uri(mime: &str, bytes: &[u8]) -> String {
    use base64::Engine as _;
    format!(
        "data:{mime};base64,{}",
        base64::engine::general_purpose::STANDARD.encode(bytes)
    )
}

fn encoded(fmt: ImageFormat) -> Vec<u8> {
    let img = halves();
    let mut out = Vec::new();
    if fmt == ImageFormat::WebP {
        // `write_to` has no WebP encoder without the lossy feature; the codec's own does.
        image::codecs::webp::WebPEncoder::new_lossless(&mut out)
            .write_image(img.as_raw(), 32, 32, image::ExtendedColorType::Rgb8)
            .expect("encode webp");
    } else {
        // Icons carry an alpha channel; the ICO reader wants RGBA entries.
        let img = match fmt {
            ImageFormat::Ico => DynamicImage::ImageRgba8(DynamicImage::ImageRgb8(img).to_rgba8()),
            _ => DynamicImage::ImageRgb8(img),
        };
        img.write_to(&mut std::io::Cursor::new(&mut out), fmt)
            .expect("encode");
    }
    out
}

fn assert_halves(img: &DynamicImage, what: &str) {
    let img = img.to_rgb8();
    assert_eq!((img.width(), img.height()), (32, 32), "{what}");
    assert_eq!(
        img.get_pixel(5, 3),
        &Rgb([220, 20, 20]),
        "{what}: top should be red"
    );
    assert_eq!(
        img.get_pixel(5, 28),
        &Rgb([20, 40, 220]),
        "{what}: bottom should be blue"
    );
}

#[test]
fn webp_bmp_tiff_and_ico_decode_like_png() {
    for (fmt, mime) in [
        (ImageFormat::Png, "image/png"),
        (ImageFormat::WebP, "image/webp"),
        (ImageFormat::Bmp, "image/bmp"),
        (ImageFormat::Tiff, "image/tiff"),
        (ImageFormat::Ico, "image/x-icon"),
    ] {
        let img =
            decode_image(&uri(mime, &encoded(fmt))).unwrap_or_else(|e| panic!("{fmt:?}: {e:#}"));
        assert_halves(&img, &format!("{fmt:?}"));
    }
}

#[test]
fn the_format_comes_from_the_bytes_not_the_declared_type() {
    // Clients mislabel: a WebP sent as image/jpeg still decodes.
    let img = decode_image(&uri("image/jpeg", &encoded(ImageFormat::WebP))).expect("decode");
    assert_halves(&img, "webp labelled jpeg");
}

#[test]
fn an_unreadable_format_is_refused_by_name() {
    // An AVIF file's leading `ftyp` box: the format is recognized but not built in.
    let avif = [
        0, 0, 0, 0x1c, b'f', b't', b'y', b'p', b'a', b'v', b'i', b'f', 0, 0, 0, 0,
    ];
    let err = format!("{:#}", decode_image(&uri("image/avif", &avif)).unwrap_err());
    assert!(
        err.contains("Avif images are not supported") && err.contains("WebP"),
        "{err}"
    );

    // Bytes no reader recognizes (HEIC is not probed by the image crate).
    let heic = [
        0, 0, 0, 0x18, b'f', b't', b'y', b'p', b'h', b'e', b'i', b'c', 0, 0, 0, 0,
    ];
    let err = format!("{:#}", decode_image(&uri("image/heic", &heic)).unwrap_err());
    assert!(
        err.contains("unrecognized image format") && err.contains("JPEG"),
        "{err}"
    );
}
