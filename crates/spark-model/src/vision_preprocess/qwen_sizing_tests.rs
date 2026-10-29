// SPDX-License-Identifier: AGPL-3.0-only

//! [`qwen_fit`] against HF's `smart_resize`. Every expected size below was
//! produced by transformers 5.3.0's own functions (imported, not
//! reimplemented), at the Qwen3.8 checkpoint's processor configs with
//! `max_pixels` lowered to the encoder's 16384-patch capacity:
//!
//! ```python
//! from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
//! smart_resize(h, w, factor=32, min_pixels=65536, max_pixels=min(16777216, 16384 * 256))
//! from transformers.models.qwen3_vl.video_processing_qwen3_vl import smart_resize
//! smart_resize(n, h, w, temporal_factor=2, factor=32, min_pixels=4096,
//!              max_pixels=min(25165824, n * 16384 * 256))
//! ```

use super::super::fixtures::qwen_config;
use super::*;
use atlas_core::config::PixelBounds;

/// Qwen3.8's `preprocessor_config.json`: `size.longest_edge`.
const Q38_MAX: Option<usize> = Some(16_777_216);

/// The Qwen3.8 checkpoint's image and video processor bounds.
fn q38() -> VisionConfig {
    VisionConfig {
        min_pixels: Some(65_536),
        video_pixels: Some(PixelBounds {
            min_pixels: 4096,
            max_pixels: 25_165_824,
        }),
        ..qwen_config()
    }
}

#[test]
fn stills_match_hf_smart_resize() {
    // (h, w, HF). The ones past the capacity note what HF gives uncapped.
    const TABLE: &[(u32, u32, (u32, u32))] = &[
        (1, 1, (256, 256)), // min_pixels upscale (Atlas used to send 32x32)
        (16, 16, (256, 256)),
        (32, 32, (256, 256)),
        (100, 50, (384, 192)),
        (50, 100, (192, 384)),
        (48, 80, (224, 352)),
        (80, 48, (352, 224)),
        (224, 224, (256, 256)),
        (255, 255, (256, 256)),
        (256, 256, (256, 256)), // exactly min_pixels
        (300, 200, (320, 224)),
        (400, 300, (384, 288)),
        (10, 2000, (32, 3648)), // exactly 200:1
        (2000, 10, (3648, 32)),
        (20, 4000, (32, 4000)),
        (33, 6600, (32, 6592)),
        (1080, 1920, (1088, 1920)),
        (1920, 1080, (1920, 1088)),
        (1344, 1344, (1344, 1344)),
        (2048, 2048, (2048, 2048)), // exactly the capacity
        (2032, 2064, (2048, 2048)), // half-to-even: 63.5 -> 64, 64.5 -> 64
        (2064, 2032, (2048, 2048)),
        (2880, 1800, (2560, 1600)),   // HF uncapped: 2880x1792
        (1800, 2880, (1600, 2560)),   // HF uncapped: 1792x2880
        (4096, 4096, (2048, 2048)),   // HF uncapped: 4096x4096
        (5293, 1337, (4064, 1024)),   // HF uncapped: 5280x1344
        (1000, 8000, (704, 5792)),    // HF uncapped: 992x8000
        (8000, 8000, (2048, 2048)),   // HF uncapped: 4096x4096
        (12240, 16320, (1760, 2336)), // HF uncapped: 3520x4704
    ];
    let cfg = q38();
    for &(h, w, want) in TABLE {
        assert_eq!(
            qwen_fit(&cfg, h, w, Q38_MAX, None).unwrap(),
            want,
            "{h}x{w}"
        );
    }
}

#[test]
fn past_200_to_1_is_refused_as_hf_does() {
    for (h, w) in [(10u32, 2001u32), (2001, 10), (1, 201)] {
        let err = qwen_fit(&q38(), h, w, Q38_MAX, None).unwrap_err();
        assert!(
            format!("{err:#}").contains("aspect ratio"),
            "{h}x{w}: {err:#}"
        );
    }
}

#[test]
fn an_operator_bound_is_hf_at_that_bound() {
    // `--vision-max-pixels 1048576` against the checkpoint's min: HF with
    // max_pixels = 1048576 (smart_resize(1080, 1920, 32, 65536, 1048576)).
    assert_eq!(
        qwen_fit(&q38(), 1080, 1920, Some(1_048_576), None).unwrap(),
        (768, 1344)
    );
    // Below the checkpoint's min (65536): the upscale cannot pass the
    // encoder's 256 patches, so the strip is shortened to fit.
    let fit = qwen_fit(&q38(), 100, 50, Some(65_536), None).unwrap();
    assert!((fit.0 / 16) * (fit.1 / 16) <= 256, "{fit:?}");
}

#[test]
fn no_declared_min_upscales_nothing_past_one_grid_unit() {
    // An area bound without a min (an operator flag on a checkpoint with
    // no processor config): HF's arithmetic at min = one 32x32 unit.
    let cfg = qwen_config();
    assert_eq!(qwen_fit(&cfg, 64, 64, Q38_MAX, None).unwrap(), (64, 64));
    assert_eq!(qwen_fit(&cfg, 1, 1, Q38_MAX, None).unwrap(), (32, 32));
}

#[test]
fn no_declared_bound_keeps_the_historical_sizing() {
    // A checkpoint with no processor config and no operator flag: the long
    // side clamped to 1280, never upscaled, nearest grid unit.
    let cfg = q38();
    for (h, w, want) in [
        (1344u32, 1344u32, (1280u32, 1280u32)),
        (64, 64, (64, 64)),
        (100, 150, (96, 160)),
        (2000, 3000, (864, 1280)),
        (1080, 1920, (736, 1280)),
    ] {
        assert_eq!(qwen_fit(&cfg, h, w, None, None).unwrap(), want, "{h}x{w}");
    }
}

#[test]
fn clips_match_hf_video_smart_resize() {
    // (frames, h, w, HF). A clip is sized over all its frames.
    const TABLE: &[(usize, u32, u32, (u32, u32))] = &[
        (4, 1080, 1920, (1088, 1920)),
        (32, 1080, 1920, (640, 1152)),
        (64, 720, 1280, (448, 832)),
        (8, 48, 64, (64, 64)),
        (2, 32, 32, (64, 64)),
        (768, 360, 640, (128, 224)),
        (4, 4096, 4096, (2048, 2048)), // HF uncapped: 2496x2496
    ];
    let cfg = q38();
    for &(n, h, w, want) in TABLE {
        assert_eq!(
            qwen_fit(&cfg, h, w, Q38_MAX, Some(n)).unwrap(),
            want,
            "{n} frames of {h}x{w}"
        );
    }
    // HF's video processor refuses frames smaller than one grid unit.
    assert!(qwen_fit(&cfg, 16, 16, Q38_MAX, Some(4)).is_err());
}

#[test]
fn clips_without_video_bounds_size_each_frame_like_a_still() {
    let cfg = VisionConfig {
        video_pixels: None,
        ..q38()
    };
    assert_eq!(
        qwen_fit(&cfg, 1080, 1920, Q38_MAX, Some(32)).unwrap(),
        qwen_fit(&cfg, 1080, 1920, Q38_MAX, None).unwrap()
    );
}

#[test]
fn every_target_is_a_grid_inside_the_encoder() {
    let cfg = q38();
    for max_pixels in [None, Q38_MAX, Some(1_000_003), Some(65_536)] {
        let p_max =
            crate::layers::vision_encoder::enc_impl::init::derive_max_patches(max_pixels, 16).0;
        for h in (17..6000u32).step_by(211) {
            for w in [33u32, 640, 1337, 1920, 2049, 4096, 5293] {
                for frames in [None, Some(2), Some(64)] {
                    let Ok((th, tw)) = qwen_fit(&cfg, h, w, max_pixels, frames) else {
                        // HF refuses these too: past 200:1, or a frame
                        // under one grid unit.
                        let video = frames.is_some();
                        assert!(h.max(w) > 200 * h.min(w) || (video && h.min(w) < 32));
                        continue;
                    };
                    assert!(th % 32 == 0 && tw % 32 == 0 && th > 0 && tw > 0);
                    let patches = (th as usize / 16) * (tw as usize / 16);
                    assert!(patches <= p_max, "{h}x{w} {max_pixels:?} {frames:?}");
                }
            }
        }
    }
}
