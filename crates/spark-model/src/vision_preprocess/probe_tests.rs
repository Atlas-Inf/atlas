// SPDX-License-Identifier: AGPL-3.0-only

use super::super::fixtures::{
    glm_config, header_only_png, header_only_png_typed, jpeg_with_orientation, png, qwen_config,
};
use super::*;

/// Qwen3.8's checkpoint bound (4096^2), which the encoder clamps to 2048^2.
const Q38: Option<usize> = Some(16_777_216);

#[test]
fn a_huge_declared_image_is_sized_from_its_header_without_decoding() {
    // 8000x8000 declared in ~60 bytes: decoded and patched it would be
    // hundreds of MiB. The probe sizes it (fitted to the encoder's 2048^2
    // capacity) from the header alone; the decode would fail on the
    // truncated data.
    let uri = header_only_png(8000, 8000);
    let t = std::time::Instant::now();
    let probed = probe_image(&uri, &qwen_config(), Q38).unwrap();
    assert!(
        t.elapsed().as_millis() < 500,
        "no pixel work: {:?}",
        t.elapsed()
    );
    assert!(probed.patches() <= 16_384, "{probed:?}");
    assert_eq!(probed.pad_count(2), probed.patches() / 4);
    assert!(
        probed.preprocess(&qwen_config()).is_err(),
        "the pixels were never there"
    );
}

#[test]
fn a_header_past_the_decoder_limits_is_refused_at_the_probe() {
    assert!(probe_image(&header_only_png(20_000, 100), &qwen_config(), Q38).is_err());
    // 16384^2 RGBA16 decodes to 2 GiB whatever it is resized to afterwards:
    // past the 640 MiB limit.
    let err = probe_image(
        &header_only_png_typed(16_384, 16_384, 16, 6),
        &qwen_config(),
        Q38,
    )
    .unwrap_err();
    assert!(format!("{err:#}").contains("MiB"), "{err:#}");
    // 8000x8000 RGB (183 MiB) is still admitted.
    assert!(probe_image(&header_only_png(8000, 8000), &qwen_config(), Q38).is_ok());
}

#[test]
fn stills_past_one_encoder_batch_are_resized_into_it() {
    // A 2880x1800 Retina screenshot used to come out 2880x1792 = 20160
    // patches: over the encoder's 16384, failing at prefill. It now fits,
    // as HF's smart_resize sizes it at the capacity.
    for (w, h) in [(2880u32, 1800u32), (2064, 2032), (1800, 2880)] {
        let probed = probe_image(&header_only_png(w, h), &qwen_config(), Q38).unwrap();
        assert!(probed.patches() <= 16_384, "{w}x{h}: {probed:?}");
        assert!(
            probed.patches() > 15_000,
            "{w}x{h} kept its detail: {probed:?}"
        );
    }
    assert_eq!(
        still_target(&qwen_config(), 2032, 2064, Q38).unwrap(),
        (2048, 2048)
    );
    // No declared bound at patch 14: the 1280px fallback is 92x92 = 8464
    // patches, past the 6400 fallback capacity, so it is fitted too.
    let mut p14 = qwen_config();
    p14.patch_size = 14;
    let (th, tw) = still_target(&p14, 1280, 1280, None).unwrap();
    assert!((th as usize / 14) * (tw as usize / 14) <= 6400, "{th}x{tw}");
}

#[test]
fn a_full_decode_matches_its_probe() {
    let uri = png(2064, 2032);
    let probed = probe_image(&uri, &qwen_config(), Q38).unwrap();
    let (gh, gw) = (probed.grid_h, probed.grid_w);
    let (pixels, ph, pw) = probed.preprocess(&qwen_config()).unwrap();
    assert_eq!((ph, pw), (gh, gw));
    assert_eq!(pixels.len(), ph * pw * 3 * 2 * 16 * 16);
    assert_eq!(
        preprocess_image_with_max_pixels(&uri, &qwen_config(), Q38)
            .unwrap()
            .1,
        ph,
        "the one-call path is the probe plus the decode"
    );
}

#[test]
fn a_rotated_photo_is_probed_as_the_decode_sees_it() {
    // A 3:2 photo stored landscape and tagged Orientation=6 (shot portrait).
    // GLM's sizing is not symmetric in its sides, so a probe of the stored
    // sides disagreed with the upright decode and the request got a 400.
    let uri = jpeg_with_orientation(600, 400, 6);
    for (cfg, max_pixels) in [
        (glm_config(), Some(2 * 300 * 200)), // small: the budget search runs
        (glm_config(), None),
        (qwen_config(), Some(300 * 200)),
        (qwen_config(), Q38),
    ] {
        let probed = probe_image(&uri, &cfg, max_pixels).unwrap();
        let (gh, gw) = (probed.grid_h, probed.grid_w);
        assert!(gh > gw, "upright is portrait: {gh}x{gw} ({max_pixels:?})");
        let (_, ph, pw) = probed.preprocess(&cfg).unwrap();
        assert_eq!(
            (ph, pw),
            (gh, gw),
            "glm={} {max_pixels:?}",
            cfg.is_glm5_next
        );
    }
    // Orientations that do not turn the image keep its sides.
    let flipped = probe_image(&jpeg_with_orientation(600, 400, 3), &qwen_config(), Q38).unwrap();
    assert!(flipped.grid_w > flipped.grid_h);
}

#[test]
fn a_probe_that_disagrees_with_its_decode_is_refused() {
    let mut probed = probe_image(&png(640, 480), &qwen_config(), Q38).unwrap();
    probed.target = (probed.target.1, probed.target.0);
    let err = probed.preprocess(&qwen_config()).unwrap_err();
    assert!(
        format!("{err:#}").contains("disagrees with its header"),
        "{err:#}"
    );
}

#[test]
fn a_108mp_phone_photo_is_admitted_and_resized_to_the_encoder() {
    // 12000x9000 RGB decodes to 309 MiB: past the large-decode mark, inside
    // the 640 MiB limit. It is sized, not refused, and fits one batch; a
    // 200 MP 16320x12240 (571 MiB) too.
    for (w, h) in [(12_000u32, 9000u32), (16_320, 12_240)] {
        let probed = probe_image(&header_only_png(w, h), &qwen_config(), Q38).unwrap();
        assert!(probed.decoded_bytes > DECODE_LARGE, "{w}x{h}");
        assert!(
            probed.patches() <= 16_384 && probed.patches() > 15_000,
            "{probed:?}"
        );
    }
}

#[test]
fn the_large_decode_gate_holds_at_most_two() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static GATE: Gate = Gate::new(|| 2);
    static NOW: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    std::thread::scope(|s| {
        for _ in 0..8 {
            s.spawn(|| {
                for _ in 0..5 {
                    let _slot = GATE.acquire();
                    let n = NOW.fetch_add(1, Ordering::SeqCst) + 1;
                    PEAK.fetch_max(n, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                    NOW.fetch_sub(1, Ordering::SeqCst);
                }
            });
        }
    });
    let peak = PEAK.load(Ordering::SeqCst);
    assert!((1..=2).contains(&peak), "peak {peak}");
    // A large decode takes the large gate and a slot; a small one only a
    // slot. One at a time: a single-core host has one slot.
    let big = decode_slot(DECODE_LARGE + 1);
    assert!(big._large.is_some());
    drop(big);
    assert!(decode_slot(DECODE_LARGE)._large.is_none());
}
