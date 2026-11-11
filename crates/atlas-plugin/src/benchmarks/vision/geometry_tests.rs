// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn the_measured_anchor_holds() {
    // 448x448 -> 196 merged tokens, measured against a live server 2026-08-14
    // (215 prompt_tokens minus 19 of chat template). 448 is on the grid and
    // above MIN_PIXELS, so neither the rounding mode nor the area bounds touch
    // it: if this one is wrong the arithmetic itself is.
    assert_eq!(expected_vision_tokens(448, 448, 16, 2), 196);
}

#[test]
fn token_count_is_quadratic_in_the_side() {
    // Catches an off-by-one in the merge divisor that a single hard-coded
    // expectation would not: doubling the side must quadruple the tokens.
    // (Starts at 448: 224x224 is below MIN_PIXELS and is scaled up.)
    let a = expected_vision_tokens(448, 448, 16, 2);
    let b = expected_vision_tokens(896, 896, 16, 2);
    let c = expected_vision_tokens(1792, 1792, 16, 2);
    assert_eq!(a, 196);
    assert_eq!(b, a * 4, "{a} -> {b} is not quadratic");
    assert_eq!(c, b * 4, "{b} -> {c} is not quadratic");
}

#[test]
fn snapping_is_applied_before_counting() {
    // 336 is NOT a multiple of 32; it snaps to 320 (10.5 grid units, half to
    // even). Counting from the raw 336 gives 110 tokens, from the snapped 320
    // gives 100. The ladder contains such sizes deliberately.
    assert_eq!(snap(336, 32), 320);
    assert_eq!(expected_vision_tokens(336, 336, 16, 2), 100);
    assert_ne!(
        expected_vision_tokens(336, 336, 16, 2),
        (336 / 16) * (336 / 16) / 4,
        "expectation must come from the SNAPPED size, not the raw one"
    );
}

#[test]
fn every_ladder_size_has_a_defined_expectation() {
    // Non-square and portrait included — a transposed grid_h/grid_w would
    // survive a square-only suite. Every value below was also served by the
    // engine at #237 (Atlas-Inf d7601f1c5, 2026-10-29 pair run).
    for (w, h, want) in [
        // Below MIN_PIXELS: 224x224 scales up to 256x256.
        (224u32, 224u32, 64u32),
        (336, 336, 100),
        (512, 384, 192),
        (640, 360, 220),
        (768, 768, 576),
        (1024, 576, 576),
        // 720 / 32 = 22.5 rounds to 22 (half to even).
        (1280, 720, 880),
        (480, 854, 405),
        // The only rung above the old clamp. See
        // `the_ladder_can_actually_detect_a_regression_to_the_old_clamp`.
        (1600, 900, 1400),
        // Tiny and strip rungs.
        (8, 8, 64),
        (64, 2048, 128),
    ] {
        assert_eq!(
            expected_vision_tokens(w, h, 16, 2),
            want,
            "{w}x{h} expectation drifted"
        );
    }
}

#[test]
fn portrait_and_landscape_of_the_same_shape_agree() {
    // Transposing must not change the count. A grid_h/grid_w swap in the
    // preprocessor is otherwise invisible on square fixtures.
    assert_eq!(
        expected_vision_tokens(512, 384, 16, 2),
        expected_vision_tokens(384, 512, 16, 2)
    );
}

#[test]
fn small_images_are_scaled_up_to_the_area_floor() {
    // The reference processor never lets an image cover less than
    // MIN_PIXELS: a sub-grid image is scaled up (ceiled to the grid), never
    // reduced to a 0x0 target.
    assert_eq!(reference_target(8, 8, 32), (256, 256));
    assert_eq!(reference_target(224, 224, 32), (256, 256));
    assert_eq!(reference_target(1, 1, 32), (256, 256));
    assert!(expected_vision_tokens(1, 1, 16, 2) >= 1);
    assert_eq!(snap(1, 32), 32);
    assert_eq!(snap(15, 32), 32);
}

#[test]
fn large_images_are_scaled_down_to_the_area_ceiling() {
    // 8192x8192 is 4x MAX_PIXELS: beta = 2, so 4096x4096.
    assert_eq!(reference_target(8192, 8192, 32), (4096, 4096));
    let (w, h) = reference_target(10_000, 6_000, 32);
    assert!((w as u64) * (h as u64) <= MAX_PIXELS, "{w}x{h}");
}

/// ★ The test that justifies the ladder's shape.
///
/// A gate that cannot fail on the defect it was written for is decoration.
/// This asserts the discriminating property directly: at least one fixture
/// must produce a DIFFERENT token count under the old 1280px long-side clamp
/// than under the checkpoint's declared area bound. Without the 1600x900 rung
/// this test fails, which is exactly the guard wanted — someone trimming the
/// ladder for runtime has to break this test to do it.
#[test]
fn the_ladder_can_actually_detect_a_regression_to_the_old_clamp() {
    /// What the retired unconditional clamp did: scale so the LONG side is
    /// 1280, never upscaling.
    fn under_old_clamp(w: u32, h: u32) -> u32 {
        let long = w.max(h) as f32;
        let s = (1280.0 / long).min(1.0);
        expected_vision_tokens(
            ((w as f32) * s).round() as u32,
            ((h as f32) * s).round() as u32,
            16,
            2,
        )
    }

    let ladder: Vec<(u32, u32)> = crate::benchmarks::vision::provision::FIXTURES
        .iter()
        .map(|&(_, _, w, h)| (w, h))
        .collect();
    let discriminating: Vec<(u32, u32)> = ladder
        .iter()
        .copied()
        .filter(|&(w, h)| under_old_clamp(w, h) != expected_vision_tokens(w, h, 16, 2))
        .collect();

    assert!(
        !discriminating.is_empty(),
        "every fixture in the ladder sits at or under the 1280px clamp, so a \
         regression to it would change no expectation and the geometry leg \
         would pass on a broken engine. Add a fixture above 1280 on the long \
         side."
    );

    // And name the numbers, so a future change to the fixture set that
    // weakens the margin is visible rather than silent.
    assert_eq!(under_old_clamp(1600, 900), 880);
    assert_eq!(expected_vision_tokens(1600, 900, 16, 2), 1400);
}

#[test]
fn the_rounding_mode_is_pinned() {
    // Python's `round` (the reference processor's) is half-to-EVEN: 336 and
    // 720 land on the EVEN grid unit. An engine on half-away-from-zero is off
    // by one grid unit at exactly these sides; pinned here so that lands as a
    // named failure rather than a mysterious token-count mismatch on a GPU box.
    assert_eq!(snap(224, 32), 224, "already exact");
    assert_eq!(snap(336, 32), 320, "10.5 rounds to even (10)");
    assert_eq!(snap(360, 32), 352, "11.25 rounds down");
    assert_eq!(snap(720, 32), 704, "22.5 rounds to even (22)");
    assert_eq!(snap(752, 32), 768, "23.5 rounds to even (24)");
    assert_eq!(snap(854, 32), 864, "26.69 rounds up");
}

#[test]
fn the_bounds_are_the_processors_not_the_images() {
    // The same 224x224 plane: scaled up under the image floor, kept under the video floor.
    assert_eq!(reference_target(224, 224, 32), (256, 256));
    assert_eq!(
        reference_target_within(224, 224, 32, 4_096, 25_165_824),
        (224, 224)
    );
}
