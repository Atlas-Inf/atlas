// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3's geometry against the serving processor's own numbers.

use super::*;

/// `(h, w) -> (resized h, resized w)` from the serving processor's
/// `smart_resize` (vLLM `glm5next.py`, image budget 16..8000 tokens), run on
/// the reference Python.
const GLM_IMAGE_PARITY: [((u32, u32), (u32, u32)); 12] = [
    ((1, 1), (112, 112)),
    ((100, 37), (196, 84)),
    ((300, 400), (308, 420)),
    ((1080, 1920), (1092, 1932)),
    ((1920, 1080), (1932, 1092)),
    ((4096, 4096), (2492, 2492)),
    ((512, 8192), (532, 8204)),
    ((8192, 512), (8204, 532)),
    // Exactly the 8000-token boundary, and one row past it.
    ((2240, 2800), (2240, 2800)),
    ((2241, 2800), (2240, 2800)),
    ((2500, 2600), (2436, 2548)),
    ((56, 20000), (56, 20020)),
];

#[test]
fn glm_image_geometry_matches_the_reference_processor() {
    let cfg = glm_budget_cfg();
    for ((h, w), want) in GLM_IMAGE_PARITY {
        let got = glm_target_size(&cfg, h, w, cfg.temporal_patch_size, None, false).unwrap();
        assert_eq!(got, want, "{h}x{w}");
    }
}

/// `(frames, h, w) -> (resized h, resized w, tokens)` from the reference
/// `smart_resize` under the serving video cap (30000 tokens). Odd frame counts
/// budget `round(t / 2)` groups half-to-even but encode `ceil(t / 2)`, which
/// is how five 8K frames carry 44988 tokens.
const GLM_VIDEO_PARITY: [((usize, u32, u32), (u32, u32, usize)); 13] = [
    ((1, 1080, 1920), (1092, 1932, 2691)),
    ((2, 1080, 1920), (1092, 1932, 2691)),
    ((2, 2160, 3840), (2184, 3864, 10764)),
    ((2, 8000, 8000), (4844, 4844, 29929)),
    ((3, 4320, 7680), (2576, 4564, 29992)),
    ((5, 4320, 7680), (2576, 4564, 44988)),
    ((9, 4320, 7680), (1820, 3220, 37375)),
    ((120, 1080, 1920), (476, 812, 29580)),
    ((121, 1080, 1920), (476, 812, 30073)),
    ((240, 1080, 1920), (336, 560, 28800)),
    ((2048, 1080, 1920), (112, 196, 28672)),
    ((4, 10, 10), (84, 84, 18)),
    ((2, 1, 1), (112, 112, 16)),
];

#[test]
fn glm_video_geometry_matches_the_reference_processor() {
    let cfg = glm_budget_cfg();
    for ((frames, h, w), (want_h, want_w, want_tokens)) in GLM_VIDEO_PARITY {
        let (got_h, got_w) = glm_target_size(&cfg, h, w, frames, None, true).unwrap();
        let tokens = frames.div_ceil(cfg.temporal_patch_size)
            * (got_h as usize / 28)
            * (got_w as usize / 28);
        assert_eq!(
            (got_h, got_w, tokens),
            (want_h, want_w, want_tokens),
            "{frames} frames of {h}x{w}"
        );
    }
}

#[test]
fn glm_budgets_come_from_the_processor_config_and_size_the_tower() {
    let mut cfg = glm_budget_cfg();
    // The checkpoint's video budget is 240000 tokens; serving caps it at
    // 30000, so one two-frame group may hold 120000 patches.
    assert_eq!(glm_max_sequence_patches(&cfg), 120_000);
    assert_eq!(glm_max_item_tokens(&cfg), 45_000);
    assert_eq!(
        glm_pixel_budget(&cfg, None, false),
        (16 * 1568, 8000 * 1568)
    );
    cfg.token_budgets.image = (4, 100);
    cfg.token_budgets.video = (4, 200);
    assert_eq!(glm_pixel_budget(&cfg, None, false), (4 * 1568, 100 * 1568));
    assert_eq!(glm_max_sequence_patches(&cfg), 800);
    assert_eq!(glm_max_item_tokens(&cfg), 300);
    // An operator volume replaces both upper bounds.
    cfg.max_pixels = Some(64 * 392);
    assert_eq!(glm_max_sequence_patches(&cfg), 64);
}
