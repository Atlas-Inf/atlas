// SPDX-License-Identifier: AGPL-3.0-only

//! Unit tests for `processor_pixels.rs` (moved from `serve_tests.rs` with
//! the reader).

use super::*;

fn image_max(dir: &std::path::Path) -> Option<usize> {
    read_declared_pixels(dir, Modality::Image).map(|d| d.max_pixels)
}

fn dir_with(files: &[(&str, &str)]) -> tempfile::TempDir {
    let d = tempfile::tempdir().expect("tempdir");
    for (name, body) in files {
        std::fs::write(d.path().join(name), body).expect("write");
    }
    d
}

/// The HF `save_pretrained` shape: image fields at the top level of
/// `preprocessor_config.json`. Qwen/Qwen3.6-35B-A3B-FP8 ships this.
#[test]
fn reads_the_flat_preprocessor_config() {
    let d = dir_with(&[(
        "preprocessor_config.json",
        r#"{"size": {"longest_edge": 16777216, "shortest_edge": 65536}}"#,
    )]);
    assert_eq!(image_max(d.path()), Some(16_777_216));
}

/// The combined-processor shape: each modality under its own key in
/// `processor_config.json`. unsloth/Qwen3.6-27B-NVFP4 ships this, and
/// reading only the other filename is why it ran the 1280 fallback
/// while declaring 4096² of permitted area.
#[test]
fn reads_the_nested_processor_config() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
    )]);
    assert_eq!(image_max(d.path()), Some(16_777_216));
}

/// THE TRAP. `processor_config.json` carries a second, LARGER bound
/// under `video_processor`, so any implementation that scans for the
/// first (or largest) `longest_edge` in the document admits still
/// images at half again their permitted area. Uses the real numbers
/// from the shipped checkpoint.
#[test]
fn the_video_bound_never_wins_over_the_image_bound() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{
            "video_processor": {"size": {"longest_edge": 25165824}, "fps": 2},
            "image_processor": {"size": {"longest_edge": 16777216}}
        }"#,
    )]);
    assert_eq!(image_max(d.path()), Some(16_777_216));
}

/// A video-only processor config declares nothing about stills, so the
/// image path must fall back rather than borrow the video bound.
#[test]
fn a_video_only_config_yields_no_image_bound() {
    let d = dir_with(&[(
        "processor_config.json",
        r#"{"video_processor": {"size": {"longest_edge": 25165824}}}"#,
    )]);
    assert_eq!(image_max(d.path()), None);
}

/// Older processors write the count directly instead of under `size`.
#[test]
fn accepts_the_direct_max_pixels_spelling() {
    let d = dir_with(&[("preprocessor_config.json", r#"{"max_pixels": 1048576}"#)]);
    assert_eq!(image_max(d.path()), Some(1_048_576));
}

/// Precedence when a checkpoint ships both: the dedicated image file
/// wins, so a stale combined config cannot override it.
#[test]
fn the_dedicated_file_outranks_the_combined_one() {
    let d = dir_with(&[
        ("preprocessor_config.json", r#"{"max_pixels": 1048576}"#),
        (
            "processor_config.json",
            r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
        ),
    ]);
    assert_eq!(image_max(d.path()), Some(1_048_576));
}

/// A checkpoint we cannot read must still SERVE, at the historical
/// behaviour — never fail to boot over a preprocessing hint.
#[test]
fn unreadable_or_absent_config_falls_back_rather_than_failing() {
    let empty = tempfile::tempdir().expect("tempdir");
    assert_eq!(image_max(empty.path()), None);

    let broken = dir_with(&[("preprocessor_config.json", "{not json")]);
    assert_eq!(image_max(broken.path()), None);

    let zero = dir_with(&[("preprocessor_config.json", r#"{"max_pixels": 0}"#)]);
    assert_eq!(image_max(zero.path()), None);
}

/// A malformed FIRST source must not shadow a good later one — the
/// loop continues rather than committing to the file it opened.
#[test]
fn a_broken_first_source_does_not_mask_a_good_second() {
    let d = dir_with(&[
        ("preprocessor_config.json", r#"{"size": {}}"#),
        (
            "processor_config.json",
            r#"{"image_processor": {"size": {"longest_edge": 16777216}}}"#,
        ),
    ]);
    assert_eq!(image_max(d.path()), Some(16_777_216));
}

/// HF's Qwen image processors scale a still below `shortest_edge` UP to
/// it, so the min is read with the max, from the same source.
#[test]
fn reads_the_image_min_beside_the_max() {
    let d = dir_with(&[(
        "preprocessor_config.json",
        r#"{"size": {"longest_edge": 16777216, "shortest_edge": 65536}}"#,
    )]);
    assert_eq!(
        read_declared_pixels(d.path(), Modality::Image),
        Some(DeclaredPixels {
            min_pixels: Some(65_536),
            max_pixels: 16_777_216
        })
    );
    let direct = dir_with(&[(
        "preprocessor_config.json",
        r#"{"min_pixels": 3136, "max_pixels": 1048576}"#,
    )]);
    assert_eq!(
        read_declared_pixels(direct.path(), Modality::Image),
        Some(DeclaredPixels {
            min_pixels: Some(3136),
            max_pixels: 1_048_576
        })
    );
}

/// The Qwen3.8 checkpoint's `video_preprocessor_config.json`, and the
/// combined shape: the video bounds, never the image ones.
#[test]
fn reads_the_video_bounds() {
    let want = Some(DeclaredPixels {
        min_pixels: Some(4096),
        max_pixels: 25_165_824,
    });
    let d = dir_with(&[
        (
            "preprocessor_config.json",
            r#"{"size": {"longest_edge": 16777216, "shortest_edge": 65536}}"#,
        ),
        (
            "video_preprocessor_config.json",
            r#"{"size": {"longest_edge": 25165824, "shortest_edge": 4096}}"#,
        ),
    ]);
    assert_eq!(read_declared_pixels(d.path(), Modality::Video), want);
    let nested = dir_with(&[(
        "processor_config.json",
        r#"{
            "image_processor": {"size": {"longest_edge": 16777216, "shortest_edge": 65536}},
            "video_processor": {"size": {"longest_edge": 25165824, "shortest_edge": 4096}}
        }"#,
    )]);
    assert_eq!(read_declared_pixels(nested.path(), Modality::Video), want);
    let image_only = dir_with(&[(
        "preprocessor_config.json",
        r#"{"size": {"longest_edge": 16777216}}"#,
    )]);
    assert_eq!(
        read_declared_pixels(image_only.path(), Modality::Video),
        None
    );
}

/// GLM-5.3 declares token budgets, not pixel areas: the nested image and
/// video blocks of the shipped `processor_config.json` are read as given
/// (the serving video cap is applied where the budget is used).
#[test]
fn reads_the_glm_token_budgets_per_modality() {
    let dir_with = |files: &[(&str, &str)]| {
        let dir = tempfile::tempdir().unwrap();
        for (name, body) in files {
            std::fs::write(dir.path().join(name), body).unwrap();
        }
        dir
    };
    let d = dir_with(&[(
        "processor_config.json",
        r#"{
            "image_processor": {"min_image_tokens": 16, "max_image_tokens": 8000},
            "video_processor": {"min_image_tokens": 16, "max_image_tokens": 240000, "fps": 2}
        }"#,
    )]);
    let budgets = read_glm_token_budgets(d.path());
    assert_eq!(budgets.image, (16, 8000));
    assert_eq!(budgets.video, (16, 240_000));
    assert!(
        crate::main_modules::processor_pixels::read_declared_pixels(
            d.path(),
            crate::main_modules::processor_pixels::Modality::Image
        )
        .is_none()
    );

    let custom = dir_with(&[(
        "processor_config.json",
        r#"{"image_processor": {"max_image_tokens": 1024}}"#,
    )]);
    let budgets = read_glm_token_budgets(custom.path());
    assert_eq!(budgets.image, (16, 1024), "a missing key keeps the default");
    assert_eq!(budgets.video, (16, 240_000));
    let empty = tempfile::tempdir().expect("tempdir");
    assert_eq!(read_glm_token_budgets(empty.path()), Default::default());
}
