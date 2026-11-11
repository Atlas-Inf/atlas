// SPDX-License-Identifier: AGPL-3.0-only

use clap::Parser;

use super::{resolve_sampling_defaults, server_default_preset};
use crate::cli;

/// A preset with values distinct from both the old hard-coded constants
/// (0.6 / 20 / 0.95) and the CLI defaults, so a wrong fallback source is
/// unmistakable in every assertion below.
fn preset() -> atlas_kernels::SamplingCategory {
    atlas_kernels::SamplingCategory {
        temperature: 0.7,
        top_p: 0.8,
        top_k: 40,
        presence_penalty: 0.0,
        frequency_penalty: 0.0,
        repetition_penalty: 1.0,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        lz_penalty: 0.0,
        min_p: None,
        top_n_sigma: None,
    }
}

fn args() -> cli::ServeArgs {
    cli::ServeArgs::parse_from(["spark", "org/model"])
}

#[test]
fn missing_config_falls_back_to_preset_not_constants() {
    let d = resolve_sampling_defaults(None, &args(), &preset());
    assert_eq!(d.temperature, 0.7, "preset, not the old 0.6 literal");
    assert_eq!(d.top_k, 40, "preset, not the old 20 literal");
    assert_eq!(d.top_p, 0.8, "preset, not the old 0.95 literal");
    assert!(d.temperature > 0.0, "absent config must never mean greedy");
}

#[test]
fn present_config_overrides_preset() {
    let cfg = serde_json::json!({"temperature": 1.1, "top_k": 64, "top_p": 0.9});
    let d = resolve_sampling_defaults(Some(&cfg), &args(), &preset());
    assert_eq!(d.temperature, 1.1);
    assert_eq!(d.top_k, 64);
    assert_eq!(d.top_p, 0.9);
}

#[test]
fn partial_config_backfills_per_field() {
    let cfg = serde_json::json!({"top_k": 64});
    let d = resolve_sampling_defaults(Some(&cfg), &args(), &preset());
    assert_eq!(d.top_k, 64, "present field honored");
    assert_eq!(d.temperature, 0.7, "absent field from preset");
    assert_eq!(d.top_p, 0.8, "absent field from preset");
}

#[test]
fn explicit_temperature_zero_is_honored() {
    // A config that is PRESENT and asks for greedy gets greedy — the
    // preset guard only backfills absent fields, it never overrides.
    let cfg = serde_json::json!({"temperature": 0.0});
    let d = resolve_sampling_defaults(Some(&cfg), &args(), &preset());
    assert_eq!(d.temperature, 0.0);
}

#[test]
fn sigma_and_min_p_fall_back_to_cli_args() {
    // Both are CLI-owned while the preset stays silent (`None`), which
    // `preset()` is.
    let a = args();
    let d = resolve_sampling_defaults(None, &a, &preset());
    assert_eq!(d.top_n_sigma, a.default_top_n_sigma);
    assert_eq!(d.min_p, a.default_min_p);
}

#[test]
fn model_declared_min_p_zero_beats_the_cli_default() {
    // The reason this field exists. A card that specifies min_p = 0 must
    // get 0, not the server's 0.08 — and 0.0 is exactly the value a
    // non-Option field could not distinguish from "unset".
    let a = args();
    assert!(a.default_min_p > 0.0, "guard: CLI default must be nonzero");
    let mut p = preset();
    p.min_p = Some(0.0);
    let d = resolve_sampling_defaults(None, &a, &p);
    assert_eq!(d.min_p, 0.0, "model preset must win over --default-min-p");
}

#[test]
fn a_model_without_filters_turns_the_cli_defaults_off() {
    // GLM-5.3's MODEL.toml: generation_config.json sets only
    // temperature/top_p, so top_k, min_p and top-n-sigma must be off —
    // not the preset's top_k 20 and the CLI's 0.08 / 1.0.
    let a = args();
    assert!(a.default_top_n_sigma > 0.0 && a.default_min_p > 0.0);
    let mut p = preset();
    p.top_k = 0;
    p.min_p = Some(0.0);
    p.top_n_sigma = Some(0.0);
    let cfg = serde_json::json!({"temperature": 1.0, "top_p": 0.95});
    let d = resolve_sampling_defaults(Some(&cfg), &a, &p);
    assert_eq!(
        (d.temperature, d.top_p, d.top_k, d.min_p, d.top_n_sigma),
        (1.0, 0.95, 0, 0.0, 0.0)
    );
}

#[test]
fn generation_config_still_outranks_a_declared_min_p() {
    // Same precedence as temperature/top_k/top_p: the checkpoint's own
    // generation_config.json is the most specific source and wins.
    let cfg = serde_json::json!({"min_p": 0.31});
    let mut p = preset();
    p.min_p = Some(0.0);
    let d = resolve_sampling_defaults(Some(&cfg), &args(), &p);
    assert_eq!(d.min_p, 0.31);
}

#[test]
fn only_glm_takes_its_sigma_into_the_server_defaults() {
    let mut p = preset();
    p.top_n_sigma = Some(0.0);
    assert_eq!(server_default_preset("glm5_next", p).top_n_sigma, Some(0.0));
    assert_eq!(server_default_preset("qwen3_next", p).top_n_sigma, None);
}
