// SPDX-License-Identifier: AGPL-3.0-only

use super::{quant_multiplier, skip_mtp};
use atlas_core::config::ModelConfig;
use clap::Parser as _;

/// qwen4_exp's `mtp.*` (5.21 GB) is uploaded only where it drafts: rank 0
/// under `--speculative`.
#[test]
fn qwen4_exp_mtp_is_uploaded_on_the_drafting_rank_only() {
    let args = |more: &[&str]| {
        let argv = ["spark", "some/model"].iter().chain(more);
        crate::cli::ServeArgs::parse_from(argv)
    };
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen4_exp".to_string();
    assert!(skip_mtp(&config, &args(&[])));
    assert!(!skip_mtp(&config, &args(&["--speculative"])));
    config.ep_rank = 1;
    assert!(skip_mtp(&config, &args(&["--speculative"])));
}

/// The dense hybrid uploads `mtp.*` unless DFlash replaces the MTP head.
#[test]
fn qwen3_5_skips_mtp_under_dflash_only() {
    let args = |more: &[&str]| {
        let argv = ["spark", "some/model"].iter().chain(more);
        crate::cli::ServeArgs::parse_from(argv)
    };
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen3_5".to_string();
    assert!(!skip_mtp(&config, &args(&[])));
    assert!(!skip_mtp(&config, &args(&["--speculative"])));
    assert!(skip_mtp(&config, &args(&["--dflash"])));
}

#[test]
fn qwen4_exp_uses_measured_load_peak() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen4_exp".to_string();
    assert_eq!(quant_multiplier(&config), Some(1.2));
}

#[test]
fn non_fp8_qwen35_keeps_generic_multiplier() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "qwen3_5".to_string();
    assert_eq!(quant_multiplier(&config), None);
}
