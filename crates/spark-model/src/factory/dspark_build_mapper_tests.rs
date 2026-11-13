// SPDX-License-Identifier: AGPL-3.0-only

//! The DSpark build mapper's Lightning SWA-window cases, on the official
//! Lightning fixtures of `dspark_admission_tests`.

use atlas_core::config::ModelConfig;
use spark_runtime::kv_cache::KvCacheDtype;

use super::DflashBuildArgs;
use super::dspark_admission::admit_lightning_dspark_build;
use super::dspark_admission_tests::{OFFICIAL_LIGHTNING_JSON, required_store};
use crate::weight_loader::dflash_loader::parse_dflash_config;

#[test]
fn build_mapper_accepts_exact_lightning_swa_window() {
    let config = parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap();
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: config,
        gamma: Some(4),
        window_size: Some(1024),
        alt: None,
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.model_type = "nemotron_h".to_owned();
    target.hidden_size = 2688;
    target.num_hidden_layers = 52;
    target.num_experts = 128;
    target.num_experts_per_tok = 6;
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    let profile = admit_lightning_dspark_build(&args, &target, 3, 16, KvCacheDtype::Fp8)
        .unwrap()
        .expect("exact Lightning SWA window must pass");
    assert_eq!(profile.attention.swa_window, 1024);
    let missing_gamma = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap(),
        gamma: None,
        window_size: Some(1024),
        alt: None,
    };
    assert!(
        admit_lightning_dspark_build(&missing_gamma, &target, 3, 16, KvCacheDtype::Fp8).is_err()
    );
}
#[test]
fn build_mapper_rejects_missing_lightning_swa_window() {
    let config = parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap();
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: config,
        gamma: Some(4),
        window_size: None,
        alt: None,
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    let error = admit_lightning_dspark_build(&args, &target, 3, 16, KvCacheDtype::Fp8)
        .expect_err("Lightning must reject an omitted served SWA window");
    assert!(
        format!("{error:#}").contains("explicit served SWA window"),
        "{error:#}"
    );
}
#[test]
fn build_mapper_rejects_wrong_lightning_swa_window() {
    let config = parse_dflash_config(OFFICIAL_LIGHTNING_JSON).unwrap();
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: config,
        gamma: Some(4),
        window_size: Some(4096),
        alt: None,
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    let error = admit_lightning_dspark_build(&args, &target, 3, 16, KvCacheDtype::Fp8)
        .expect_err("Lightning must reject a non-contract served SWA window");
    assert!(
        format!("{error:#}").contains("served SWA window must be 1024"),
        "{error:#}"
    );
}
