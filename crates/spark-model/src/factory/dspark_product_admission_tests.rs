// SPDX-License-Identifier: AGPL-3.0-only

//! Factory admission tests for the official Lightning DSpark product build.

use atlas_core::config::ModelConfig;
use serde_json::Value;
use spark_runtime::kv_cache::KvCacheDtype;

use super::DflashBuildArgs;
use super::dspark_admission::admit_lightning_dspark_product_build;
use super::dspark_admission_tests::{official_value, parse_value, required_store};
use crate::layers::dflash_head::{LightningDsparkProductPolicy, LightningDsparkRuntimeToggles};

fn exact_product_toggles() -> LightningDsparkRuntimeToggles {
    LightningDsparkRuntimeToggles {
        option_b_enabled: true,
        proposal_lane_count: 1,
        proposal_graph_eligible: true,
        target_verify_graph_eligible: true,
        batched_verify_enabled: true,
        seam_serial_enabled: false,
        draft_cap_override: None,
        adaptive_enabled: false,
        batch_parity_enabled: false,
    }
}

fn product_build(
    value: &Value,
    toggles: LightningDsparkRuntimeToggles,
) -> anyhow::Result<Option<LightningDsparkProductPolicy>> {
    let store = required_store();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: parse_value(value),
        gamma: Some(4),
        window_size: Some(1024),
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.model_type = "nemotron_h".to_owned();
    target.hidden_size = 2688;
    target.num_hidden_layers = 52;
    target.num_experts = 128;
    target.num_experts_per_tok = 6;
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    admit_lightning_dspark_product_build(&args, &target, 3, 16, KvCacheDtype::Fp8, toggles)
}

#[test]
fn executable_factory_wrapper_admits_official_fake_product() {
    let policy = product_build(&official_value(), exact_product_toggles())
        .expect("factory wrapper admission")
        .expect("official fake args must produce product policy");
    assert_eq!(policy.profile().model_identity, "Qwen3DSparkModel");
}

#[test]
fn executable_factory_wrapper_returns_none_for_generic_architecture() {
    let mut value = official_value();
    value["architectures"] = serde_json::json!(["Qwen3ForCausalLM"]);
    assert!(
        product_build(&value, exact_product_toggles())
            .expect("generic factory wrapper admission")
            .is_none()
    );
}

#[test]
fn executable_factory_wrapper_rejects_non_lightning_target_identity() {
    let store = required_store();
    let value = official_value();
    let args = DflashBuildArgs {
        drafter_store: &store,
        drafter_config: parse_value(&value),
        gamma: Some(4),
        window_size: Some(1024),
    };
    let mut target = ModelConfig::qwen3_next_80b_nvfp4();
    target.tp_world_size = 1;
    target.ep_world_size = 1;
    let error = admit_lightning_dspark_product_build(
        &args,
        &target,
        3,
        16,
        KvCacheDtype::Fp8,
        exact_product_toggles(),
    )
    .expect_err("Qwen3 target must not admit the Lightning product");
    assert!(
        error.to_string().contains("target identity mismatch"),
        "{error:#}"
    );
}

#[test]
fn executable_factory_wrapper_rejects_invalid_lightning_toggles() {
    let mut toggles = exact_product_toggles();
    toggles.proposal_lane_count = 2;
    let error = product_build(&official_value(), toggles)
        .expect_err("invalid Lightning toggles must be an error, not generic admission");
    assert!(
        error.to_string().contains("proposal_lane_count"),
        "{error:#}"
    );
}
