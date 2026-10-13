// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn routed_moe_models_take_the_prefill_transpose() {
    assert!(runs_prefill_transpose(&ModelConfig::qwen3_next_80b_nvfp4()));
}

#[test]
fn glm5_next_keeps_its_n_major_routed_experts() {
    let config = ModelConfig {
        model_type: "glm5_next".into(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
    assert!(config.num_experts > 0);
    assert!(!runs_prefill_transpose(&config));
}

#[test]
fn dense_models_skip_the_prefill_transpose() {
    let config = ModelConfig {
        num_experts: 0,
        ..ModelConfig::qwen3_next_80b_nvfp4()
    };
    assert!(!runs_prefill_transpose(&config));
}
