// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn unused_glm_mtp_policy_uses_actual_prefix_and_target_layer_count() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    assert!(unused_glm_mtp_prefix(&config, false).is_none());
    config.model_type = "glm5_next".into();
    config.weight_prefix = "model.language_model".into();
    config.num_hidden_layers = 45;
    assert_eq!(
        unused_glm_mtp_prefix(&config, false).as_deref(),
        Some("model.language_model.layers.45.")
    );
    assert!(unused_glm_mtp_prefix(&config, true).is_none());
    config.weight_prefix = "fixture".into();
    config.num_hidden_layers = 3;
    assert_eq!(
        unused_glm_mtp_prefix(&config, false).as_deref(),
        Some("fixture.layers.3.")
    );
}
