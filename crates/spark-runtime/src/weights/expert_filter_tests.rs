// SPDX-License-Identifier: AGPL-3.0-only
use super::SafetensorsLoader;

#[test]
fn replicated_prefix_overrides_ep_expert_filter() {
    let mut loader = SafetensorsLoader::with_ep(1, 2, 288);
    let appended = "model.language_model.layers.45.mlp.experts.1.gate_proj.weight";
    let target = "model.language_model.layers.44.mlp.experts.1.gate_proj.weight";
    assert!(loader.should_skip_tensor(appended));
    loader.replicated_expert_prefix = Some(".layers.45.".into());
    assert!(!loader.should_skip_tensor(appended));
    assert!(loader.should_skip_tensor(target));
}
