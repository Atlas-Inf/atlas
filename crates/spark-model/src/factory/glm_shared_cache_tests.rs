// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
fn config() -> ModelConfig {
    ModelConfig {
        model_type: "glm5_next".into(),
        num_hidden_layers: 45,
        mlp_only_layers: vec![0, 1, 2],
        layer_types: (0..45)
            .map(|i| {
                if i % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::LinearAttention
                }
            })
            .collect(),
        ..ModelConfig::qwen3_next_80b_nvfp4()
    }
}
fn infos(c: &ModelConfig) -> Vec<TargetInfo> {
    (0..45)
        .map(|i| TargetInfo {
            ordinal: i,
            kind: c.layer_type(i),
            moe: i >= 3,
        })
        .collect()
}
#[test]
fn actual_deferred_plan_visits_exact42_targets_never_appended_mtp() {
    let c = config();
    let valid = infos(&c);
    assert_eq!(valid[44].ordinal, 44);
    assert!(valid[44].moe);
    assert_eq!(valid[44].kind, LayerType::LinearAttention);
    assert_eq!(
        cache_ordinals(&c, &valid).unwrap(),
        (3..45).collect::<Vec<_>>()
    );
}
#[test]
fn actual_deferred_plan_rejects_missing_reordered_wrong_kind_or_ffn_before_install() {
    let c = config();
    let good = infos(&c);
    for fault in 0..6 {
        let mut wrong = good.clone();
        match fault {
            0 => {
                wrong.pop();
            }
            1 => wrong.swap(3, 4),
            2 => wrong[10].kind = LayerType::Moe,
            3 => wrong[44].moe = false,
            4 => wrong[0].moe = true,
            _ => wrong.push(TargetInfo {
                ordinal: 45,
                kind: LayerType::FullAttention,
                moe: true,
            }),
        }
        assert!(cache_ordinals(&c, &wrong).is_err(), "fault={fault}");
    }
}

#[test]
fn disabled_factory_pass_performs_no_gpu_work_or_layer_inspection() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(std::collections::HashMap::new());
    let mut layers = Vec::new();
    initialize(&config(), &store, &gpu, &mut layers, None).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(gpu.launch_count(), 0);
    assert_eq!(gpu.d2h_blocking_count(), 0);
    assert!(
        initialize(
            &config(),
            &store,
            &gpu,
            &mut layers,
            Some(SharedFp8Reserve::new(0, 0).unwrap())
        )
        .unwrap_err()
        .to_string()
        .contains("exactly 45")
    );
    assert_eq!(gpu.alloc_count(), 0);
}
