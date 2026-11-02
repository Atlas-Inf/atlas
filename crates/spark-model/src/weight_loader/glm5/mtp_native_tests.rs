// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

#[test]
fn native_predictor_retires_expanded_checkpoint_sources_only() {
    use super::super::fp4l::tests::{FP4L_HEADERS, NVFP4_HEADERS, build, uploaded};
    for (headers, expected) in [(FP4L_HEADERS, 154_226_688), (NVFP4_HEADERS, 0)] {
        let gpu = MockGpuBackend::new();
        let store = build(&gpu, uploaded(headers));
        let prefix = "model.language_model.layers.45";
        let derived = gpu.alloc(16).unwrap();
        gpu.copy_h2d(&[0x5a; 16], derived).unwrap();
        let lazy_bytes = || {
            crate::layers::ops::lazy_bf16_copy_bytes(
                store
                    .names()
                    .filter(|name| !store.was_reclaimed(name))
                    .map(|name| {
                        let tensor = store.get(name).unwrap();
                        (name, tensor.dtype, tensor.shape.as_slice())
                    }),
            )
        };
        let lazy_before = lazy_bytes();
        assert_eq!(
            release_native_expanded(&store, prefix, &gpu).unwrap(),
            expected
        );
        assert_eq!(release_native_expanded(&store, prefix, &gpu).unwrap(), 0);
        assert_eq!(store.reclaimed_count(), if expected == 0 { 0 } else { 22 });
        assert_eq!(
            lazy_before - lazy_bytes(),
            if expected == 0 { 0 } else { 299_106_304 },
            "already-expanded native weights must not reserve a second BF16 copy"
        );
        for name in store.names() {
            let tensor = store.get(name).unwrap();
            let retired = name.starts_with(&format!("{prefix}."))
                && !name.contains(".experts.")
                && (name.ends_with(".weight") && tensor.dtype == WeightDtype::FP8E4M3
                    || name.ends_with(".weight_scale") && tensor.dtype == WeightDtype::UInt8);
            assert_eq!(store.was_reclaimed(name), retired, "{name}");
        }
        let mut bytes = [0; 16];
        gpu.copy_d2h(derived, &mut bytes).unwrap();
        assert_eq!(bytes, [0x5a; 16], "derived BF16 allocation remains alive");
        assert_eq!(
            gpu.launch_count(),
            0,
            "retirement cannot change weight values"
        );
    }
}

#[test]
fn native_predictor_retirement_preserves_real_dense_allocations() {
    let gpu = MockGpuBackend::new();
    let source = gpu.alloc(64).unwrap();
    let scale = gpu.alloc(2).unwrap();
    let router = gpu.alloc(64).unwrap();
    gpu.copy_h2d(&[127, 127], scale).unwrap();
    gpu.copy_h2d(&[0x73; 64], router).unwrap();
    let store = WeightStore::from_map(HashMap::from([
        (
            "l.mlp.shared_experts.gate_proj.weight".into(),
            WeightTensor {
                ptr: source,
                dtype: WeightDtype::FP8E4M3,
                shape: vec![2, 32],
            },
        ),
        (
            "l.mlp.shared_experts.gate_proj.weight_scale".into(),
            WeightTensor {
                ptr: scale,
                dtype: WeightDtype::UInt8,
                shape: vec![2, 1],
            },
        ),
        (
            "l.mlp.gate.weight".into(),
            WeightTensor {
                ptr: router,
                dtype: WeightDtype::BF16,
                shape: vec![1, 32],
            },
        ),
    ]));
    let dense = exact_dense(&store, "l.mlp.shared_experts.gate_proj", &gpu).unwrap();
    let native = exact_dense(&store, "l.mlp.gate", &gpu).unwrap();
    assert_ne!(dense.weight, source);
    assert_eq!(native.weight, router);
    // The mock records the real expansion launch; sentinel bytes verify that
    // retiring its sources never frees or overwrites the returned allocation.
    gpu.copy_h2d(&[0x5a; 128], dense.weight).unwrap();
    assert_eq!(release_native_expanded(&store, "l", &gpu).unwrap(), 66);
    let mut bytes = [0; 64];
    gpu.copy_d2h(dense.weight, &mut bytes).unwrap();
    assert_eq!(bytes, [0x5a; 64]);
    gpu.copy_d2h(native.weight, &mut bytes).unwrap();
    assert_eq!(bytes, [0x73; 64]);
    assert_eq!(gpu.launch_count(), 1, "one exact expansion only");
    assert!(gpu.copy_d2h(source, &mut [0; 32]).is_err());
    assert!(gpu.copy_d2h(scale, &mut [0; 1]).is_err());
}

#[test]
fn actual_fp4local_predictor_dense_formats_expand_exactly() {
    use super::super::fp4l::tests::{FP4L_HEADERS, build, uploaded};
    let gpu = MockGpuBackend::new();
    let st = build(&gpu, uploaded(FP4L_HEADERS));
    let prefix = "model.language_model.layers.45";
    for stem in [
        "eh_proj",
        "mlp.gate",
        "mlp.shared_experts.gate_proj",
        "mlp.shared_experts.up_proj",
        "mlp.shared_experts.down_proj",
        "self_attn.q_a_proj",
        "self_attn.q_b_proj",
        "self_attn.kv_a_proj_with_mqa",
        "self_attn.kv_b_proj",
        "self_attn.o_proj",
        "self_attn.indexer.wq_b",
        "self_attn.indexer.wk",
        "self_attn.indexer.weights_proj",
    ] {
        validate_exact_dense(&st, &format!("{prefix}.{stem}")).unwrap();
    }
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn native_mtp_moe_never_builds_requantized_shared_weights() {
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    let mut add = |name: String, dtype: WeightDtype, shape: Vec<usize>| {
        let bytes = shape.iter().product::<usize>() * dtype.byte_size();
        let ptr = gpu.alloc(bytes).unwrap();
        if shape.is_empty() {
            gpu.copy_h2d(&1f32.to_le_bytes(), ptr).unwrap();
        } else if dtype == WeightDtype::UInt8 && name.ends_with(".weight_scale") {
            // Unit E8M0 scales, within the exact BF16 expansion domain.
            gpu.copy_h2d(&vec![127u8; bytes], ptr).unwrap();
        }
        map.insert(name, WeightTensor { ptr, dtype, shape });
    };
    for (name, n, k) in [
        ("gate_proj", 64, 32),
        ("up_proj", 64, 32),
        ("down_proj", 32, 64),
    ] {
        let routed = format!("l.mlp.experts.0.{name}");
        add(
            format!("{routed}.weight"),
            WeightDtype::UInt8,
            vec![n, k / 2],
        );
        add(
            format!("{routed}.weight_scale"),
            WeightDtype::FP8E4M3,
            vec![n, k / 16],
        );
        add(
            format!("{routed}.weight_scale_2"),
            WeightDtype::FP32,
            vec![],
        );
        let shared = format!("l.mlp.shared_experts.{name}");
        add(format!("{shared}.weight"), WeightDtype::FP8E4M3, vec![n, k]);
        add(
            format!("{shared}.weight_scale"),
            WeightDtype::UInt8,
            vec![n, k / 32],
        );
    }
    add("l.mlp.gate.weight".into(), WeightDtype::BF16, vec![1, 32]);
    add(
        "l.mlp.gate.e_score_correction_bias".into(),
        WeightDtype::FP32,
        vec![1],
    );
    let st = WeightStore::from_map(map);
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 32;
    c.moe_intermediate_size = 64;
    c.shared_expert_intermediate_size = 64;
    c.num_experts = 1;
    c.num_experts_per_tok = 1;
    let FfnComponent::Moe(layer) = load_native_moe(&st, "l", &c, &gpu).unwrap() else {
        panic!("expected MoE")
    };
    assert!(layer.weights.shared_expert.gate_proj.is_null());
    assert!(layer.weights.shared_expert.up_proj.is_null());
    assert!(layer.weights.shared_expert.down_proj.is_null());
    assert_eq!(
        layer.weights.experts[0].gate_proj.weight,
        st.get("l.mlp.experts.0.gate_proj.weight").unwrap().ptr
    );
    assert_eq!(
        gpu.launch_count(),
        3,
        "only three exact MXFP8 expansions, no requantization"
    );
}
