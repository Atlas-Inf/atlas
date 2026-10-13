// SPDX-License-Identifier: AGPL-3.0-only
//! Exercise the real GLM FFN loader with synthetic ModelOpt packed weights.
use super::*;
use crate::layers::FfnComponent;
use crate::weight_map::{Nvfp4Variant, QuantizeCtx};
use spark_runtime::gpu::{KernelHandle, mock::MockGpuBackend};
use spark_runtime::weights::{WeightDtype, WeightTensor};
use std::collections::HashMap;

fn fixture(gpu: &MockGpuBackend, defect: Option<(&str, &str)>) -> WeightStore {
    let mut tensors = HashMap::new();
    for (projection, n, k) in [
        ("gate_proj", 64, 32),
        ("up_proj", 64, 32),
        ("down_proj", 32, 64),
    ] {
        for (suffix, mut shape, dtype, value) in [
            ("weight", vec![n, k / 2], WeightDtype::UInt8, None),
            ("weight_scale", vec![n, k / 16], WeightDtype::FP8E4M3, None),
            ("weight_scale_2", vec![], WeightDtype::FP32, Some(0.125_f32)),
            ("input_scale", vec![], WeightDtype::FP32, Some(0.5_f32)),
        ] {
            if defect == Some((projection, suffix)) {
                if shape.is_empty() {
                    shape.push(2);
                } else {
                    shape[0] += 1;
                }
            }
            let ptr = gpu
                .alloc(shape.iter().product::<usize>() * dtype.byte_size())
                .unwrap();
            if let Some(value) = value {
                gpu.copy_h2d(&value.to_le_bytes(), ptr).unwrap();
            }
            tensors.insert(
                format!("l.mlp.{projection}.{suffix}"),
                WeightTensor { ptr, shape, dtype },
            );
        }
    }
    WeightStore::from_map(tensors)
}

fn load(store: &WeightStore, gpu: &MockGpuBackend) -> anyhow::Result<FfnComponent> {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 32;
    config.intermediate_size = 64;
    config.mlp_only_layers = vec![0];
    components::load_ffn(
        store,
        "l",
        0,
        &config,
        gpu,
        Nvfp4Variant::Standard,
        QuantizeCtx {
            absmax_k: KernelHandle(1),
            quantize_k: KernelHandle(2),
            stream: 0,
        },
        true,
    )
}

#[test]
fn nvidia_dense_ffn_preserves_checkpoint_bytes_scales_without_conversion() {
    let gpu = MockGpuBackend::new();
    let store = fixture(&gpu, None);
    let layer = load(&store, &gpu).unwrap();
    let FfnComponent::Dense(layer) = layer else {
        panic!("expected dense FFN")
    };
    for (name, actual) in [
        ("gate_proj", layer.weights.gate_proj),
        ("up_proj", layer.weights.up_proj),
        ("down_proj", layer.weights.down_proj),
    ] {
        assert_eq!(
            actual.weight,
            store.get(&format!("l.mlp.{name}.weight")).unwrap().ptr
        );
        assert_eq!(
            actual.weight_scale,
            store
                .get(&format!("l.mlp.{name}.weight_scale"))
                .unwrap()
                .ptr
        );
        assert_eq!(
            actual.input_scale,
            store.get(&format!("l.mlp.{name}.input_scale")).unwrap().ptr
        );
        assert_eq!(actual.weight_scale_2, 0.125);
    }
    assert_eq!(
        gpu.launch_count(),
        0,
        "native packed weights must not dequantize/requantize"
    );
}

#[test]
fn nvidia_dense_ffn_rejects_malformed_packed_geometry_before_launch() {
    for suffix in ["weight", "weight_scale", "weight_scale_2", "input_scale"] {
        let gpu = MockGpuBackend::new();
        let store = fixture(&gpu, Some(("gate_proj", suffix)));
        assert!(load(&store, &gpu).is_err(), "accepted malformed {suffix}");
        assert_eq!(gpu.launch_count(), 0);
    }
}

#[test]
fn nvidia_dense_ffn_rejects_nonfinite_or_nonpositive_scalars() {
    for suffix in ["weight_scale_2", "input_scale"] {
        for value in [0.0_f32, -1.0, f32::NAN, f32::INFINITY] {
            let gpu = MockGpuBackend::new();
            let store = fixture(&gpu, None);
            let ptr = store.get(&format!("l.mlp.gate_proj.{suffix}")).unwrap().ptr;
            gpu.copy_h2d(&value.to_le_bytes(), ptr).unwrap();
            assert!(load(&store, &gpu).is_err());
            assert_eq!(gpu.launch_count(), 0);
        }
    }
}

#[test]
fn nvidia_dense_ffn_keeps_existing_unquantized_fallback() {
    let gpu = MockGpuBackend::new();
    let mut tensors = HashMap::new();
    for (projection, n, k) in [
        ("gate_proj", 64, 32),
        ("up_proj", 64, 32),
        ("down_proj", 32, 64),
    ] {
        tensors.insert(
            format!("l.mlp.{projection}.weight"),
            WeightTensor {
                ptr: gpu.alloc(n * k * 2).unwrap(),
                shape: vec![n, k],
                dtype: WeightDtype::BF16,
            },
        );
    }
    let store = WeightStore::from_map(tensors);
    let layer = load(&store, &gpu).unwrap();
    let FfnComponent::Dense(layer) = layer else {
        panic!("expected dense FFN")
    };
    assert_ne!(
        layer.weights.gate_proj.weight,
        store.get("l.mlp.gate_proj.weight").unwrap().ptr
    );
    assert!(
        gpu.launch_count() > 0,
        "BF16 checkpoints still use runtime quantization"
    );
}
