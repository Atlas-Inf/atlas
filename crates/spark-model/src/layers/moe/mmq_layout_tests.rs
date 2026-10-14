// SPDX-License-Identifier: AGPL-3.0-only
//! Check ownership with the real payload helper and checkpoint teardown.
use super::*;
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;

// Reuse the byte recorder to force allocation-address reuse independently
// of optional carveout support.
use super::super::unified_test_gpu as recording;

fn payload(gpu: &dyn GpuBackend) -> (WeightStore, QuantizedWeight) {
    let alloc = || gpu.alloc(256).unwrap();
    let weight = QuantizedWeight {
        weight: alloc(),
        weight_scale: alloc(),
        ..QuantizedWeight::null()
    };
    let store = WeightStore::from_map(HashMap::from([
        (
            "w.weight".into(),
            WeightTensor {
                ptr: weight.weight,
                shape: vec![256],
                dtype: WeightDtype::UInt8,
            },
        ),
        (
            "w.weight_scale".into(),
            WeightTensor {
                ptr: weight.weight_scale,
                shape: vec![256],
                dtype: WeightDtype::FP8E4M3,
            },
        ),
    ]));
    (store, weight)
}

#[test]
fn mmq_payload_retirement_prevents_checkpoint_double_free() {
    let gpu = MockGpuBackend::new();
    let (mut store, mut weight) = payload(&gpu);
    free_quantized_payload(&gpu, &store, &mut weight).unwrap();
    store.release(&gpu).unwrap();
    assert_eq!(gpu.invalid_free_count(), 0);
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn mmq_retired_checkpoint_cannot_free_reused_slab_address() {
    let gpu = recording::RecordingGpu::new(false);
    let (mut store, mut weight) = payload(&gpu);
    let original = weight.weight;
    free_quantized_payload(&gpu, &store, &mut weight).unwrap();
    gpu.reuse_next(original);
    let slab = gpu.alloc(256).unwrap();
    assert_eq!(slab, original);
    gpu.copy_h2d(&[31; 256], slab).unwrap();
    store.release(&gpu).unwrap();
    let mut bytes = [0; 256];
    gpu.copy_d2h(slab, &mut bytes)
        .expect("replacement slab remains owned by the layer");
    assert_eq!(bytes, [31; 256]);
    gpu.free(slab).unwrap();
    assert!(gpu.live().is_empty());
}

#[test]
fn mmq_partial_payload_free_clears_each_pointer_before_the_next_attempt() {
    let gpu = MockGpuBackend::new();
    let (mut store, mut weight) = payload(&gpu);
    let original = weight.weight;
    gpu.fail_next_free();
    assert!(free_quantized_payload(&gpu, &store, &mut weight).is_err());
    assert!(
        weight.weight.is_null(),
        "a failed free belongs to backend cleanup, not another layer/store attempt"
    );
    assert!(store.was_reclaimed("w.weight"));
    assert!(!store.was_reclaimed("w.weight_scale"));
    // This mock's injected failure retains memory. Simulate backend cleanup
    // after the layer/store gave up ownership; neither may attempt it again.
    gpu.free(original).unwrap();
    free_quantized_payload(&gpu, &store, &mut weight).unwrap();
    store.release(&gpu).unwrap();
    assert_eq!(gpu.invalid_free_count(), 0);
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn mmq_repack_rejects_expert_tp_before_gpu_work() {
    let gpu = MockGpuBackend::new();
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".into();
    config.expert_tp = true;
    config.tp_world_size = 2;
    config.hidden_size = 256;
    config.moe_intermediate_size = 256;
    config.shared_expert_intermediate_size = 0;
    config.num_experts = 2;
    config.num_experts_per_tok = 1;
    let mut layer = MoeLayer::new(
        crate::weight_map::MoeWeights::empty(2),
        2,
        None,
        &gpu,
        &config,
    )
    .unwrap();
    let store = WeightStore::from_map(HashMap::new());
    let before = (gpu.alloc_count(), gpu.launch_count(), gpu.sync_count());
    let err = layer
        .repack_nvfp4_mmq_unified(&gpu, &config, &store)
        .unwrap_err();
    assert!(err.to_string().contains("expert TP"), "{err:#}");
    assert_eq!(
        before,
        (gpu.alloc_count(), gpu.launch_count(), gpu.sync_count())
    );
}

fn layer_fixture(gpu: &dyn GpuBackend) -> (MoeLayer, WeightStore, atlas_core::config::ModelConfig) {
    let config = atlas_core::config::ModelConfig {
        model_type: "glm5_next".into(),
        hidden_size: 256,
        moe_intermediate_size: 256,
        shared_expert_intermediate_size: 0,
        num_experts: 2,
        num_experts_per_tok: 1,
        ..atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4()
    };
    let mut map = HashMap::new();
    let mut weights = crate::weight_map::MoeWeights::empty(2);
    // Slot zero is not owned by this EP rank.
    let expert = &mut weights.experts[1];
    for (name, projection) in [
        ("gate", &mut expert.gate_proj),
        ("up", &mut expert.up_proj),
        ("down", &mut expert.down_proj),
    ] {
        for (suffix, ptr, bytes) in [
            ("weight", &mut projection.weight, 256 * 256 / 2),
            ("weight_scale", &mut projection.weight_scale, 256 * 256 / 16),
        ] {
            *ptr = gpu.alloc(bytes).unwrap();
            map.insert(
                format!("{name}.{suffix}"),
                WeightTensor {
                    ptr: *ptr,
                    shape: vec![bytes],
                    dtype: WeightDtype::UInt8,
                },
            );
        }
    }
    let layer = MoeLayer::new(weights, 2, None, gpu, &config).unwrap();
    (layer, WeightStore::from_map(map), config)
}

#[test]
fn mmq_full_repack_retires_all_checkpoint_sources() {
    let gpu = MockGpuBackend::new();
    let (mut layer, mut store, config) = layer_fixture(&gpu);
    layer
        .repack_nvfp4_mmq_unified(&gpu, &config, &store)
        .unwrap();
    assert!(layer.nvfp4_mmq_layout);
    assert_eq!(layer._nvfp4_mmq_owned.len(), 3);
    store.release(&gpu).unwrap();
    assert_eq!(gpu.invalid_free_count(), 0);
}

#[test]
fn mmq_later_repack_failure_keeps_earlier_retirements() {
    let gpu = MockGpuBackend::new();
    let (mut layer, mut store, config) = layer_fixture(&gpu);
    // One slab plus three table buffers completes gate, then up fails.
    gpu.fail_alloc_after(4);
    assert!(
        layer
            .repack_nvfp4_mmq_unified(&gpu, &config, &store)
            .is_err()
    );
    assert!(store.was_reclaimed("gate.weight"));
    assert!(store.was_reclaimed("gate.weight_scale"));
    assert!(!store.was_reclaimed("up.weight"));
    assert!(!store.was_reclaimed("down.weight"));
    store.release(&gpu).unwrap();
    assert_eq!(gpu.invalid_free_count(), 0);
}

#[cfg(feature = "cuda")]
#[path = "mmq_layout_cuda_tests.rs"]
mod cuda;
