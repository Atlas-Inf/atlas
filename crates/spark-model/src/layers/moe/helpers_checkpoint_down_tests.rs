// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::weight_loader::glm5::retirement::RetirementLog;
use atlas_core::scope::ModelResource;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;

fn checkpoint_fixture(
    gpu: &RecordingGpu,
    rank: usize,
) -> (MoeLayer, atlas_core::config::ModelConfig, WeightStore) {
    let mut config = atlas_core::config::ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = 128;
    config.moe_intermediate_size = 64;
    config.shared_expert_intermediate_size = 32;
    config.num_experts = 4;
    config.num_experts_per_tok = 2;
    config.ep_world_size = 2;
    config.ep_rank = rank;
    let mut weights = MoeWeights::empty(4);
    let mut map = HashMap::new();
    for expert in 0..4 {
        if !config.is_local_expert(expert) {
            continue;
        }
        weights.experts[expert] = ExpertWeight {
            gate_proj: make_weight(gpu, 64, 128, 16, 1),
            up_proj: make_weight(gpu, 64, 128, 16, 2),
            down_proj: make_weight(gpu, 128, 64, 16, 3),
        };
        for (name, q, n, k) in [
            ("gate_proj", weights.experts[expert].gate_proj, 64, 128),
            ("up_proj", weights.experts[expert].up_proj, 64, 128),
            ("down_proj", weights.experts[expert].down_proj, 128, 64),
        ] {
            let prefix = format!("{}.mlp.experts.{expert}.{name}", config.layer_prefix(0));
            map.insert(
                format!("{prefix}.weight"),
                WeightTensor {
                    ptr: q.weight,
                    shape: vec![n, k / 2],
                    dtype: WeightDtype::UInt8,
                },
            );
            map.insert(
                format!("{prefix}.weight_scale"),
                WeightTensor {
                    ptr: q.weight_scale,
                    shape: vec![n, k / 16],
                    dtype: WeightDtype::FP8E4M3,
                },
            );
        }
    }
    let store = WeightStore::from_map(map);
    let layer = MoeLayer::new(weights, 4, None, gpu, &config).unwrap();
    (layer, config, store)
}

#[test]
fn actual_down_transform_retirement_then_store_release_never_frees_native_twice() {
    for rank in [0, 1] {
        let gpu = RecordingGpu::new(false);
        let (mut layer, config, mut store) = checkpoint_fixture(&gpu, rank);
        let originals = store
            .names()
            .map(|name| store.get(name).unwrap().ptr.0)
            .collect::<std::collections::HashSet<_>>();
        let log = RetirementLog::new(&store, &gpu).unwrap();
        gpu.clear();
        layer
            .transpose_checkpoint_down(&gpu, &config, 0, &log)
            .unwrap();
        assert!(layer.down_ptrs_t.is_some());
        log.finish().rebuild(&mut store, &gpu).unwrap();
        assert_eq!(
            store.len(),
            8,
            "only native GU allocations remain in adopted store"
        );
        store.release(&gpu).unwrap();
        let frees = gpu
            .trace()
            .into_iter()
            .filter_map(|e| {
                if let Event::Free(p) = e {
                    originals.contains(&p.0).then_some(p.0)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        assert_eq!(frees.len(), 12);
        assert_eq!(
            frees.iter().collect::<std::collections::HashSet<_>>().len(),
            12
        );
        store.release(&gpu).unwrap();
    }
}

#[test]
fn actual_down_phase_every_checkpoint_free_fault_has_no_retry_in_cleanup() {
    let baseline = RecordingGpu::new(false);
    let (mut layer, config, store) = checkpoint_fixture(&baseline, 0);
    let original_down = store
        .names()
        .filter(|name| name.contains("down_proj"))
        .map(|name| store.get(name).unwrap().ptr)
        .collect::<Vec<_>>();
    let log = RetirementLog::new(&store, &baseline).unwrap();
    baseline.clear();
    layer
        .transpose_checkpoint_down(&baseline, &config, 0, &log)
        .unwrap();
    let positions = baseline
        .trace()
        .iter()
        .enumerate()
        .filter_map(|(i, event)| {
            if let Event::Free(ptr) = event {
                original_down.contains(ptr).then_some(i + 1)
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    assert_eq!(positions.len(), 4);
    for position in positions {
        let gpu = RecordingGpu::new(false);
        let (mut layer, config, mut store) = checkpoint_fixture(&gpu, 0);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        gpu.clear();
        gpu.fail(position);
        assert!(
            layer
                .transpose_checkpoint_down(&gpu, &config, 0, &log)
                .is_err()
        );
        let receipt = log.finish();
        assert!(receipt.rebuild(&mut store, &gpu).is_err());
        let prior = gpu.trace();
        let failed = match prior.last().unwrap() {
            Event::Free(ptr) => *ptr,
            _ => panic!("free fault"),
        };
        gpu.clear();
        assert!(
            receipt.cleanup(&mut store, &gpu).is_err(),
            "construction failure stays visible"
        );
        assert!(
            !gpu.trace()
                .iter()
                .any(|event| matches!(event,Event::Free(ptr) if *ptr==failed))
        );
        assert!(store.is_empty());
    }
}

#[test]
fn actual_recycled_allocation_is_not_confused_with_a_retired_checkpoint_key() {
    let gpu = RecordingGpu::new(false);
    let down = gpu.alloc(128).unwrap();
    let gate = gpu.alloc(128).unwrap();
    let mut store = WeightStore::from_map(HashMap::from([
        (
            "down".into(),
            WeightTensor {
                ptr: down,
                shape: vec![128],
                dtype: WeightDtype::UInt8,
            },
        ),
        (
            "gate".into(),
            WeightTensor {
                ptr: gate,
                shape: vec![128],
                dtype: WeightDtype::UInt8,
            },
        ),
    ]));
    let log = RetirementLog::new(&store, &gpu).unwrap();
    log.release_checkpoint(&store, "down", down, &gpu).unwrap();
    gpu.reuse_next(down);
    let arena = gpu.alloc(128).unwrap();
    assert_eq!(arena, down);
    log.disjoint_live(arena, 128, &gpu).unwrap();
    assert!(log.disjoint_live(gate, 128, &gpu).is_err());
    log.finish().rebuild(&mut store, &gpu).unwrap();
    store.release(&gpu).unwrap();
    assert!(
        gpu.live().contains_key(&arena.0),
        "old checkpoint metadata must not free the new generation"
    );
    gpu.free(arena).unwrap();
    assert!(gpu.live().is_empty());
}

#[test]
fn actual_down_phase_rejects_last_projection_mismatch_before_any_backend_work() {
    for scale in [false, true] {
        let gpu = RecordingGpu::new(false);
        let (mut layer, config, store) = checkpoint_fixture(&gpu, 1);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        if scale {
            layer.weights.experts[3].down_proj.weight_scale = DevicePtr::NULL;
        } else {
            layer.weights.experts[3].down_proj.weight =
                layer.weights.experts[3].down_proj.weight.offset(16);
        }
        gpu.clear();
        assert!(
            layer
                .transpose_checkpoint_down(&gpu, &config, 0, &log)
                .is_err()
        );
        assert!(gpu.trace().is_empty());
    }
}
