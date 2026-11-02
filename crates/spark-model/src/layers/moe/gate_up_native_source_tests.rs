// SPDX-License-Identifier: AGPL-3.0-only
use super::super::super::gate_up_repack_test_gpu::{Event, RecordingGpu, fixture};
use super::*;
use spark_runtime::weights::{WeightDtype as D, WeightTensor};
use std::collections::HashMap;
use std::sync::atomic::Ordering;

fn remap(store: &WeightStore) -> HashMap<String, WeightTensor> {
    store
        .names()
        .map(|name| {
            let t = store.get(name).unwrap();
            (
                name.to_owned(),
                WeightTensor {
                    ptr: t.ptr,
                    shape: t.shape.clone(),
                    dtype: t.dtype,
                },
            )
        })
        .collect()
}

#[test]
fn actual_live_source_allows_retired_owner_reuse_but_rejects_live_checkpoint_alias() {
    let gpu = RecordingGpu::new();
    let (store, config, local) = fixture(0, &gpu);
    let retired = gpu.alloc(PACKED_BYTES).unwrap();
    let mut map = remap(&store);
    map.insert(
        "retired_down".into(),
        WeightTensor {
            ptr: retired,
            shape: vec![PACKED_BYTES],
            dtype: D::UInt8,
        },
    );
    let store = WeightStore::from_map(map);
    let log = RetirementLog::new(&store, &gpu).unwrap();
    log.release_checkpoint(&store, "retired_down", retired, &gpu)
        .unwrap();
    gpu.allocation.store(retired.0, Ordering::Relaxed);
    let reused = gpu.alloc(PACKED_BYTES).unwrap();
    assert_eq!(reused, retired);
    let source = NativeGateUpLayer::from_live(&log, &config, 0, &local, &gpu, 77).unwrap();
    assert!(source.scratch_is_disjoint(Span::new(reused, PACKED_BYTES, 16).unwrap()));
    assert!(!source.scratch_is_disjoint(source.projections()[0].packed));
    assert!(
        !NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77)
            .unwrap()
            .scratch_is_disjoint(Span::new(reused, PACKED_BYTES, 16).unwrap())
    );
    drop(source);
    let _receipt = log.finish();
    gpu.free(reused).unwrap();
}

#[test]
fn actual_native_store_provenance_full_both_ranks_and_scalar_bits() {
    for rank in 0..2 {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(rank, &gpu);
        let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        assert_eq!(source.projections.len(), 288);
        for (i, p) in source.projections.iter().enumerate() {
            assert_eq!(p.expert, rank * 144 + i / 2);
            assert_eq!(p.is_up, i % 2 != 0);
            assert_eq!(p.packed.bytes, PACKED_BYTES);
            assert_eq!(p.scales.bytes, SCALE_BYTES);
            assert_eq!(p.scalar_bits, 1.25f32.to_bits());
            assert_eq!(p.input_bits, Some((-0.0f32).to_bits()));
        }
        assert_eq!(gpu.trace().len(), 576);
        assert!(
            gpu.trace()
                .iter()
                .all(|e| matches!(e, Event::Scalar(_, 77)))
        );
    }
}

#[test]
fn actual_provenance_rejects_all_metadata_and_alias_faults_before_scalar_reads() {
    for fault in 0..20 {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(0, &gpu);
        let mut map = remap(&store);
        let p = format!("{}.mlp.experts.143.up_proj", config.layer_prefix(0));
        let first = format!("{}.mlp.experts.0.gate_proj", config.layer_prefix(0));
        let key = format!("{p}.weight");
        match fault {
            0 => map.get_mut(&key).unwrap().dtype = D::BF16,
            1 => map.get_mut(&key).unwrap().dtype = D::FP8E4M3,
            2 => map.get_mut(&key).unwrap().shape = vec![2048, 2047],
            3 => map.get_mut(&key).unwrap().shape = vec![usize::MAX, 2048],
            4 => map.get_mut(&key).unwrap().ptr = DevicePtr::NULL,
            5 => map.get_mut(&key).unwrap().ptr.0 += 1,
            6 => map.get_mut(&key).unwrap().ptr = DevicePtr(u64::MAX - 15),
            7 => map.get_mut(&format!("{p}.weight_scale")).unwrap().dtype = D::FP8E8M0,
            8 => map.get_mut(&format!("{p}.weight_scale")).unwrap().shape = vec![2048, 128],
            9 => map.get_mut(&format!("{p}.weight_scale_2")).unwrap().shape = vec![2048],
            10 => {
                map.remove(&key);
            }
            11 => {
                map.remove(&format!("{p}.weight_scale_2"));
            }
            12 => {
                let ptr = map[&format!("{first}.weight")].ptr;
                map.get_mut(&key).unwrap().ptr = ptr;
            }
            13 => {
                let ptr = map[&format!("{first}.weight")].ptr;
                map.get_mut(&format!("{p}.weight_scale")).unwrap().ptr = ptr;
            }
            14 => {
                let ptr = map[&format!("{first}.weight")].ptr;
                map.get_mut(&format!("{p}.weight_scale_2")).unwrap().ptr = ptr;
            }
            15 => map.get_mut(&format!("{p}.input_scale")).unwrap().dtype = D::UInt8,
            16 => {
                map.insert(
                    format!("{p}.weight_packed"),
                    WeightTensor {
                        ptr: DevicePtr(0x9000),
                        shape: vec![1],
                        dtype: D::UInt8,
                    },
                );
            }
            17 => {
                map.insert(
                    format!("{p}.weight_global_scale"),
                    WeightTensor {
                        ptr: DevicePtr(0x9000),
                        shape: vec![1],
                        dtype: D::FP32,
                    },
                );
            }
            18 => {
                map.insert(
                    format!("{p}.scale"),
                    WeightTensor {
                        ptr: DevicePtr(0x9000),
                        shape: vec![1],
                        dtype: D::FP8E8M0,
                    },
                );
            }
            19 => {
                let ptr = map[&key].ptr;
                map.insert(
                    "model.shared.weight".into(),
                    WeightTensor {
                        ptr,
                        shape: vec![2048, 2048],
                        dtype: D::UInt8,
                    },
                );
            }
            _ => unreachable!(),
        }
        assert!(
            NativeGateUpLayer::from_store(
                &WeightStore::from_map(map),
                &config,
                0,
                &local,
                &gpu,
                77
            )
            .is_err(),
            "fault {fault}"
        );
        assert!(gpu.trace().is_empty(), "fault {fault} performed work");
    }
}

#[test]
fn actual_provenance_profile_locality_capture_and_optional_input() {
    for fault in 0..12 {
        let gpu = RecordingGpu::new();
        let (store, mut config, mut local) = fixture(0, &gpu);
        let mut layer = 0;
        match fault {
            0 => config.model_type = "qwen3_next".into(),
            1 => config.hidden_size = 8192,
            2 => config.moe_intermediate_size = 4096,
            3 => config.num_experts = 256,
            4 => config.num_experts_per_tok = 4,
            5 => config.tp_world_size = 1,
            6 => config.ep_world_size = 1,
            7 => config.ep_rank = 2,
            8 => config.tp_rank = 1,
            9 => local.swap(0, 144),
            10 => layer = config.num_hidden_layers,
            11 => gpu.capturing.store(true, Ordering::Relaxed),
            _ => unreachable!(),
        }
        assert!(NativeGateUpLayer::from_store(&store, &config, layer, &local, &gpu, 77).is_err());
        assert!(gpu.trace().is_empty());
    }
    let gpu = RecordingGpu::new();
    let (store, config, local) = fixture(1, &gpu);
    let mut map = remap(&store);
    map.retain(|name, _| !name.ends_with(".input_scale"));
    let store = WeightStore::from_map(map);
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    assert!(
        source
            .projections
            .iter()
            .all(|p| p.input.is_none() && p.input_bits.is_none())
    );
    assert_eq!(gpu.trace().len(), 288);
}

#[test]
fn actual_provenance_scalar_read_faults_and_nonfinite_values_never_mutate() {
    for fault in 1..=576 {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(0, &gpu);
        gpu.fail_at.store(fault, Ordering::Relaxed);
        assert!(NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).is_err());
        assert_eq!(gpu.trace().len(), fault);
        assert!(
            gpu.trace()
                .iter()
                .all(|e| matches!(e, Event::Scalar(_, 77)))
        );
    }
    for bits in [f32::INFINITY.to_bits(), f32::NAN.to_bits()] {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(0, &gpu);
        gpu.scalars
            .lock()
            .unwrap()
            .values_mut()
            .for_each(|b| *b = bits);
        assert!(NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).is_err());
        assert_eq!(gpu.trace().len(), 1);
    }
}
