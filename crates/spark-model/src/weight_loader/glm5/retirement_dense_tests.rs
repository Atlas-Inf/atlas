// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::tp_shard::{TpShardKind, shard_dense_bf16};
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn actual_dense_conversion_and_tp_shard_preserve_checkpoint_origin() {
    for dtype in [WeightDtype::BF16, WeightDtype::FP32] {
        for rank in [0, 1] {
            let gpu = MockGpuBackend::new();
            let mut store = WeightStore::from_map(HashMap::from([(
                "q.weight".into(),
                WeightTensor {
                    ptr: gpu.alloc(16 * dtype.byte_size()).unwrap(),
                    shape: vec![4, 4],
                    dtype,
                },
            )]));
            let log = RetirementLog::new(&store, &gpu).unwrap();
            let source = LoadedDense::load(&store, "q.weight", &gpu, Some(&log), false).unwrap();
            let (local, _, _) = shard_dense_bf16(
                source.dense.weight,
                4,
                4,
                TpShardKind::ColumnParallel,
                rank,
                2,
                &gpu,
            )
            .unwrap();
            let local = source.replaced(local, 16, &gpu).unwrap();
            local.release(&gpu).unwrap();
            log.finish().rebuild(&mut store, &gpu).unwrap();
            assert_eq!(store.contains("q.weight"), dtype == WeightDtype::FP32);
            assert_eq!(gpu.alloc_count(), usize::from(dtype == WeightDtype::FP32));
        }
    }
}

#[test]
fn actual_fp8_packed_and_keep_f32_conversion_sources_keep_exact_provenance() {
    for (dtype, keep_f32) in [
        (WeightDtype::FP8E4M3, false),
        (WeightDtype::UInt8, false),
        (WeightDtype::BF16, true),
        (WeightDtype::FP32, true),
    ] {
        let gpu = MockGpuBackend::new();
        let mut map = HashMap::new();
        map.insert(
            "q.weight".into(),
            WeightTensor {
                ptr: gpu.alloc(16 * dtype.byte_size()).unwrap(),
                shape: vec![2, 8],
                dtype,
            },
        );
        let scale_dtype = if dtype == WeightDtype::UInt8 {
            WeightDtype::FP8E4M3
        } else {
            WeightDtype::FP32
        };
        map.insert(
            "q.weight_scale".into(),
            WeightTensor {
                ptr: gpu.alloc(scale_dtype.byte_size()).unwrap(),
                shape: vec![1],
                dtype: scale_dtype,
            },
        );
        map.insert(
            "q.weight_scale_2".into(),
            WeightTensor {
                ptr: gpu.alloc(4).unwrap(),
                shape: vec![1],
                dtype: WeightDtype::FP32,
            },
        );
        let mut store = WeightStore::from_map(map);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        LoadedDense::load(&store, "q.weight", &gpu, Some(&log), keep_f32)
            .unwrap()
            .release(&gpu)
            .unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        assert_eq!(
            store.contains("q.weight"),
            !(keep_f32 && dtype == WeightDtype::FP32)
        );
        assert!(store.contains("q.weight_scale"));
        assert!(store.contains("q.weight_scale_2"));
    }
}

#[test]
fn production_dense_conversion_and_sharding_keep_one_checkpoint_owner() {
    use atlas_core::scope::ModelResource;
    for (dtype, keep_f32) in [
        (WeightDtype::BF16, false),
        (WeightDtype::FP32, false),
        (WeightDtype::BF16, true),
        (WeightDtype::FP32, true),
    ] {
        let gpu = MockGpuBackend::new();
        let original = gpu.alloc(16 * dtype.byte_size()).unwrap();
        let mut store = WeightStore::from_map(HashMap::from([(
            "q.weight".into(),
            WeightTensor {
                ptr: original,
                shape: vec![4, 4],
                dtype,
            },
        )]));
        let loaded = LoadedDense::load(&store, "q.weight", &gpu, None, keep_f32).unwrap();
        let native = if keep_f32 {
            dtype == WeightDtype::FP32
        } else {
            dtype == WeightDtype::BF16
        };
        let derived = gpu.alloc(16).unwrap();
        loaded
            .replaced(derived, 16, &gpu)
            .unwrap()
            .release(&gpu)
            .unwrap();
        assert_eq!(store.was_reclaimed("q.weight"), native);
        store.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.invalid_free_count(), 0);
    }
}

#[test]
fn production_checkpoint_free_failure_remains_backend_owned_not_retried_by_store() {
    use atlas_core::scope::ModelResource;
    let gpu = MockGpuBackend::new();
    let original = gpu.alloc(32).unwrap();
    let mut store = WeightStore::from_map(HashMap::from([(
        "q.weight".into(),
        WeightTensor {
            ptr: original,
            shape: vec![4, 4],
            dtype: WeightDtype::BF16,
        },
    )]));
    let loaded = LoadedDense::load(&store, "q.weight", &gpu, None, false).unwrap();
    gpu.fail_next_free();
    assert!(
        loaded
            .release(&gpu)
            .unwrap_err()
            .to_string()
            .contains("mock free failed")
    );
    assert!(store.was_reclaimed("q.weight"));
    store.release(&gpu).unwrap();
    assert_eq!(
        gpu.alloc_count(),
        1,
        "failed free remains the backend's responsibility"
    );
    assert_eq!(gpu.invalid_free_count(), 0);
    gpu.free(original).unwrap(); // Stand in for the backend's final cleanup sweep.
}
