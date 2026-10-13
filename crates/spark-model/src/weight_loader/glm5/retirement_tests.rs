// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use atlas_core::scope::ModelResource;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightTensor};
#[path = "retirement_test_gpu.rs"]
pub(super) mod recording;

fn store(gpu: &dyn GpuBackend) -> WeightStore {
    WeightStore::from_map(
        ["gate", "down", "scale"]
            .into_iter()
            .map(|name| {
                (
                    name.to_owned(),
                    WeightTensor {
                        ptr: gpu.alloc(16).unwrap(),
                        shape: vec![4, 4],
                        dtype: WeightDtype::UInt8,
                    },
                )
            })
            .collect(),
    )
}

#[test]
fn actual_checkpoint_free_rebuild_and_store_release_keep_gate_live() {
    let gpu = MockGpuBackend::new();
    let mut store = store(&gpu);
    let log = RetirementLog::new(&store, &gpu).unwrap();
    for name in ["down", "scale"] {
        let ptr = store.get(name).unwrap().ptr;
        log.release_checkpoint(&store, name, ptr, &gpu).unwrap();
    }
    assert_eq!(gpu.alloc_count(), 1);
    assert!(store.contains("down"), "metadata remains until owned seam");
    log.finish().rebuild(&mut store, &gpu).unwrap();
    assert_eq!(store.names().collect::<Vec<_>>(), ["gate"]);
    assert_eq!(gpu.alloc_count(), 1, "rebuild must not free or allocate");
    store.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
    store.release(&gpu).unwrap();
}

#[test]
fn actual_checkpoint_duplicate_free_refuses_before_backend() {
    let gpu = MockGpuBackend::new();
    let store = store(&gpu);
    let log = RetirementLog::new(&store, &gpu).unwrap();
    let ptr = store.get("down").unwrap().ptr;
    log.release_checkpoint(&store, "down", ptr, &gpu).unwrap();
    assert!(log.release_checkpoint(&store, "down", ptr, &gpu).is_err());
    assert_eq!(gpu.alloc_count(), 2);
}

#[test]
fn actual_derived_free_and_cleanup_use_existing_store_release() {
    let gpu = MockGpuBackend::new();
    let mut store = store(&gpu);
    let log = RetirementLog::new(&store, &gpu).unwrap();
    log.release_derived(&store, gpu.alloc(8).unwrap(), 8, &gpu)
        .unwrap();
    log.finish().cleanup(&mut store, &gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn every_checkpoint_free_fault_is_removed_once_before_cleanup() {
    for fail_name in ["down", "scale"] {
        let gpu = recording::Gpu::default();
        let mut store = store(&gpu);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        *gpu.fail.lock() = Some(store.get(fail_name).unwrap().ptr);
        for name in ["down", "scale"] {
            let result = log.release_checkpoint(&store, name, store.get(name).unwrap().ptr, &gpu);
            if name == fail_name {
                assert!(format!("{:#}", result.unwrap_err()).contains("ownership unknown"));
                break;
            }
            result.unwrap();
        }
        let receipt = log.finish();
        assert!(receipt.rebuild(&mut store, &gpu).is_err());
        assert!(
            receipt.cleanup(&mut store, &gpu).is_err(),
            "preserve construction failure"
        );
        let frees = gpu.frees.lock();
        assert_eq!(frees.len(), 3);
        assert_eq!(
            frees
                .iter()
                .map(|p| p.0)
                .collect::<std::collections::HashSet<_>>()
                .len(),
            3
        );
        assert!(store.is_empty());
        assert_eq!(gpu.inner.alloc_count(), 0);
    }
}

#[test]
fn actual_foreign_backend_changed_identity_and_live_alias_refuse_free() {
    let gpu = recording::Gpu::default();
    let store = store(&gpu);
    let log = RetirementLog::new(&store, &gpu).unwrap();
    let foreign = recording::Gpu::default();
    let down = store.get("down").unwrap().ptr;
    assert!(
        log.release_checkpoint(&store, "down", down, &foreign)
            .is_err()
    );
    assert!(
        log.release_checkpoint(&store, "down", down.offset(1), &gpu)
            .is_err()
    );
    assert!(log.release_derived(&store, down, 16, &gpu).is_err());
    let origin = log.capture(&store, "down", &gpu).unwrap();
    let changed = WeightStore::from_map(HashMap::from([(
        "down".into(),
        WeightTensor {
            ptr: down,
            shape: vec![2, 8],
            dtype: WeightDtype::UInt8,
        },
    )]));
    assert!(log.release_origin(&changed, origin, &gpu).is_err());
    assert!(gpu.frees.lock().is_empty());
    assert!(foreign.frees.lock().is_empty());
}

#[test]
fn actual_immutable_index_rejects_overlaps_and_finish_revalidates_all_identities() {
    let gpu = recording::Gpu::default();
    let mut store = store(&gpu);
    let down = store.get("down").unwrap().ptr;
    let alias = WeightStore::from_map(HashMap::from([
        (
            "first".into(),
            WeightTensor {
                ptr: down,
                shape: vec![16],
                dtype: WeightDtype::UInt8,
            },
        ),
        (
            "second".into(),
            WeightTensor {
                ptr: down.offset(8),
                shape: vec![16],
                dtype: WeightDtype::UInt8,
            },
        ),
    ]));
    assert!(RetirementLog::new(&alias, &gpu).is_err());
    let log = RetirementLog::new(&store, &gpu).unwrap();
    let receipt = log.finish();
    let replacement = store
        .names()
        .map(|name| {
            let tensor = store.get(name).unwrap();
            (
                name.into(),
                WeightTensor {
                    ptr: tensor.ptr,
                    shape: if name == "gate" {
                        vec![2, 8]
                    } else {
                        tensor.shape.clone()
                    },
                    dtype: tensor.dtype,
                },
            )
        })
        .collect();
    store = WeightStore::from_map(replacement);
    assert!(receipt.rebuild(&mut store, &gpu).is_err());
    assert!(gpu.frees.lock().is_empty());
    assert_eq!(store.len(), 3);
}

#[test]
fn cleanup_keeps_both_failed_attempt_and_later_cleanup_error_visible() {
    let gpu = recording::Gpu::default();
    let mut store = store(&gpu);
    let log = RetirementLog::new(&store, &gpu).unwrap();
    let down = store.get("down").unwrap().ptr;
    *gpu.fail.lock() = Some(down);
    assert!(log.release_checkpoint(&store, "down", down, &gpu).is_err());
    *gpu.fail.lock() = Some(store.get("gate").unwrap().ptr);
    let error = log.finish().cleanup(&mut store, &gpu).unwrap_err();
    let message = format!("{error:#}");
    assert!(
        message.contains("ownership unknown")
            && message.contains("remaining checkpoint cleanup: Err")
    );
    assert_eq!(gpu.frees.lock().iter().filter(|&&p| p == down).count(), 1);
    assert!(store.is_empty());
}
