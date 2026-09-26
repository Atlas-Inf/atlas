// SPDX-License-Identifier: AGPL-3.0-only

//! `WeightStore::insert` / `remove` / `was_reclaimed` — map edits after load.
//! Nothing here touches GPU memory except the reclaim test, which uses the
//! mock backend exactly like `teardown_tests.rs`.

use super::*;
use crate::gpu::mock::MockGpuBackend;
use atlas_core::scope::ModelResource;
use std::collections::HashMap;

fn tensor(ptr: DevicePtr) -> WeightTensor {
    WeightTensor {
        ptr,
        shape: vec![2],
        dtype: WeightDtype::BF16,
    }
}

fn store_with(names: &[&str]) -> WeightStore {
    let mut map = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        map.insert(
            (*name).to_string(),
            tensor(DevicePtr(0x1000 + 0x100 * i as u64)),
        );
    }
    WeightStore::from_map(map)
}

#[test]
fn insert_then_get_contains_len() {
    let mut store = WeightStore::from_map(HashMap::new());
    assert_eq!(store.len(), 0);
    store
        .insert("a.weight".into(), tensor(DevicePtr(0x1000)))
        .expect("insert");
    assert!(store.contains("a.weight"));
    assert_eq!(store.len(), 1);
    assert_eq!(store.get("a.weight").expect("get").ptr, DevicePtr(0x1000));
}

#[test]
fn insert_existing_name_errors_and_leaves_original() {
    let mut store = store_with(&["a.weight"]);
    let err = store
        .insert("a.weight".into(), tensor(DevicePtr(0x2000)))
        .unwrap_err();
    assert!(err.to_string().contains("already in the store"));
    assert_eq!(store.len(), 1, "store unchanged");
    assert_eq!(
        store.get("a.weight").expect("get").ptr,
        DevicePtr(0x1000),
        "the original tensor is still there"
    );
}

#[test]
fn remove_returns_tensor_and_deletes_entry() {
    let mut store = store_with(&["a.weight", "b.weight"]);
    let t = store.remove("a.weight").expect("removed");
    assert_eq!(t.ptr, DevicePtr(0x1000));
    assert!(!store.contains("a.weight"));
    assert!(store.contains("b.weight"));
    assert_eq!(store.len(), 1);
}

#[test]
fn remove_missing_name_returns_none() {
    let mut store = store_with(&["a.weight"]);
    assert!(store.remove("nope").is_none());
    assert_eq!(store.len(), 1, "untouched");
}

#[test]
fn insert_remove_do_not_disturb_deferred() {
    let mut store = WeightStore::from_map(HashMap::new());
    store.defer(
        "a.weight".into(),
        DeferredTensor {
            path: std::path::PathBuf::from("x.safetensors"),
            offset: 4096,
            shape: vec![2],
            dtype: WeightDtype::BF16,
        },
    );
    // A deferred name may also be inserted/resident (different lookup path);
    // neither insert nor remove may touch the deferred locator.
    store
        .insert("a.weight".into(), tensor(DevicePtr(0x1000)))
        .expect("insert");
    assert_eq!(store.deferred("a.weight").expect("deferred").offset, 4096);
    store.remove("a.weight").expect("removed");
    assert_eq!(store.deferred("a.weight").expect("deferred").offset, 4096);
    assert!(!store.contains("a.weight"));
}

#[test]
fn remove_of_reclaimed_tensor_reports_it_and_release_does_not_double_free() {
    let gpu = MockGpuBackend::new();
    let mut map = HashMap::new();
    map.insert(
        "a.weight".to_string(),
        tensor(gpu.alloc(1024).expect("alloc")),
    );
    map.insert(
        "b.weight".to_string(),
        tensor(gpu.alloc(1024).expect("alloc")),
    );
    let mut store = WeightStore::from_map(map);

    store.reclaim(&gpu, "a.weight").expect("reclaim");
    assert_eq!(gpu.alloc_count(), 1, "a's memory is gone, entry stays");
    assert!(store.was_reclaimed("a.weight"));
    assert!(!store.was_reclaimed("b.weight"));
    assert!(!store.was_reclaimed("absent"), "absent is never reclaimed");

    // The returned tensor has a DEAD pointer: the caller must not free it.
    let t = store.remove("a.weight").expect("removed");
    assert!(
        !store.reclaimed.lock().unwrap().contains(&t.ptr.0),
        "the reclaimed set dropped the pointer"
    );

    // Teardown frees only b: a.weight's map entry is gone, so `release` never
    // sees its (already-freed) pointer — no double free — and the mock's
    // allocation table proves b was the only buffer freed here.
    let b_ptr = store.get("b.weight").expect("b").ptr;
    store.release(&gpu).expect("release");
    assert_eq!(gpu.alloc_count(), 0);
    assert!(
        gpu.read_alloc(t.ptr).is_none(),
        "reclaim already freed a.weight's buffer; it must not be live again"
    );
    assert!(gpu.read_alloc(b_ptr).is_none(), "b was freed at release");
}
