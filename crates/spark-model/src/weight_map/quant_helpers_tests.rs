// SPDX-License-Identifier: AGPL-3.0-only

//! Ownership tests for [`free_loader_source`]: a store alias is reclaimed (so
//! teardown does not free it twice), a loader allocation is freed directly.

use std::collections::HashMap;

use atlas_core::scope::ModelResource;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::*;

fn store_with(gpu: &MockGpuBackend, name: &str, dtype: WeightDtype) -> WeightStore {
    let mut map = HashMap::new();
    map.insert(
        name.to_string(),
        WeightTensor {
            ptr: gpu.alloc(1024).unwrap(),
            shape: vec![16, 32],
            dtype,
        },
    );
    WeightStore::from_map(map)
}

/// BF16 checkpoint: `dense_auto` handed back the store's pointer. Releasing it
/// must go through reclaim, and teardown must then not free it again.
#[test]
fn store_alias_is_reclaimed_not_freed_twice() {
    let gpu = MockGpuBackend::new();
    let name = "model.layers.0.linear_attn.in_proj_qkv.weight";
    let mut store = store_with(&gpu, name, WeightDtype::BF16);
    let alias = DenseWeight {
        weight: store.get(name).unwrap().ptr,
    };
    free_loader_source(&store, &gpu, name, alias).unwrap();
    assert_eq!(
        store.reclaimed_count(),
        1,
        "the alias was recorded as reclaimed"
    );
    assert_eq!(gpu.alloc_count(), 0, "and its memory is gone");
    // Tripwire: the next free fails. The store's only tensor was reclaimed, so
    // teardown must not call free at all; a second free of that pointer (the
    // bug: a plain `gpu.free` left the store owning it) would trip it.
    gpu.fail_next_free();
    store
        .release(&gpu)
        .expect("teardown must not free the reclaimed alias again");
}

/// FP8 checkpoint: the loader's BF16 is a fresh dequant output, not the store
/// tensor (which stays FP8 under the same name). Free the copy, leave the store.
#[test]
fn loader_allocation_is_freed_and_store_untouched() {
    let gpu = MockGpuBackend::new();
    let name = "model.layers.0.linear_attn.in_proj_z.weight";
    let mut store = store_with(&gpu, name, WeightDtype::FP8E4M3);
    let copy = DenseWeight {
        weight: gpu.alloc(2048).unwrap(),
    };
    assert_eq!(gpu.alloc_count(), 2);
    free_loader_source(&store, &gpu, name, copy).unwrap();
    assert_eq!(
        store.reclaimed_count(),
        0,
        "the FP8 source was not reclaimed"
    );
    assert_eq!(gpu.alloc_count(), 1, "only the loader copy was freed");
    store.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), 0, "teardown freed the FP8 source once");
}

/// Name absent from the store (a synthesized buffer): plain free.
#[test]
fn unknown_name_frees_directly() {
    let gpu = MockGpuBackend::new();
    let store = WeightStore::from_map(HashMap::new());
    let buf = DenseWeight {
        weight: gpu.alloc(512).unwrap(),
    };
    free_loader_source(&store, &gpu, "absent.weight", buf).unwrap();
    assert_eq!(gpu.alloc_count(), 0);
    assert_eq!(store.reclaimed_count(), 0);
}
