// SPDX-License-Identifier: AGPL-3.0-only
use super::super::gate_up_repack_test_gpu::{Arg, Event, RecordingGpu, fixture, fixture_layer};
use super::native_source::SCALE_BYTES;
use super::*;
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use std::sync::atomic::Ordering;

fn family(
    gpu: &dyn GpuBackend,
    packed: KernelHandle,
    transpose: KernelHandle,
) -> kernels::KernelFamily<'_> {
    let donor = RecordingGpu::new();
    let (_, mut config, _) = fixture(0, &donor);
    config.shared_expert_intermediate_size = 2048;
    let mut family = kernels::KernelFamily::resolve(gpu, &config, 77).unwrap();
    // Explicitly corrupt handles only in the existing negative-handle test.
    family.handles[0] = packed;
    family.handles[1] = transpose;
    family
}

#[test]
fn actual_production_byte_dispatch_full_layer_exact_abi_and_single_workspace() {
    for rank in 0..2 {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(rank, &gpu);
        let input = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        let expected: Vec<_> = input
            .projections()
            .iter()
            .map(|p| (p.packed.ptr, p.scales.ptr))
            .collect();
        gpu.clear();
        let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
        let scratch = workspace.scratch.unwrap().ptr;
        let result = workspace
            .repack(input, &family(&gpu, KernelHandle(101), KernelHandle(102)))
            .unwrap();
        assert_eq!(result.source.projections().len(), 288);
        let events = gpu.trace();
        assert_eq!(events[0], Event::Alloc(PACKED_BYTES));
        assert_eq!(events.len(), 1 + 288 * 6);
        for (i, &(packed, scales)) in expected.iter().enumerate() {
            assert_eq!(
                &events[1 + i * 6..1 + (i + 1) * 6],
                &[
                    Event::Copy(packed, scratch, PACKED_BYTES, 77),
                    Event::Launch(
                        101,
                        [16384, 1, 1],
                        [256, 1, 1],
                        0,
                        77,
                        vec![
                            Arg::Ptr(scratch),
                            Arg::Ptr(packed),
                            Arg::Bytes(2048u32.to_le_bytes().to_vec()),
                            Arg::Bytes(4096u32.to_le_bytes().to_vec())
                        ]
                    ),
                    Event::Sync(77),
                    Event::Copy(scales, scratch, SCALE_BYTES, 77),
                    Event::Launch(
                        102,
                        [8, 64, 1],
                        [32, 8, 1],
                        0,
                        77,
                        vec![
                            Arg::Ptr(scratch),
                            Arg::Ptr(scales),
                            Arg::Bytes(2048u32.to_le_bytes().to_vec()),
                            Arg::Bytes(256u32.to_le_bytes().to_vec())
                        ]
                    ),
                    Event::Sync(77),
                ]
            );
        }
        workspace.close().unwrap();
        assert!(gpu.live.lock().unwrap().is_empty());
        assert_eq!(gpu.trace().last(), Some(&Event::Free(scratch)));
    }
}

#[test]
fn actual_transaction_every_copy_launch_sync_fault_poisoned_and_cleaned() {
    // Every operation position, including partial earlier-expert conversion.
    for fault in 1..=288 * 6 {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(0, &gpu);
        let input = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
        let scratch = workspace.scratch.unwrap().ptr;
        gpu.clear();
        gpu.fail_at.store(fault, Ordering::Relaxed);
        let error = workspace
            .repack(input, &family(&gpu, KernelHandle(101), KernelHandle(102)))
            .err()
            .expect("injected failure");
        assert!(error.to_string().contains("abandon"), "{error:#}");
        assert!(workspace.poisoned);
        assert!(workspace.scratch.is_none());
        assert!(gpu.live.lock().unwrap().is_empty());
        let events = gpu.trace();
        assert_eq!(events.len(), fault + 2);
        assert_eq!(&events[fault..], &[Event::Sync(77), Event::Free(scratch)]);
        // Refusing a new checked input proves uncertain async failures cannot
        // be hidden by reusing the workspace for another layer.
        gpu.clear();
        let next = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        gpu.clear();
        assert!(
            workspace
                .repack(next, &family(&gpu, KernelHandle(101), KernelHandle(102)))
                .is_err()
        );
        assert!(gpu.trace().is_empty());
    }
}

#[test]
fn actual_transaction_refuses_handles_stream_and_capture_before_copy() {
    for fault in 0..4 {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(0, &gpu);
        let input = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        let mut workspace = RepackWorkspace::new(&gpu, if fault == 2 { 78 } else { 77 }).unwrap();
        gpu.clear();
        let handles = family(
            &gpu,
            KernelHandle(if fault == 0 { 0 } else { 101 }),
            KernelHandle(if fault == 1 { 0 } else { 102 }),
        );
        if fault == 3 {
            gpu.capturing.store(true, Ordering::Relaxed);
        }
        assert!(workspace.repack(input, &handles).is_err());
        assert!(gpu.trace().is_empty());
        gpu.capturing.store(false, Ordering::Relaxed);
        workspace.close().unwrap();
    }
    let gpu = RecordingGpu::new();
    gpu.capturing.store(true, Ordering::Relaxed);
    assert!(RepackWorkspace::new(&gpu, 77).is_err());
    assert!(gpu.trace().is_empty());
    gpu.capturing.store(false, Ordering::Relaxed);
    gpu.fail_at.store(1, Ordering::Relaxed);
    assert!(RepackWorkspace::new(&gpu, 77).is_err());
    assert!(gpu.live.lock().unwrap().is_empty());
}

#[test]
fn actual_checked_packed_op_rejects_span_alias_overflow_and_zero_handle() {
    use crate::layers::ops::moe_gate_up_repack::glm_native_to_btile;
    let gpu = RecordingGpu::new();
    for (source, dest, handle) in [
        (0, 0x1000000, 101),
        (0x1001, 0x1000000, 101),
        (0x1000, 0x2000, 101),
        (u64::MAX - 15, 0x1000000, 101),
        (0x1000, 0x1000000, 0),
    ] {
        assert!(
            glm_native_to_btile(
                &gpu,
                KernelHandle(handle),
                DevicePtr(source),
                DevicePtr(dest),
                77
            )
            .is_err()
        );
        assert!(gpu.trace().is_empty());
    }
}

#[test]
fn actual_workspace_close_free_error_reports_lost_ledger_and_pointer() {
    let gpu = RecordingGpu::new();
    let workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let scratch = workspace.scratch.unwrap().ptr;
    gpu.clear();
    gpu.fail_at.store(2, Ordering::Relaxed);
    let error = workspace.close().unwrap_err();
    assert!(error.to_string().contains(&format!("{scratch:?}")));
    assert!(error.to_string().contains("context/process teardown"));
    assert_eq!(gpu.trace(), vec![Event::Sync(77), Event::Free(scratch)]);
    assert!(
        gpu.live.lock().unwrap().is_empty(),
        "matches real remove-before-free ledger semantics"
    );
}

#[test]
fn actual_workspace_backend_alias_fault_never_frees_original_owner() {
    for scalar in [false, true] {
        let gpu = RecordingGpu::new();
        let (store, config, local) = fixture(0, &gpu);
        let input = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        let p = &input.projections()[0];
        let alias = if scalar { p.scalar.ptr } else { p.packed.ptr };
        gpu.allocation.store(alias.0, Ordering::Relaxed);
        let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
        gpu.clear();
        assert!(
            workspace
                .repack(input, &family(&gpu, KernelHandle(101), KernelHandle(102)))
                .is_err()
        );
        assert!(workspace.poisoned);
        assert!(workspace.scratch.is_none());
        drop(workspace);
        assert!(gpu.trace().is_empty(), "must not free overlapping original");
        // A broken allocator returned an already-owned pointer: refuse to
        // 'clean up' an original tensor; leave diagnosis to model teardown.
        assert!(gpu.live.lock().unwrap().contains(&alias.0));
    }
}

#[test]
fn actual_transaction_refuses_foreign_backend_without_launch() {
    let gpu = RecordingGpu::new();
    let other = RecordingGpu::new();
    let (store, config, local) = fixture(0, &gpu);
    let input = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&other, 77).unwrap();
    other.clear();
    gpu.clear();
    assert!(
        workspace
            .repack(input, &family(&gpu, KernelHandle(101), KernelHandle(102)))
            .is_err()
    );
    assert!(other.trace().is_empty());
    assert!(gpu.trace().is_empty());
    workspace.close().unwrap();
}

#[test]
fn actual_workspace_reused_across_distinct_layers_without_device_allocations() {
    let gpu = RecordingGpu::new();
    let (store, cfg, local) = fixture(0, &gpu);
    let (second, _, _) = fixture_layer(0, 1, &gpu);
    let first = NativeGateUpLayer::from_store(&store, &cfg, 0, &local, &gpu, 77).unwrap();
    let next = NativeGateUpLayer::from_store(&second, &cfg, 1, &local, &gpu, 77).unwrap();
    gpu.clear();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let scratch = workspace.scratch.unwrap().ptr;
    let a = workspace
        .repack(first, &family(&gpu, KernelHandle(101), KernelHandle(102)))
        .unwrap();
    let b = workspace
        .repack(next, &family(&gpu, KernelHandle(101), KernelHandle(102)))
        .unwrap();
    assert_ne!(
        a.source.projections()[0].packed.ptr,
        b.source.projections()[0].packed.ptr
    );
    assert_eq!(workspace.scratch.unwrap().ptr, scratch);
    workspace.close().unwrap();
    assert_eq!(
        gpu.trace()
            .iter()
            .filter(|e| matches!(e, Event::Alloc(_)))
            .count(),
        1
    );
    assert_eq!(
        gpu.trace()
            .iter()
            .filter(|e| matches!(e, Event::Free(_)))
            .count(),
        1
    );
    assert!(gpu.live.lock().unwrap().is_empty());
}

#[test]
fn actual_async_failure_preserves_primary_and_cleanup_error_context() {
    for cleanup in [2, 3] {
        let gpu = RecordingGpu::new();
        let (store, cfg, local) = fixture(0, &gpu);
        let input = NativeGateUpLayer::from_store(&store, &cfg, 0, &local, &gpu, 77).unwrap();
        let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
        gpu.clear();
        gpu.fail_at.store(1, Ordering::Relaxed);
        gpu.fail_also.store(cleanup, Ordering::Relaxed);
        let error = workspace
            .repack(input, &family(&gpu, KernelHandle(101), KernelHandle(102)))
            .err()
            .unwrap();
        let message = format!("{error:#}");
        assert!(
            message.contains("at 1") && message.contains(&format!("at {cleanup}")),
            "{message}"
        );
        assert!(workspace.poisoned && workspace.scratch.is_none());
        assert_eq!(gpu.trace().len(), 3);
        assert!(
            gpu.live.lock().unwrap().is_empty(),
            "real backend removes ledger entry before free attempt"
        );
        if cleanup == 3 {
            assert!(message.contains("context/process teardown"));
        }
    }
}

#[test]
fn actual_workspace_foreign_shared_down_mtp_alias_never_mutates_or_frees_owner() {
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
    for name in [
        "model.layers.0.mlp.shared_experts.gate_proj.weight",
        "model.layers.0.mlp.experts.0.down_proj.weight",
        "model.layers.42.mtp.weight",
    ] {
        let gpu = RecordingGpu::new();
        let (store, cfg, local) = fixture(0, &gpu);
        let mut tensors: std::collections::HashMap<_, _> = store
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
            .collect();
        let foreign = DevicePtr(0x7000_0000_0000);
        tensors.insert(
            name.into(),
            WeightTensor {
                ptr: foreign,
                shape: vec![4096],
                dtype: WeightDtype::BF16,
            },
        );
        let store = WeightStore::from_map(tensors);
        let input = NativeGateUpLayer::from_store(&store, &cfg, 0, &local, &gpu, 77).unwrap();
        gpu.allocation.store(foreign.0, Ordering::Relaxed);
        let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
        gpu.clear();
        assert!(
            workspace
                .repack(input, &family(&gpu, KernelHandle(101), KernelHandle(102)))
                .is_err(),
            "{name}"
        );
        assert!(workspace.poisoned && workspace.scratch.is_none());
        drop(workspace);
        assert!(
            gpu.trace().is_empty(),
            "{name}: original was used as scratch or freed"
        );
    }
}

#[test]
fn actual_weight_store_release_owns_all_originals_exactly_once_after_unpublished_drop() {
    use atlas_core::scope::ModelResource;
    let gpu = RecordingGpu::new();
    let (mut store, config, local) = fixture(0, &gpu);
    let originals: std::collections::HashSet<_> = store
        .names()
        .map(|name| store.get(name).unwrap().ptr.0)
        .collect();
    gpu.live.lock().unwrap().extend(&originals);
    let input = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace
        .repack(input, &family(&gpu, KernelHandle(101), KernelHandle(102)))
        .unwrap();
    workspace.close().unwrap();
    assert_eq!(*gpu.live.lock().unwrap(), originals);
    // Borrowed descriptors must be dropped before the actual mutable store
    // release. This is teardown, never permission to export converted bytes.
    drop(unpublished);
    gpu.clear();
    store.release(&gpu as &dyn GpuBackend).unwrap();
    assert!(store.is_empty());
    assert!(gpu.live.lock().unwrap().is_empty());
    let events = gpu.trace();
    assert_eq!(events.len(), originals.len());
    assert!(
        events
            .iter()
            .all(|event| matches!(event, Event::Free(p) if originals.contains(&p.0)))
    );
    store.release(&gpu as &dyn GpuBackend).unwrap();
    assert_eq!(gpu.trace().len(), events.len(), "idempotent release");
}
