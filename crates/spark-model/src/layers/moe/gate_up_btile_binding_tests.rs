// SPDX-License-Identifier: AGPL-3.0-only
use super::recording::{Event, Gpu};
use super::*;
use crate::layers::moe::{
    MoeLayer,
    gate_up_repack_test_gpu::{RecordingGpu, fixture},
};
use crate::weight_map::{MoeWeights, QuantizedWeight};
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

pub(super) fn setup(gpu: &Gpu, rank: usize) -> (WeightStore, ModelConfig, Vec<bool>, MoeLayer) {
    let donor = RecordingGpu::new();
    let (store, mut config, local) = fixture(rank, &donor);
    config.shared_expert_intermediate_size = 2048;
    *gpu.scalars.lock().unwrap() = donor.scalars.lock().unwrap().clone();
    let mut weights = MoeWeights::empty(288);
    for (e, expert) in weights.experts.iter_mut().enumerate() {
        for (name, q) in [
            ("gate_proj", &mut expert.gate_proj),
            ("up_proj", &mut expert.up_proj),
        ] {
            *q = QuantizedWeight::null();
            if local[e] {
                let prefix = format!("{}.mlp.experts.{e}.{name}", config.layer_prefix(0));
                q.weight = store.get(&format!("{prefix}.weight")).unwrap().ptr;
                q.weight_scale = store.get(&format!("{prefix}.weight_scale")).unwrap().ptr;
                q.weight_scale_2 = 1.25;
            }
        }
    }
    let layer = MoeLayer::new(weights, 288, None, gpu, &config).unwrap();
    (store, config, local, layer)
}

#[test]
fn actual_repacked_source_and_real_layer_tables_create_bounded_lease() {
    for rank in 0..2 {
        let gpu = Gpu::new();
        let (store, config, local, mut layer) = setup(&gpu, rank);
        let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
        let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
        let unpublished = workspace.repack(source, &family).unwrap();
        workspace.close().unwrap();
        gpu.clear();
        let _lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
        assert_eq!(
            gpu.trace()
                .iter()
                .map(|e| match e {
                    Event::Read(_, n, 77) => *n,
                    _ => panic!("non-read {e:?}"),
                })
                .sum::<usize>(),
            11520
        );
        assert_eq!(gpu.trace().len(), 6);
    }
}

#[test]
fn actual_lease_rejects_each_read_failure_and_each_payload_column() {
    use std::sync::atomic::Ordering;
    for failure in 0..12 {
        let gpu = Gpu::new();
        let (store, config, local, mut layer) = setup(&gpu, 0);
        let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
        let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
        let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
        let unpublished = workspace.repack(source, &family).unwrap();
        workspace.close().unwrap();
        let table_ptrs = [
            layer.gate_ptrs.packed_ptrs,
            layer.gate_ptrs.scale_ptrs,
            layer.gate_ptrs.scale2_vals,
            layer.up_ptrs.packed_ptrs,
            layer.up_ptrs.scale_ptrs,
            layer.up_ptrs.scale2_vals,
        ];
        gpu.clear();
        if failure < 6 {
            gpu.fail.store(failure + 1, Ordering::Relaxed);
        } else {
            // Remote expert 287 must have exact +0 scalar bits and null pointers.
            let column = failure - 6;
            let width = if column % 3 == 2 { 4 } else { 8 };
            gpu.change(table_ptrs[column], 287 * width, &[1]);
        }
        assert!(binding::Lease::bind(&unpublished, &family, &mut layer).is_err());
        assert_eq!(gpu.trace().len(), failure % 6 + 1);
    }
}

#[test]
fn actual_lease_refuses_stale_table_backend_and_capture_before_readback() {
    use std::sync::atomic::Ordering;
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    gpu.clear();
    gpu.capture.store(true, Ordering::Relaxed);
    assert!(binding::Lease::bind(&unpublished, &family, &mut layer).is_err());
    gpu.capture.store(false, Ordering::Relaxed);
    let foreign = Gpu::new();
    let foreign_family = kernels::KernelFamily::resolve(&foreign, &config, 77).unwrap();
    assert!(binding::Lease::bind(&unpublished, &foreign_family, &mut layer).is_err());
    layer.gate_ptrs.packed_ptrs.0 += 8;
    assert!(binding::Lease::bind(&unpublished, &family, &mut layer).is_err());
    assert!(gpu.trace().is_empty());
}

#[test]
fn actual_lease_refuses_hybrid_and_existing_routed_t_owners_before_readback() {
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    for fault in 0..3 {
        layer.hybrid_layout = fault == 0;
        layer.gate_ptrs_t = None;
        layer.up_ptrs_t = None;
        if fault != 0 {
            let table = crate::layers::moe::ptr_table_build::build_ptr_table_from_qw(
                &vec![crate::weight_map::QuantizedWeight::null(); 288],
                &gpu,
            )
            .unwrap();
            if fault == 1 {
                layer.gate_ptrs_t = Some(table);
            } else {
                layer.up_ptrs_t = Some(table);
            }
        }
        gpu.clear();
        assert!(binding::Lease::bind(&unpublished, &family, &mut layer).is_err());
        assert!(gpu.trace().is_empty());
    }
}
