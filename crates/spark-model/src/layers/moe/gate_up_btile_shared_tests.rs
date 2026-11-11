// SPDX-License-Identifier: AGPL-3.0-only
use super::recording::Gpu;
use super::*;
use crate::layers::moe::{MoeLayer, helpers_a::btile_shared_fixture};
use crate::weight_map::QuantizedWeight;

pub(super) fn install_native_shared(layer: &mut MoeLayer, gpu: &Gpu) {
    for q in [
        &mut layer.weights.shared_expert.gate_proj,
        &mut layer.weights.shared_expert.up_proj,
    ] {
        *q = QuantizedWeight {
            weight: gpu.alloc(PACKED_BYTES).unwrap(),
            weight_scale: gpu.alloc(native_source::SCALE_BYTES).unwrap(),
            weight_scale_2: 0.75,
            ..QuantizedWeight::null()
        };
    }
}

#[test]
fn actual_shared_only_transform_retains_allocation_authority() {
    let gpu = Gpu::new();
    let (_store, config, _local, mut layer) = binding_tests::setup(&gpu, 0);
    install_native_shared(&mut layer, &gpu);
    btile_shared_fixture(&mut layer, &gpu, &config).unwrap();
    let regions = layer
        .shared_gate_up_receipt
        .as_ref()
        .expect("actual shared transform receipt")
        .validate(
            &gpu,
            layer.shared_gate_t.unwrap(),
            layer.shared_up_t.unwrap(),
        )
        .unwrap();
    assert_eq!(
        regions.map(|r| r.1),
        [
            PACKED_BYTES,
            native_source::SCALE_BYTES,
            PACKED_BYTES,
            native_source::SCALE_BYTES
        ]
    );
}

#[test]
fn actual_stale_or_incomplete_shared_layer_refuses_binding_before_readbacks() {
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    install_native_shared(&mut layer, &gpu);
    btile_shared_fixture(&mut layer, &gpu, &config).unwrap();
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let gate = layer.shared_gate_t;
    let up = layer.shared_up_t;
    for fault in 0..4 {
        layer.shared_gate_t = gate;
        layer.shared_up_t = up;
        match fault {
            0 => layer.shared_gate_t.as_mut().unwrap().weight.0 += 16,
            1 => layer.shared_gate_t.as_mut().unwrap().weight_scale_2 = f32::NAN,
            2 => layer.shared_up_t = None,
            _ => layer.shared_gate_up_receipt = None,
        }
        gpu.clear();
        assert!(binding::Lease::bind(&unpublished, &family, &mut layer).is_err());
        assert!(gpu.trace().is_empty());
    }
}

#[test]
fn actual_shared_receipt_admits_active_decode() {
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    install_native_shared(&mut layer, &gpu);
    btile_shared_fixture(&mut layer, &gpu, &config).unwrap();
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
    let arena = spark_runtime::buffers::BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    let resources = arena_tests::ContextResources::new();
    let ctx = resources.view(&arena, &config, &gpu);
    let checked = lease.check_arena(&ctx).unwrap();
    gpu.clear();
    // Arena construction is eager; launches remain capture-compatible and do
    // not re-read tables or scan WeightStore owners during graph recording.
    use super::recording::{Arg, Event};
    use std::sync::atomic::Ordering;
    gpu.capture.store(true, Ordering::Relaxed);
    for word in [decode::WordPolicy::Word, decode::WordPolicy::Vector] {
        for count in 1..=3 {
            gpu.clear();
            checked
                .decode(
                    &arena,
                    decode::DecodeRows {
                        count,
                        input: 1,
                        routes: 0,
                        output: 1,
                    },
                    word,
                    decode::SharedMode::Active,
                )
                .unwrap();
            let trace = gpu.trace();
            assert_eq!(trace.len(), 1);
            let Event::Launch(_, _, _, _, _, args) = &trace[0] else {
                panic!("launch")
            };
            let ([gate, up], _) = lease.shared.unwrap();
            assert_eq!(
                &args[10..18],
                &[
                    Arg::Ptr(gate.weight),
                    Arg::Ptr(gate.weight_scale),
                    Arg::Bytes(gate.weight_scale_2.to_le_bytes().to_vec()),
                    Arg::Ptr(arena.ssm_deinterleaved().offset(4096)),
                    Arg::Ptr(up.weight),
                    Arg::Ptr(up.weight_scale),
                    Arg::Bytes(up.weight_scale_2.to_le_bytes().to_vec()),
                    Arg::Ptr(arena.ssm_qkvz().offset(4096))
                ]
            );
        }
    }
    gpu.capture.store(false, Ordering::Relaxed);
}

#[test]
fn actual_decode_independent_origins_and_dead_zero_sinks_preserve_shared() {
    use super::recording::{Arg, Event};
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
    let arena = spark_runtime::buffers::BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    let resources = arena_tests::ContextResources::new();
    let ctx = resources.view(&arena, &config, &gpu);
    let checked = lease.check_arena(&ctx).unwrap();
    for (input, routes, output) in [(2, 0, 0), (1, 1, 1), (0, 0, 3)] {
        gpu.clear();
        checked
            .decode(
                &arena,
                decode::DecodeRows {
                    count: 2,
                    input,
                    routes,
                    output,
                },
                decode::WordPolicy::Word,
                decode::SharedMode::RoutedOnly,
            )
            .unwrap();
        let trace = gpu.trace();
        let Event::Launch(_, _, _, _, _, args) = &trace[0] else {
            panic!("launch")
        };
        assert_eq!(args[0], Arg::Ptr(arena.norm_output().offset(input * 8192)));
        assert_eq!(args[9], Arg::Ptr(arena.scratch().offset(routes * 32)));
        assert_eq!(
            args[4],
            Arg::Ptr(arena.expert_gate_out().offset(output * 32768))
        );
        assert_eq!(
            args[8],
            Arg::Ptr(arena.expert_up_out().offset(output * 32768))
        );
        assert_eq!(args[13], Arg::Ptr(arena.expert_down_out()));
        assert_eq!(args[17], Arg::Ptr(arena.expert_down_out().offset(8192)));
    }
    for (input, routes, output) in [
        (usize::MAX, 0, 0),
        (0, usize::MAX, 0),
        (0, 0, usize::MAX),
        (4, 0, 0),
        (0, 4, 0),
        (0, 0, 4),
    ] {
        gpu.clear();
        assert!(
            checked
                .decode(
                    &arena,
                    decode::DecodeRows {
                        count: 2,
                        input,
                        routes,
                        output
                    },
                    decode::WordPolicy::Word,
                    decode::SharedMode::RoutedOnly
                )
                .is_err()
        );
        assert!(gpu.trace().is_empty());
    }
}

#[test]
fn actual_shared_receipt_rejects_stale_fields_foreign_backend_and_failed_repeat() {
    use std::sync::atomic::Ordering;
    for fail in 1..=12 {
        let gpu = Gpu::new();
        let (_store, config, _local, mut layer) = binding_tests::setup(&gpu, 0);
        install_native_shared(&mut layer, &gpu);
        btile_shared_fixture(&mut layer, &gpu, &config).unwrap();
        let gate = layer.shared_gate_t.unwrap();
        let up = layer.shared_up_t.unwrap();
        let receipt = layer.shared_gate_up_receipt.as_ref().unwrap();
        assert!(receipt.validate(&Gpu::new(), gate, up).is_err());
        for field in 0..5 {
            let mut wrong = gate;
            match field {
                0 => wrong.weight.0 += 16,
                1 => wrong.weight_scale.0 += 16,
                2 => wrong.weight_scale_2 = f32::NAN,
                3 => wrong.input_scale.0 += 4,
                _ => wrong.weight_scale_2_vec.0 += 4,
            }
            assert!(receipt.validate(&gpu, wrong, up).is_err());
        }
        gpu.clear();
        gpu.fail.store(fail, Ordering::Relaxed);
        // All twelve real lookup/alloc/transpose/sync positions, including the
        // second transform, invalidate prior authority before attempting work.
        assert!(btile_shared_fixture(&mut layer, &gpu, &config).is_err());
        assert!(layer.shared_gate_up_receipt.is_none());
    }
}

#[test]
fn legacy_unsupported_shared_transforms_succeed_without_tiled_authority() {
    for fault in 0..4 {
        let gpu = Gpu::new();
        let (_store, mut config, _local, mut layer) = binding_tests::setup(&gpu, 0);
        install_native_shared(&mut layer, &gpu);
        match fault {
            0 => config.hidden_size = 2048,
            1 => config.shared_expert_intermediate_size = 1024,
            2 => layer.shared_experts_scale_kind = crate::weight_map::WeightQuantFormat::Mxfp4E8m0,
            _ => {
                layer.weights.shared_expert.gate_proj.weight_scale_2_vec =
                    spark_runtime::gpu::DevicePtr(16)
            }
        }
        btile_shared_fixture(&mut layer, &gpu, &config).unwrap();
        assert!(layer.shared_gate_up_receipt.is_none());
    }
}
