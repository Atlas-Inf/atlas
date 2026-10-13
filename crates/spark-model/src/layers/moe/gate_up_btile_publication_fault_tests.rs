// SPDX-License-Identifier: AGPL-3.0-only
//! Actual publication/preflight and injected production operations; no forged Ready.
use super::super::{
    load::BTileLoadSession,
    recording::{Event, Gpu},
};
use super::setup;
use crate::layers::moe::MoeLayer;
use crate::weight_loader::glm5::retirement::RetirementLog;
use crate::weight_map::QuantizedWeight;
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use std::sync::atomic::Ordering;

fn mutation(event: &Event) -> bool {
    matches!(
        event,
        Event::Alloc(..)
            | Event::H2d(..)
            | Event::Copy(..)
            | Event::Launch(..)
            | Event::Free(..)
            | Event::Memset(..)
    )
}
fn views(layer: &MoeLayer) -> Vec<[u64; 5]> {
    layer
        .weights
        .experts
        .iter()
        .chain(std::iter::once(&layer.weights.shared_expert))
        .flat_map(|e| [e.gate_proj, e.up_proj, e.down_proj])
        .map(|q| {
            [
                q.weight.0,
                q.weight_scale.0,
                u64::from(q.weight_scale_2.to_bits()),
                q.input_scale.0,
                q.weight_scale_2_vec.0,
            ]
        })
        .collect()
}
fn scratch(gpu: &Gpu) -> DevicePtr {
    let allocations: Vec<_> = gpu
        .trace()
        .into_iter()
        .filter_map(|event| match event {
            Event::Alloc(ptr, n) => Some((ptr, n)),
            _ => None,
        })
        .collect();
    assert_eq!(
        allocations.len(),
        1,
        "session owns exactly one reusable allocation"
    );
    assert_eq!(allocations[0].1, 4_194_304);
    allocations[0].0
}
fn free_count(trace: &[Event], ptr: DevicePtr) -> usize {
    trace
        .iter()
        .filter(|event| matches!(event,Event::Free(p) if *p==ptr))
        .count()
}

#[test]
fn predictable_profile_and_actual_source_faults_refuse_before_first_mutation() {
    for rank in [0, 1] {
        for case in 0..23 {
            let gpu = Gpu::new();
            let (store, mut config, mut layer) = setup(&gpu, rank);
            let local = (0..288).find(|&e| config.is_local_expert(e)).unwrap();
            let remote = (0..288).find(|&e| !config.is_local_expert(e)).unwrap();
            let log = RetirementLog::new(&store, &gpu).unwrap();
            gpu.clear();
            let mut session = BTileLoadSession::new(&gpu, &config, 77).unwrap();
            let scratch = scratch(&gpu);
            let mut ordinal = 0;
            match case {
                0 => config.hidden_size = 4097,
                1 => config.scoring_func = "softmax".into(),
                2 => config.adapter_max_rank = 1,
                3 => ordinal = config.num_hidden_layers,
                4 => layer.unified_layout = true,
                5 => layer.hybrid_layout = true,
                6 => layer.nvfp4_mmq_layout = true,
                7 => layer.weights.shared_expert.gate_proj = QuantizedWeight::null(),
                8 => {
                    layer.weights.shared_expert.gate_proj.weight =
                        layer.weights.shared_expert.gate_proj.weight.offset(16)
                }
                9 => {
                    layer.weights.shared_expert.up_proj.weight_scale =
                        layer.weights.shared_expert.gate_proj.weight_scale
                }
                10 => layer.weights.shared_expert.down_proj.weight_scale_2 = f32::NAN,
                11 => layer.weights.shared_expert.gate_proj.weight_scale_2 = 3.0,
                12 => {
                    layer.weights.shared_expert.up_proj.weight_scale_2_vec =
                        layer.weights.shared_expert.up_proj.weight_scale
                }
                13 => {
                    layer.weights.experts[local].down_proj.weight_scale_2_vec =
                        layer.weights.experts[local].down_proj.weight_scale
                }
                14 => {
                    layer.weights.experts[local].gate_proj.weight =
                        layer.weights.experts[local].up_proj.weight
                }
                15 => {
                    layer.weights.experts[remote].gate_proj = layer.weights.experts[local].gate_proj
                }
                16 => layer.weights.experts[local].up_proj = QuantizedWeight::null(),
                17 => layer.up_ptrs.packed_ptrs = layer.gate_ptrs.packed_ptrs,
                18 => gpu.change(
                    layer.gate_ptrs.packed_ptrs,
                    remote * 8,
                    &16u64.to_le_bytes(),
                ),
                19 => gpu.change(layer.up_ptrs.scale2_vals, local * 4, &2f32.to_le_bytes()),
                20 => layer.moe_transpose_u8_batched_k = KernelHandle(0),
                21 => {
                    layer.weights.shared_expert.down_proj.weight =
                        layer.weights.shared_expert.up_proj.weight
                }
                22 => {
                    layer.weights.experts[local].down_proj.weight =
                        layer.weights.experts[local].gate_proj.weight
                }
                _ => unreachable!(),
            }
            let original = views(&layer);
            let allocations = gpu.allocation_count();
            gpu.clear();
            let error = session.prepare(&mut layer, &log, &config, ordinal).err();
            assert!(
                error.is_some(),
                "rank={rank} case={case} published invalid source"
            );
            assert!(
                gpu.trace().iter().all(|event| !mutation(event)),
                "rank={rank} case={case}: {:?}",
                gpu.trace()
            );
            assert!(
                layer.btile_storage.require_legacy().is_ok(),
                "preflight spent construction: rank={rank} case={case}"
            );
            assert_eq!(views(&layer), original, "preflight changed native views");
            assert_eq!(gpu.allocation_count(), allocations);
            gpu.clear();
            session.close().unwrap();
            assert_eq!(gpu.trace(), vec![Event::Sync(77), Event::Free(scratch)]);
            assert_eq!(gpu.allocation_count(), allocations - 1);
        }
    }
}

#[test]
fn shared_transpose_lookup_error_or_zero_refuses_before_mutation() {
    for rank in [0, 1] {
        for zero in [false, true] {
            let gpu = Gpu::new();
            let (store, config, mut layer) = setup(&gpu, rank);
            let log = RetirementLog::new(&store, &gpu).unwrap();
            let mut session = BTileLoadSession::new(&gpu, &config, 77).unwrap();
            gpu.clear();
            gpu.lookup_failure.store(1, Ordering::Relaxed);
            gpu.lookup_zero.store(zero, Ordering::Relaxed);
            assert!(session.prepare(&mut layer, &log, &config, 0).is_err());
            assert!(
                gpu.trace().iter().all(|event| !mutation(event)),
                "zero={zero}: {:?}",
                gpu.trace()
            );
            assert!(layer.btile_storage.require_legacy().is_ok());
            assert!(!layer.btile_storage.is_published());
            session.close().unwrap();
        }
    }
}

#[test]
fn foreign_retirement_backend_refuses_before_first_mutation() {
    let gpu = Gpu::new();
    let foreign = Gpu::new();
    let (store, config, mut layer) = setup(&gpu, 0);
    let log = RetirementLog::new(&store, &foreign).unwrap();
    let mut session = BTileLoadSession::new(&gpu, &config, 77).unwrap();
    gpu.clear();
    assert!(session.prepare(&mut layer, &log, &config, 0).is_err());
    assert!(gpu.trace().iter().all(|event| !mutation(event)));
    assert!(foreign.trace().is_empty());
    assert!(layer.btile_storage.require_legacy().is_ok());
    session.close().unwrap();
}

fn completed_trace(rank: usize) -> Vec<Event> {
    let gpu = Gpu::new();
    let (store, config, mut layer) = setup(&gpu, rank);
    let log = RetirementLog::new(&store, &gpu).unwrap();
    let mut session = BTileLoadSession::new(&gpu, &config, 77).unwrap();
    gpu.clear();
    session.prepare(&mut layer, &log, &config, 0).unwrap();
    assert!(
        layer.btile_storage.is_published(),
        "obtain a real owner before deriving faults"
    );
    let trace = gpu.trace();
    gpu.clear();
    assert!(session.prepare(&mut layer, &log, &config, 0).is_err());
    assert!(
        gpu.trace().is_empty(),
        "published owner cannot be republished"
    );
    session.close().unwrap();
    trace
}

#[test]
fn every_prepare_mutation_and_sync_fault_spends_publication_and_cleans_scratch_once() {
    for rank in [0, 1] {
        let baseline = completed_trace(rank);
        let first = baseline.iter().position(mutation).unwrap();
        let points: Vec<_> = baseline
            .iter()
            .enumerate()
            .filter_map(|(i, e)| {
                (i >= first && (mutation(e) || matches!(e, Event::Sync(_)))).then_some(i)
            })
            .collect();
        assert!(
            points.len() > 1700,
            "must include every routed byte-permutation operation"
        );
        for index in points {
            let gpu = Gpu::new();
            let (store, config, mut layer) = setup(&gpu, rank);
            let log = RetirementLog::new(&store, &gpu).unwrap();
            gpu.clear();
            let mut session = BTileLoadSession::new(&gpu, &config, 77).unwrap();
            let scratch = scratch(&gpu);
            gpu.clear();
            gpu.fail.store(index + 1, Ordering::Relaxed);
            let error = session.prepare(&mut layer, &log, &config, 0).unwrap_err();
            let first_trace = gpu.trace();
            assert!(
                format!("{error:#}").contains(&format!("injected op {}", index + 1)),
                "{error:#}"
            );
            assert_eq!(
                &first_trace[..index + 1],
                &baseline[..index + 1],
                "rank={rank} fault={index}"
            );
            assert!(!layer.btile_storage.is_published());
            assert!(
                layer.btile_storage.require_legacy().is_err(),
                "must not fall back to native after construction starts"
            );
            gpu.clear();
            assert!(session.prepare(&mut layer, &log, &config, 0).is_err());
            assert!(gpu.trace().is_empty(), "failed publication retried I/O");
            session.close().unwrap();
            let mut all = first_trace;
            all.extend(gpu.trace());
            assert_eq!(
                free_count(&all, scratch),
                1,
                "scratch cleanup rank={rank} fault={index}"
            );
            let mut freed = std::collections::HashSet::new();
            for event in &all {
                if let Event::Free(ptr) = event {
                    assert!(freed.insert(ptr.0), "duplicate free {ptr:?}");
                }
            }
            // Existing shared/down helpers can leave partial derived allocations
            // for backend teardown. Do not pretend this tests a global sweep.
        }
    }
}

#[test]
fn close_sync_and_free_faults_keep_successful_publication_but_never_retry_free() {
    for failure in [1, 2] {
        let gpu = Gpu::new();
        let (store, config, mut layer) = setup(&gpu, 0);
        let log = RetirementLog::new(&store, &gpu).unwrap();
        gpu.clear();
        let mut session = BTileLoadSession::new(&gpu, &config, 77).unwrap();
        let scratch = scratch(&gpu);
        session.prepare(&mut layer, &log, &config, 0).unwrap();
        assert!(layer.btile_storage.is_published());
        gpu.clear();
        gpu.fail.store(failure, Ordering::Relaxed);
        assert!(session.close().is_err());
        assert_eq!(gpu.trace(), vec![Event::Sync(77), Event::Free(scratch)]);
        assert!(layer.btile_storage.is_published());
        // A failed free has already removed the recorder/real CUDA ledger entry.
        // It is not retried by Drop; caller must abandon construction on error.
    }
}
