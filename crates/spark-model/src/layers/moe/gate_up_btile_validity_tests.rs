// SPDX-License-Identifier: AGPL-3.0-only
//! Real publication lifetimes and stale owner rejection before reader work.
use super::recording::{Event, Gpu};
use super::*;
use spark_runtime::{buffers::BufferArena, gpu::DevicePtr};

#[test]
fn actual_bind_rejects_empty_foreign_replaced_and_overlapping_retained_store() {
    use spark_runtime::weights::{WeightDtype as D, WeightStore, WeightTensor};
    use std::collections::HashMap;
    let copy = |store: &WeightStore| -> HashMap<String, WeightTensor> {
        store
            .names()
            .map(|name| {
                let w = store.get(name).unwrap();
                (
                    name.to_owned(),
                    WeightTensor {
                        ptr: w.ptr,
                        shape: w.shape.clone(),
                        dtype: w.dtype,
                    },
                )
            })
            .collect()
    };
    for rank in [0, 1] {
        for case in 0..11 {
            let gpu = Gpu::new();
            let (mut store, config, mut layer) = resident_tests::setup(&gpu, rank);
            let expert = (0..288).rev().find(|&e| config.is_local_expert(e)).unwrap();
            let prefix = format!("{}.mlp.experts.{expert}.up_proj", config.layer_prefix(0));
            let key = format!("{prefix}.weight");
            let native = store.get(&key).unwrap().ptr;
            // Optional scalar metadata is captured through the actual source,
            // not synthesized in a Ready fixture.
            let mut original = copy(&store);
            let input = gpu.alloc(4).unwrap();
            gpu.copy_h2d(&1.0f32.to_le_bytes(), input).unwrap();
            original.insert(
                format!("{prefix}.input_scale"),
                WeightTensor {
                    ptr: input,
                    shape: vec![1],
                    dtype: D::FP32,
                },
            );
            store = WeightStore::from_map(original);
            let log =
                crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
            let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
            session.prepare(&mut layer, &log, &config, 0).unwrap();
            session.close().unwrap();
            log.finish().rebuild(&mut store, &gpu).unwrap();
            let mut replaced = copy(&store);
            match case {
                0 => replaced.clear(),
                1 => {
                    let (foreign, _, _) = resident_tests::setup(&gpu, rank);
                    replaced = copy(&foreign);
                }
                2 => replaced.get_mut(&key).unwrap().ptr = gpu.alloc(4_194_304).unwrap(),
                3 => replaced.get_mut(&key).unwrap().dtype = D::FP8E4M3,
                4 => replaced.get_mut(&key).unwrap().shape = vec![1024, 4096],
                5 => {
                    replaced.remove(&format!("{prefix}.weight_scale_2"));
                }
                6 => {
                    replaced
                        .get_mut(&format!("{prefix}.weight_scale_2"))
                        .unwrap()
                        .shape = vec![1, 1]
                }
                7 => {
                    replaced
                        .get_mut(&format!("{prefix}.weight_scale"))
                        .unwrap()
                        .ptr = gpu.alloc(524_288).unwrap()
                }
                8 => {
                    replaced.remove(&key);
                    gpu.next_allocation(native);
                }
                9 => {
                    gpu.next_allocation(native);
                }
                10 => {
                    replaced.remove(&format!("{prefix}.input_scale"));
                }
                _ => unreachable!(),
            }
            let replaced = WeightStore::from_map(replaced);
            let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
            gpu.clear();
            let error = layer
                .bind_btile_arena(&replaced, &config, &gpu, &arena, 77)
                .err();
            assert!(
                error.is_some(),
                "rank={rank} case={case} accepted changed retained store"
            );
            if matches!(case, 8 | 9) {
                assert!(
                    format!("{:#}", error.unwrap()).contains("sealed GU"),
                    "must check sealed spans independently"
                );
            }
            assert!(gpu.trace().is_empty(), "bind rejection performed GPU work");
            // A rejected bind consumes no binding authority. Supply the actual
            // rebuilt owner and a disjoint arena, then exercise a real reader.
            gpu.next_allocation(DevicePtr(0xa000_0000_0000));
            let valid = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
            gpu.clear();
            layer
                .bind_btile_arena(&store, &config, &gpu, &valid, 77)
                .unwrap();
            assert!(gpu.trace().is_empty());
            let resources = arena_tests::ContextResources::new();
            let ctx = resources.view(&valid, &config, &gpu);
            layer.forward(valid.norm_output(), &ctx, 91).unwrap();
        }
    }
}

#[test]
fn actual_failed_publication_blocks_readers_and_legacy_arena_binding_is_inert() {
    use std::sync::atomic::Ordering;
    let gpu = Gpu::new();
    let (store, config, mut layer) = resident_tests::setup(&gpu, 0);
    let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    gpu.clear();
    layer
        .bind_btile_arena_if_resident(&store, &config, &gpu, &arena, 77)
        .unwrap();
    assert!(gpu.trace().is_empty(), "legacy binding changed GPU events");
    let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
    let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
    gpu.clear();
    session.prepare(&mut layer, &log, &config, 0).unwrap();
    let first_mutation = gpu
        .trace()
        .iter()
        .position(|e| matches!(e, Event::Alloc(..)))
        .unwrap();
    session.close().unwrap();

    let gpu = Gpu::new();
    let (store, config, mut layer) = resident_tests::setup(&gpu, 0);
    let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
    let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
    gpu.clear();
    gpu.fail.store(first_mutation + 1, Ordering::Relaxed);
    assert!(session.prepare(&mut layer, &log, &config, 0).is_err());
    let resources = arena_tests::ContextResources::new();
    let ctx = resources.view(&arena, &config, &gpu);
    gpu.clear();
    assert!(layer.forward(arena.norm_output(), &ctx, 91).is_err());
    assert!(layer.forward_k2(arena.norm_output(), &ctx, 91).is_err());
    assert!(layer.forward_k3(arena.norm_output(), &ctx, 91).is_err());
    assert!(
        layer
            .forward_prefill(arena.norm_output(), 5, &ctx, 91)
            .is_err()
    );
    assert!(
        gpu.trace().is_empty(),
        "failed construction allowed reader work"
    );
    session.close().unwrap();
}

#[test]
fn actual_resident_rejects_stale_views_foreign_context_and_input_bounds_before_work() {
    for case in 0..10 {
        let gpu = Gpu::new();
        let (mut store, config, mut layer) = resident_tests::setup(&gpu, 0);
        let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
        let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
        session.prepare(&mut layer, &log, &config, 0).unwrap();
        session.close().unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
        if case != 0 {
            layer
                .bind_btile_arena(&store, &config, &gpu, &arena, 77)
                .unwrap();
        }
        let resources = arena_tests::ContextResources::new();
        let mut ctx = resources.view(&arena, &config, &gpu);
        let foreign = Gpu::new();
        let mut wrong = config.clone();
        wrong.ep_rank = 1;
        wrong.tp_rank = 1;
        match case {
            1 => layer.shared_down_t.as_mut().unwrap().weight = DevicePtr(16),
            2 => layer.down_ptrs_t.as_mut().unwrap().packed_ptrs = DevicePtr(16),
            3 => layer.weights.shared_expert.gate_proj.weight_scale_2 = 3.0,
            4 => layer.nvfp4_prequant_moe = false,
            5 => ctx.gpu = &foreign,
            6 => ctx.config = &wrong,
            7 => ctx.routed_lora_layers = Some(&[]),
            _ => {}
        }
        let input = if case == 8 {
            arena.norm_output().offset(4 * 8192)
        } else if case == 9 {
            arena.norm_output().offset(2)
        } else {
            arena.norm_output()
        };
        gpu.clear();
        assert!(layer.forward_k3(input, &ctx, 91).is_err(), "case {case}");
        assert!(gpu.trace().is_empty(), "case {case} performed reader work");
        assert!(foreign.trace().is_empty());
    }
}

#[test]
fn actual_arena_binding_rejects_derived_down_alias_and_accepts_moved_arena() {
    let gpu = Gpu::new();
    let (mut store, config, mut layer) = resident_tests::setup(&gpu, 0);
    let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
    let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
    gpu.clear();
    session.prepare(&mut layer, &log, &config, 0).unwrap();
    session.close().unwrap();
    let slab = gpu
        .trace()
        .into_iter()
        .find_map(|e| match e {
            Event::Alloc(p, n) if n == 144 * 4_194_304 => Some(p),
            _ => None,
        })
        .unwrap();
    log.finish().rebuild(&mut store, &gpu).unwrap();
    // Return an actual derived-down address for an actual BufferArena owner.
    gpu.next_allocation(slab);
    let aliased = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    gpu.clear();
    assert!(
        layer
            .bind_btile_arena(&store, &config, &gpu, &aliased, 77)
            .is_err()
    );
    assert!(gpu.trace().is_empty());
    gpu.next_allocation(DevicePtr(0x9000_0000_0000));
    let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    layer
        .bind_btile_arena(&store, &config, &gpu, &arena, 77)
        .unwrap();
    let moved = Box::new(arena);
    let resources = arena_tests::ContextResources::new();
    let ctx = resources.view(&moved, &config, &gpu);
    gpu.clear();
    layer.forward(moved.norm_output(), &ctx, 91).unwrap();
    assert!(
        layer
            .bind_btile_arena(&store, &config, &gpu, &moved, 77)
            .is_err(),
        "binding is once-only"
    );
}
