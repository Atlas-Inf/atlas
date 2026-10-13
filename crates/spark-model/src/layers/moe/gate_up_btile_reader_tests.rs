// SPDX-License-Identifier: AGPL-3.0-only
//! Real forward methods must consume exclusively published GU ownership.
//! Actual publication and reader-entry tests; no forged Ready capability.
use super::recording::{Arg, Event, Gpu};
use super::*;
use spark_runtime::buffers::BufferArena;

#[test]
fn actual_k4_k5_readers_keep_two_wave_offsets_and_routed_only_zero_sinks() {
    for rows in [4, 5] {
        let gpu = Gpu::new();
        let (mut store, config, mut layer) = resident_tests::setup(&gpu, 0);
        let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
        let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
        session.prepare(&mut layer, &log, &config, 0).unwrap();
        session.close().unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
        layer
            .bind_btile_arena(&store, &config, &gpu, &arena, 77)
            .unwrap();
        let resources = arena_tests::ContextResources::new();
        let ctx = resources.view(&arena, &config, &gpu);
        let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
        gpu.clear();
        if rows == 4 {
            layer.forward_k4(arena.norm_output(), &ctx, 91).unwrap();
        } else {
            layer.forward_k5(arena.norm_output(), &ctx, 91).unwrap();
        }
        let calls: Vec<_> = gpu
            .trace()
            .into_iter()
            .filter_map(|e| match e {
                Event::Launch(h, _, _, _, stream, args)
                    if [family.handles[3].0, family.handles[4].0].contains(&h) =>
                {
                    assert_eq!(stream, 91);
                    Some((h, args))
                }
                Event::Read(..) | Event::Sync(..) | Event::Alloc(..) => panic!("reader I/O {e:?}"),
                _ => None,
            })
            .collect();
        assert_eq!(calls.len(), 2, "K{rows} must retain two-wave decomposition");
        for (i, (h, args)) in calls.iter().enumerate() {
            assert_eq!(
                *h,
                family.handles[if i == 1 && rows == 5 { 4 } else { 3 }].0
            );
            assert_eq!(args[0], Arg::Ptr(arena.norm_output().offset(i * 2 * 8192)));
            assert_eq!(args[4], Arg::Ptr(arena.expert_gate_out()));
            assert_eq!(
                args[9],
                Arg::Ptr(arena.scratch().offset(if rows == 5 { i * 64 } else { 0 }))
            );
            assert_eq!(args[13], Arg::Ptr(arena.expert_down_out()));
            assert_eq!(
                args[17],
                Arg::Ptr(arena.expert_down_out().offset(if rows == 5 && i == 1 {
                    3 * 4096
                } else {
                    2 * 4096
                }))
            );
        }
    }
}

#[test]
fn actual_small_readers_close_native_fallbacks_and_preserve_shifted_inputs() {
    for reader in [2, 3, 4] {
        let gpu = Gpu::new();
        let (mut store, config, mut layer) = resident_tests::setup(&gpu, 0);
        let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
        let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
        session.prepare(&mut layer, &log, &config, 0).unwrap();
        session.close().unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
        layer
            .bind_btile_arena(&store, &config, &gpu, &arena, 77)
            .unwrap();
        let resources = arena_tests::ContextResources::new();
        let ctx = resources.view(&arena, &config, &gpu);
        let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
        gpu.clear();
        let input = arena.norm_output().offset(2 * 8192);
        match reader {
            2 => layer.forward_k2(input, &ctx, 91).unwrap(),
            3 => layer.forward_k3(input, &ctx, 91).unwrap(),
            _ => layer.forward_batched(input, 3, &ctx, 91).unwrap(),
        }
        let expected = if reader == 4 {
            family.handles[2].0
        } else {
            family.handles[reader + 1].0
        };
        let calls: Vec<_> = gpu
            .trace()
            .into_iter()
            .filter_map(|e| match e {
                Event::Launch(h, _, _, _, stream, args) if h == expected => {
                    assert_eq!(stream, 91);
                    Some(args)
                }
                Event::Read(..) | Event::Sync(..) | Event::Alloc(..) => panic!("reader I/O {e:?}"),
                _ => None,
            })
            .collect();
        assert_eq!(
            calls.len(),
            if reader == 4 { 3 } else { 1 },
            "reader {reader}"
        );
        for (i, args) in calls.iter().enumerate() {
            assert_eq!(args[0], Arg::Ptr(input.offset(i * 8192)));
            assert_eq!(args[4], Arg::Ptr(arena.expert_gate_out()));
        }
    }
}

#[test]
fn actual_scalar_reader_uses_live_stream_and_original_logits_shared_scratch() {
    let gpu = Gpu::new();
    let (mut store, config, mut layer) = resident_tests::setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
    let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
    session.prepare(&mut layer, &log, &config, 0).unwrap();
    session.close().unwrap();
    log.finish().rebuild(&mut store, &gpu).unwrap();
    let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
    layer
        .bind_btile_arena(&store, &config, &gpu, &arena, 77)
        .unwrap();
    let resources = arena_tests::ContextResources::new();
    let mut ctx = resources.view(&arena, &config, &gpu);
    for (stream, capture) in [(91, false), (92, true)] {
        ctx.graph_capture = capture;
        gpu.capture
            .store(capture, std::sync::atomic::Ordering::Relaxed);
        gpu.clear();
        layer
            .forward(arena.norm_output().offset(8192), &ctx, stream)
            .unwrap();
        let events = gpu.trace();
        let gu: Vec<_> = events
            .iter()
            .filter_map(|event| match event {
                Event::Launch(handle, _, _, _, actual, args) if *handle == family.handles[2].0 => {
                    assert_eq!(*actual, stream);
                    Some(args)
                }
                Event::Launch(_, _, _, _, actual, _) => {
                    assert_eq!(*actual, stream);
                    None
                }
                Event::Read(..) | Event::Sync(..) | Event::Alloc(..) => {
                    panic!("launch-time I/O {event:?}")
                }
                _ => None,
            })
            .collect();
        assert_eq!(gu.len(), 1);
        assert_eq!(gu[0][0], Arg::Ptr(arena.norm_output().offset(8192)));
        assert_eq!(gu[0][13], Arg::Ptr(arena.logits()));
        assert_eq!(gu[0][17], Arg::Ptr(arena.ssm_qkvz()));
    }
}
