// SPDX-License-Identifier: AGPL-3.0-only
//! Actual prefill producer -> private resident GU -> existing routed down chain.
use super::recording::{Arg, Event, Gpu};
use super::*;
use spark_runtime::buffers::BufferArena;

#[test]
fn actual_prefill_uses_promoted_readers_through_padded_1088_without_readback() {
    for vector in [false, true] {
        let gpu = Gpu::new();
        let (mut store, config, mut layer) = resident_tests::setup(&gpu, 0);
        layer.nvfp4_vecscale = vector;
        let log = crate::weight_loader::glm5::retirement::RetirementLog::new(&store, &gpu).unwrap();
        let mut session = load::BTileLoadSession::new(&gpu, &config, 77).unwrap();
        session.prepare(&mut layer, &log, &config, 0).unwrap();
        session.close().unwrap();
        log.finish().rebuild(&mut store, &gpu).unwrap();
        let arena = BufferArena::new(&config, 1088, 2048, 64, 1, &gpu).unwrap();
        layer
            .bind_btile_arena(&store, &config, &gpu, &arena, 77)
            .unwrap();
        let resources = arena_tests::ContextResources::new();
        let mut ctx = resources.view(&arena, &config, &gpu);
        let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
        for capture in [false, true] {
            ctx.graph_capture = capture;
            gpu.capture
                .store(capture, std::sync::atomic::Ordering::Relaxed);
            for rows in [1, 5, 6, 64, 129, 1025, 1088] {
                gpu.clear();
                layer
                    .forward_prefill(arena.norm_output(), rows, &ctx, 91)
                    .unwrap();
                let calls: Vec<_> = gpu
                    .trace()
                    .into_iter()
                    .filter_map(|e| match e {
                        Event::Launch(h, grid, _, _, stream, args)
                            if h == family.handles[12 + usize::from(vector)].0 =>
                        {
                            assert_eq!(stream, 91);
                            Some((grid, args))
                        }
                        Event::Read(..) | Event::Sync(..) | Event::Alloc(..) => {
                            panic!("prefill I/O {e:?}")
                        }
                        _ => None,
                    })
                    .collect();
                assert_eq!(calls.len(), 2, "rows={rows} capture={capture}");
                for (grid, args) in calls {
                    assert_eq!(grid, [16, rows.div_ceil(64) as u32, 288]);
                    assert_eq!(args[0], Arg::Ptr(arena.expert_down_out()));
                    assert_eq!(
                        args[1],
                        Arg::Ptr(arena.expert_down_out().offset(rows * 2048))
                    );
                    assert_eq!(args[6], Arg::Ptr(arena.gate_logits().offset(rows * 8 * 8)));
                    assert_eq!(args[7], Arg::Ptr(arena.gate_logits()));
                }
            }
        }
    }
}
