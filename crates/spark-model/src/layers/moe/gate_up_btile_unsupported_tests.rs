// SPDX-License-Identifier: AGPL-3.0-only
//! Actual public converters/readers must refuse a real published allocation set.
use super::recording::Gpu;
use super::*;
use spark_runtime::buffers::BufferArena;

#[test]
fn actual_native_converters_refuse_published_storage_without_work() {
    for case in 0..8 {
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
        gpu.clear();
        let result = match case {
            0 => layer.transpose_for_prefill(&gpu, &config),
            1 => layer.transpose_gate_up_for_prefill(&gpu, &config),
            2 => layer
                .transpose_for_prefill_unified(&gpu, &config)
                .map(|_| ()),
            3 => layer.transpose_for_prefill_hybrid(&gpu, &config),
            4 => layer.build_cutlass_grouped_sfb(&gpu, &config, 91),
            5 => layer.repack_nvfp4_mmq_unified(&gpu, &config),
            6 => layer.forward_atomic_c4_decode(arena.norm_output(), 4, &ctx, 91),
            _ => layer.forward_token_major_decode(arena.norm_output(), 4, &ctx, 91),
        };
        assert!(result.is_err(), "unsupported reader/converter {case}");
        assert!(
            gpu.trace().is_empty(),
            "unsupported reader/converter {case} performed I/O"
        );
    }
}
