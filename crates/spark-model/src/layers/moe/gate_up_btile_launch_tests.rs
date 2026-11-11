// SPDX-License-Identifier: AGPL-3.0-only
use super::recording::{Arg, Event, Gpu};
use super::*;
use spark_runtime::buffers::BufferArena;

#[test]
fn actual_private_lease_submits_decode_and_grouped_abis() {
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
    let arena = BufferArena::new(&config, 1088, 2048, 64, 1, &gpu).unwrap();
    let resources = arena_tests::ContextResources::new();
    let ctx = resources.view(&arena, &config, &gpu);
    let lease = lease.check_arena(&ctx).unwrap();
    gpu.clear();
    lease
        .decode(
            &arena,
            decode::DecodeRows {
                count: 3,
                input: 0,
                routes: 0,
                output: 0,
            },
            decode::WordPolicy::Vector,
            decode::SharedMode::RoutedOnly,
        )
        .unwrap();
    lease
        .grouped(
            &arena,
            1088,
            grouped::ScalePolicy::Vector,
            grouped::InputLayout::Gathered,
            grouped::GroupedMode::Dense,
        )
        .unwrap();
    let trace = gpu.trace();
    assert_eq!(trace.len(), 3);
    assert!(matches!(&trace[0],Event::Launch(_, [64,27,2],[32,1,1],0,77,a) if a.len()==21));
    assert!(
        trace[1..]
            .iter()
            .all(|e| matches!(e,Event::Launch(_, [16,17,288],[128,1,1],0,77,a) if a.len()==11))
    );
}

fn scalar(value: u32) -> Arg {
    Arg::Bytes(value.to_le_bytes().to_vec())
}

#[test]
fn actual_all_decode_exports_and_grouped_graph_abis_are_exact() {
    use decode::{SharedMode, WordPolicy};
    use grouped::{GroupedMode, InputLayout, ScalePolicy};
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
    let arena = BufferArena::new(&config, 1088, 2048, 64, 1, &gpu).unwrap();
    let resources = arena_tests::ContextResources::new();
    let ctx = resources.view(&arena, &config, &gpu);
    let lease = lease.check_arena(&ctx).unwrap();
    for (word, index) in [(WordPolicy::Word, 2), (WordPolicy::Vector, 5)] {
        for rows in 1..=3 {
            gpu.clear();
            lease
                .decode(
                    &arena,
                    decode::DecodeRows {
                        count: rows,
                        input: 1,
                        routes: 1,
                        output: 0,
                    },
                    word,
                    SharedMode::RoutedOnly,
                )
                .unwrap();
            let mut args = vec![Arg::Ptr(arena.norm_output().offset(8192))];
            for (base, out) in [(0, arena.expert_gate_out()), (3, arena.expert_up_out())] {
                args.extend(
                    lease.lease.tables[base..base + 3]
                        .iter()
                        .map(|s| Arg::Ptr(s.ptr)),
                );
                args.push(Arg::Ptr(out));
            }
            args.push(Arg::Ptr(arena.scratch().offset(32)));
            for out in [
                arena.expert_down_out(),
                arena.expert_down_out().offset(rows * 4096),
            ] {
                args.extend([
                    Arg::Ptr(spark_runtime::gpu::DevicePtr::NULL),
                    Arg::Ptr(spark_runtime::gpu::DevicePtr::NULL),
                    scalar(0),
                    Arg::Ptr(out),
                ]);
            }
            args.extend([scalar(2048), scalar(4096), scalar(8)]);
            assert_eq!(
                gpu.trace(),
                vec![Event::Launch(
                    family.handles[index + rows - 1].0,
                    [64, rows as u32 * 9, 2],
                    [32, 1, 1],
                    0,
                    77,
                    args
                )]
            );
        }
    }
    for (scale, vector) in [(ScalePolicy::Scalar, 0), (ScalePolicy::Vector, 1)] {
        for (layout, gathered) in [
            (InputLayout::Gathered, true),
            (InputLayout::RouteMajor, false),
        ] {
            for rows in [1usize, 5, 6, 64, 1088] {
                for mode in [
                    GroupedMode::Dense,
                    GroupedMode::SeparateCompact,
                    GroupedMode::Fused {
                        prefer_small: false,
                    },
                    GroupedMode::Fused { prefer_small: true },
                ] {
                    gpu.clear();
                    lease.grouped(&arena, rows, scale, layout, mode).unwrap();
                    let expanded = rows * 8;
                    let max = expanded * 16;
                    let prefix = vec![
                        Arg::Ptr(arena.expert_down_out()),
                        Arg::Ptr(arena.expert_down_out().offset(if gathered {
                            rows * 2048
                        } else {
                            expanded * 2048
                        })),
                    ];
                    let metadata = vec![
                        Arg::Ptr(arena.gate_logits().offset(expanded * 8)),
                        Arg::Ptr(if gathered {
                            arena.gate_logits()
                        } else {
                            spark_runtime::gpu::DevicePtr::NULL
                        }),
                        scalar(288),
                        scalar(2048),
                        scalar(4096),
                    ];
                    let work = vec![
                        Arg::Ptr(arena.moe_router_in_f32().offset(16)),
                        Arg::Ptr(arena.moe_router_in_f32()),
                        scalar(max as u32),
                    ];
                    let mut expected = Vec::new();
                    if let GroupedMode::Fused { prefer_small } = mode {
                        let mut args = prefix.clone();
                        for (base, out) in
                            [(0, arena.expert_gate_out()), (3, arena.expert_up_out())]
                        {
                            args.extend(
                                lease.lease.tables[base..base + 3]
                                    .iter()
                                    .map(|s| Arg::Ptr(s.ptr)),
                            );
                            args.push(Arg::Ptr(out));
                        }
                        args.extend(metadata.clone());
                        args.extend(work.clone());
                        let index = if prefer_small && gathered && rows <= 5 {
                            8 + vector
                        } else {
                            10 + vector
                        };
                        expected.push(Event::Launch(
                            family.handles[index].0,
                            [max as u32, 2, 1],
                            [128, 1, 1],
                            0,
                            77,
                            args,
                        ));
                    } else {
                        let compact = matches!(mode, GroupedMode::SeparateCompact);
                        for (base, out) in
                            [(0, arena.expert_gate_out()), (3, arena.expert_up_out())]
                        {
                            let mut args = prefix.clone();
                            args.extend(
                                lease.lease.tables[base..base + 3]
                                    .iter()
                                    .map(|s| Arg::Ptr(s.ptr)),
                            );
                            args.push(Arg::Ptr(out));
                            args.extend(metadata.clone());
                            if compact {
                                args.extend(work.clone());
                            }
                            expected.push(Event::Launch(
                                family.handles[if compact { 14 + vector } else { 12 + vector }].0,
                                if compact {
                                    [max as u32, 1, 1]
                                } else {
                                    [16, rows.div_ceil(64) as u32, 288]
                                },
                                [128, 1, 1],
                                0,
                                77,
                                args,
                            ));
                        }
                    }
                    assert_eq!(gpu.trace(), expected);
                }
            }
        }
    }
}

#[test]
fn actual_launch_bounds_and_backend_failure_do_no_fallback_work() {
    use decode::{SharedMode, WordPolicy};
    use grouped::{GroupedMode, InputLayout, ScalePolicy};
    use std::sync::atomic::Ordering;
    let gpu = Gpu::new();
    let (store, config, local, mut layer) = binding_tests::setup(&gpu, 0);
    let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
    let source = NativeGateUpLayer::from_store(&store, &config, 0, &local, &gpu, 77).unwrap();
    let mut workspace = RepackWorkspace::new(&gpu, 77).unwrap();
    let unpublished = workspace.repack(source, &family).unwrap();
    workspace.close().unwrap();
    let lease = binding::Lease::bind(&unpublished, &family, &mut layer).unwrap();
    for capacity in [1, 5, 1088, 1089] {
        let arena = BufferArena::new(&config, capacity, 2048, 64, 1, &gpu).unwrap();
        let resources = arena_tests::ContextResources::new();
        let ctx = resources.view(&arena, &config, &gpu);
        if capacity > 1088 {
            assert!(lease.check_arena(&ctx).is_err());
            continue;
        }
        let lease = lease.check_arena(&ctx).unwrap();
        gpu.clear();
        for rows in [0, capacity + 1, usize::MAX] {
            assert!(
                lease
                    .grouped(
                        &arena,
                        rows,
                        ScalePolicy::Scalar,
                        InputLayout::Gathered,
                        GroupedMode::Dense
                    )
                    .is_err()
            );
            assert!(
                lease
                    .decode(
                        &arena,
                        decode::DecodeRows {
                            count: rows,
                            input: 0,
                            routes: 0,
                            output: 0
                        },
                        WordPolicy::Word,
                        SharedMode::RoutedOnly
                    )
                    .is_err()
            );
        }
        assert!(
            lease
                .decode(
                    &arena,
                    decode::DecodeRows {
                        count: 1,
                        input: usize::MAX,
                        routes: usize::MAX,
                        output: 0
                    },
                    WordPolicy::Word,
                    SharedMode::RoutedOnly
                )
                .is_err()
        );
        assert!(gpu.trace().is_empty());
        // A one-row arena's gate-logit owner cannot hold all 289 offsets;
        // rejection there is the intended host-capacity gate, not a launch.
        if (5..=1088).contains(&capacity) {
            for mode in [GroupedMode::Dense, GroupedMode::SeparateCompact] {
                for fail in 1..=2 {
                    gpu.clear();
                    gpu.fail.store(fail, Ordering::Relaxed);
                    assert!(
                        lease
                            .grouped(&arena, 1, ScalePolicy::Scalar, InputLayout::Gathered, mode)
                            .is_err()
                    );
                    assert_eq!(gpu.trace().len(), fail);
                }
            }
            gpu.clear();
            gpu.fail.store(1, Ordering::Relaxed);
            assert!(
                lease
                    .grouped(
                        &arena,
                        1,
                        ScalePolicy::Scalar,
                        InputLayout::Gathered,
                        GroupedMode::Fused { prefer_small: true }
                    )
                    .is_err()
            );
            assert_eq!(gpu.trace().len(), 1);
            gpu.clear();
            gpu.fail.store(1, Ordering::Relaxed);
            assert!(
                lease
                    .decode(
                        &arena,
                        decode::DecodeRows {
                            count: 1,
                            input: 0,
                            routes: 0,
                            output: 0,
                        },
                        WordPolicy::Word,
                        SharedMode::RoutedOnly
                    )
                    .is_err()
            );
            assert_eq!(gpu.trace().len(), 1);
            gpu.clear();
        }
    }
}
