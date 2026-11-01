// SPDX-License-Identifier: AGPL-3.0-only
//! Real Published readers and communication ordering, not numerical/concurrency emulation.
use super::recording::{Arg, Event, Gpu};
use super::{arena_tests::ContextResources, kernels, load::BTileLoadSession, resident_tests};
use crate::layer::AttnMetadataDev;
use crate::layers::{moe::MoeLayer, ops};
use crate::weight_loader::glm5::retirement::RetirementLog;
use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::{
    buffers::BufferArena,
    gpu::{DevicePtr, GpuBackend},
};
use std::{
    process::Command,
    sync::{Mutex, atomic::Ordering},
};

const SENTINEL: &str = "ATLAS_TEST_BTILE_COMPACT_READER";
#[derive(Clone, Debug, PartialEq, Eq)]
struct Reduction {
    ptr: u64,
    bytes: usize,
    stream: Option<u64>,
    before_event: usize,
}
struct Comm<'g> {
    gpu: &'g Gpu,
    rank: usize,
    calls: Mutex<Vec<Reduction>>,
}
impl Comm<'_> {
    fn reduce(&self, ptr: u64, bytes: usize, stream: Option<u64>) -> Result<()> {
        self.calls.lock().unwrap().push(Reduction {
            ptr,
            bytes,
            stream,
            before_event: self.gpu.trace().len(),
        });
        Ok(())
    }
}
impl CommBackend for Comm<'_> {
    fn all_reduce(&self, p: u64, n: usize) -> Result<()> {
        self.reduce(p, n, None)
    }
    fn all_reduce_async(&self, p: u64, n: usize, s: u64) -> Result<()> {
        self.reduce(p, n, Some(s))
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected all-gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected reduce-scatter")
    }
    fn broadcast(&self, _: u64, _: usize, _: usize) -> Result<()> {
        anyhow::bail!("unexpected broadcast")
    }
    fn barrier(&self) -> Result<()> {
        anyhow::bail!("unexpected barrier")
    }
    fn send_to(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected send")
    }
    fn recv_from(&self, _: u64, _: usize, _: usize, _: u64) -> Result<()> {
        anyhow::bail!("unexpected receive")
    }
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
}
fn u32_arg(value: u32) -> Arg {
    Arg::Bytes(value.to_le_bytes().to_vec())
}

#[test]
fn actual_compact_c4_and_deferred_hc_subprocess_matrix() {
    if let Ok(mode) = std::env::var(SENTINEL) {
        run(&mode);
        return;
    }
    let name = concat!(
        module_path!(),
        "::actual_compact_c4_and_deferred_hc_subprocess_matrix"
    );
    let name = name.split_once("::").unwrap().1;
    for mode in [
        "rows",
        "c4scalar",
        "c4m64",
        "c4m16",
        "k5separate",
        "k5m64",
        "k5m16",
    ] {
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args(["--exact", name, "--nocapture", "--test-threads=1"])
            .env(SENTINEL, mode);
        // Child-only explicit flags: never mutate the parallel parent environment.
        for flag in [
            "ATLAS_HOST_TRANSPOSE",
            "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY",
            "ATLAS_GLM_C3_GROUPED_MOE",
            "ATLAS_GLM_K5_FUSED_SHARED_GATE_UP",
            "ATLAS_GLM_K5_ROUTER_M5",
            "ATLAS_MOE_GROUPED_CUTLASS",
            "ATLAS_HOLO_MOE_GROUPED_CUTLASS",
            "ATLAS_HOLO_MOE_GROUPED_DOWN",
            "ATLAS_MOE_PREFILL_EXACT_TILES",
            "ATLAS_MOE_SHARED_REDUCE_OVERLAP",
            "ATLAS_GLM_SHARED_FP8_CACHE",
        ] {
            command.env(flag, "0");
        }
        for flag in [
            "ATLAS_NVFP4_PREQUANT_MOE",
            "ATLAS_NVFP4_FUSED_SILU_QUANT",
            "ATLAS_GLM_C4_DECODE",
            "ATLAS_GLM_K5_BATCHED_SHARED",
            "ATLAS_GLM_K5_FUSED_MOE_HC",
        ] {
            command.env(flag, "1");
        }
        command.env(
            "ATLAS_GLM_MOE_GATE_UP_M16",
            if mode.ends_with("m16") { "1" } else { "0" },
        );
        command.env(
            "ATLAS_GLM_C4_GROUPED_MOE",
            if mode == "c4m64" || mode == "c4m16" {
                "1"
            } else {
                "0"
            },
        );
        command.env(
            "ATLAS_GLM_K5_GROUPED_MOE",
            if mode.starts_with("k5") { "1" } else { "0" },
        );
        command.env(
            "ATLAS_GLM_K5_COMPACT_MOE",
            if mode.starts_with("k5") { "1" } else { "0" },
        );
        command.env(
            "ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP",
            if mode == "k5m64" || mode == "k5m16" {
                "1"
            } else {
                "0"
            },
        );
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "mode={mode}\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "child filter did not execute"
        );
    }
}

fn run(mode: &str) {
    for rank in [0, 1] {
        for vector in [false, true] {
            for capture in [false, true] {
                let gpu = Gpu::new();
                let (mut store, config, mut layer) = resident_tests::setup(&gpu, rank);
                layer.nvfp4_vecscale = vector;
                layer.nvfp4_fused_silu_quant = true;
                let log = RetirementLog::new(&store, &gpu).unwrap();
                let mut session = BTileLoadSession::new(&gpu, &config, 77).unwrap();
                session.prepare(&mut layer, &log, &config, 0).unwrap();
                session.close().unwrap();
                log.finish().rebuild(&mut store, &gpu).unwrap();
                assert!(layer.btile_storage.is_published());
                let arena = BufferArena::new(&config, 5, 2048, 64, 1, &gpu).unwrap();
                layer
                    .bind_btile_arena(&store, &config, &gpu, &arena, 77)
                    .unwrap();
                let family = kernels::KernelFamily::resolve(&gpu, &config, 77).unwrap();
                let meta = gpu.alloc(256).unwrap();
                let mut metadata_bytes = [0u8; 256];
                for row in 0..4 {
                    metadata_bytes[32 + row * 8..40 + row * 8]
                        .copy_from_slice(&(row as i64).to_le_bytes());
                    metadata_bytes[64 + row * 4..68 + row * 4].copy_from_slice(&1i32.to_le_bytes());
                    metadata_bytes[96 + row * 4..100 + row * 4]
                        .copy_from_slice(&(row as i32).to_le_bytes());
                }
                gpu.copy_h2d(&metadata_bytes, meta).unwrap();
                let metadata = AttnMetadataDev {
                    positions: meta,
                    positions_h: meta,
                    positions_w: meta,
                    slot: meta.offset(32),
                    seq_len: meta.offset(64),
                    block_table: meta.offset(96),
                    max_blocks_per_seq: 1,
                    num_seqs: 4,
                    seq_slot: DevicePtr::NULL,
                    moe_row_adapter: DevicePtr::NULL,
                };
                let comm = Comm {
                    gpu: &gpu,
                    rank,
                    calls: Mutex::new(vec![]),
                };
                let resources = ContextResources::new();
                let mut levers = ops::ModelLevers::defaults();
                levers.max_decode_seqs = 4;
                let mut ctx = resources.view(&arena, &config, &gpu);
                ctx.levers = &levers;
                ctx.comm = Some(&comm);
                ctx.graph_capture = capture;
                ctx.attn_metadata = Some(metadata);
                let stream = if capture { 92 } else { 91 };
                gpu.capture.store(capture, Ordering::Relaxed);
                let cases: Vec<_> = if mode == "rows" {
                    (1..=3).map(|n| (n, false)).collect()
                } else if mode.starts_with("c4") {
                    vec![(4, false)]
                } else {
                    vec![(5, false), (5, true)]
                };
                for (rows, defer) in cases {
                    gpu.clear();
                    comm.calls.lock().unwrap().clear();
                    let input = arena.norm_output();
                    let gate = if rows <= 3 {
                        layer.forward_batched(input, rows, &ctx, stream).unwrap();
                        None
                    } else if rows == 4 {
                        assert_eq!(
                            layer.forward_c4(input, &ctx, stream).unwrap(),
                            arena.moe_output()
                        );
                        None
                    } else {
                        let (out, gate) =
                            layer.forward_k5_for_hc(input, defer, &ctx, stream).unwrap();
                        assert_eq!(out, arena.moe_output());
                        gate
                    };
                    let trace = gpu.trace();
                    assert_no_host_work(&trace, stream);
                    let selected: Vec<_> = trace
                        .iter()
                        .enumerate()
                        .filter_map(|(i, e)| match e {
                            Event::Launch(h, g, b, _, s, a)
                                if family.handles[2..].iter().any(|k| k.0 == *h) =>
                            {
                                Some((i, *h, *g, *b, *s, a))
                            }
                            _ => None,
                        })
                        .collect();
                    let scalar = rows <= 3 || mode == "c4scalar";
                    if scalar {
                        assert_eq!(selected.len(), rows);
                        for (i, (_, h, _, _, _, args)) in selected.iter().enumerate() {
                            assert_eq!(*h, family.handles[if vector { 5 } else { 2 }].0);
                            let row = if rows == 4 { 3 - i } else { i };
                            assert_eq!(args[0], Arg::Ptr(input.offset(row * 8192)));
                        }
                    } else {
                        assert_compact(
                            &layer,
                            &arena,
                            &trace,
                            &selected,
                            &family.handles,
                            (rows, vector, mode, stream),
                        );
                    }
                    let calls = comm.calls.lock().unwrap().clone();
                    assert_eq!(calls.len(), if scalar { rows } else { 1 });
                    for (i, call) in calls.iter().enumerate() {
                        assert_eq!(call.stream, if capture { None } else { Some(stream) });
                        assert_eq!(call.bytes, if scalar { 8192 } else { rows * 8192 });
                        assert_eq!(
                            call.ptr,
                            arena
                                .moe_output()
                                .offset(if rows <= 3 { i * 8192 } else { 0 })
                                .0
                        );
                    }
                    if !scalar {
                        let blends: Vec<_> = trace
                            .iter()
                            .enumerate()
                            .filter_map(|(i, e)| match e {
                                Event::Launch(h, _, _, _, _, a)
                                    if *h == layer.moe_batched_blend.0 =>
                                {
                                    Some((i, a))
                                }
                                _ => None,
                            })
                            .collect();
                        assert_eq!(blends.len(), usize::from(!defer));
                        if let Some((index, args)) = blends.first() {
                            assert!(
                                *index >= calls[0].before_event,
                                "shared blend must follow routed EP reduction"
                            );
                            assert_eq!(args[0], Arg::Ptr(arena.moe_output()));
                            assert_eq!(args[1], Arg::Ptr(arena.attn_output()));
                            assert_eq!(args[5], u32_arg(rows as u32));
                        }
                    }
                    if rows == 5 {
                        assert_eq!(
                            gate,
                            if defer {
                                Some(layer.weights.shared_expert_gate.weight)
                            } else {
                                None
                            }
                        );
                        if let Some(gate) = gate {
                            gpu.clear();
                            comm.calls.lock().unwrap().clear();
                            layer
                                .finish_k5_deferred_shared_blend(
                                    arena.moe_output(),
                                    arena.attn_output(),
                                    input,
                                    gate,
                                    &ctx,
                                    stream,
                                )
                                .unwrap();
                            assert!(comm.calls.lock().unwrap().is_empty());
                            let done = gpu.trace();
                            assert_eq!(done.len(), 1);
                            let Event::Launch(h, _, _, _, s, args) = &done[0] else {
                                panic!("expected deferred blend");
                            };
                            assert_eq!(*h, layer.moe_batched_blend.0);
                            assert_eq!(*s, stream);
                            assert_eq!(
                                &args[..4],
                                &[
                                    Arg::Ptr(arena.moe_output()),
                                    Arg::Ptr(arena.attn_output()),
                                    Arg::Ptr(input),
                                    Arg::Ptr(gate)
                                ]
                            );
                        }
                    }
                }
                if mode.starts_with("c4") {
                    gpu.clear();
                    comm.calls.lock().unwrap().clear();
                    ctx.attn_metadata = None;
                    assert!(layer.forward_c4(arena.norm_output(), &ctx, stream).is_err());
                    assert!(gpu.trace().is_empty() && comm.calls.lock().unwrap().is_empty());
                    let wrong = AttnMetadataDev {
                        num_seqs: 3,
                        ..metadata
                    };
                    ctx.attn_metadata = Some(wrong);
                    assert!(layer.forward_c4(arena.norm_output(), &ctx, stream).is_err());
                    assert!(gpu.trace().is_empty());
                }
            }
        }
    }
}

type GuCall<'a> = (usize, u64, [u32; 3], [u32; 3], u64, &'a Vec<Arg>);
fn assert_compact(
    layer: &MoeLayer,
    arena: &BufferArena,
    trace: &[Event],
    calls: &[GuCall<'_>],
    handles: &[spark_runtime::gpu::KernelHandle; 16],
    policy: (usize, bool, &str, u64),
) {
    let (rows, vector, mode, stream) = policy;
    let separate = mode == "k5separate";
    let expected = if separate {
        14
    } else if mode.ends_with("m16") {
        8
    } else {
        10
    } + usize::from(vector);
    assert_eq!(calls.len(), if separate { 2 } else { 1 });
    let expanded = rows * 8;
    let capacity = expanded * 16;
    let offsets = arena.gate_logits().offset(expanded * 8);
    let work = arena.moe_router_in_f32().offset(16);
    let total = arena.moe_router_in_f32();
    let builders: Vec<_> = trace
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Event::Launch(h, g, b, _, s, a) if *h == layer.moe_build_tile_worklist_k.0 => {
                Some((i, g, b, s, a))
            }
            _ => None,
        })
        .collect();
    assert_eq!(builders.len(), 1, "exactly one GU worklist build");
    let (build_index, g, b, s, args) = builders[0];
    assert_eq!(*g, [1, 1, 1]);
    assert_eq!(*b, [256, 1, 1]);
    assert_eq!(*s, stream);
    assert_eq!(args[0], Arg::Ptr(offsets));
    assert_eq!(args[2], Arg::Ptr(work));
    assert_eq!(args[3], Arg::Ptr(total));
    assert_eq!(&args[4..], &[u32_arg(288), u32_arg(16), u32_arg(64)]);
    let quant: Vec<_> = trace
        .iter()
        .enumerate()
        .filter_map(|(i, e)| match e {
            Event::Launch(h, _, _, _, _, a)
                if *h == layer.quantize_nvfp4_k.0 && a[0] == Arg::Ptr(arena.norm_output()) =>
            {
                Some((i, a))
            }
            _ => None,
        })
        .collect();
    assert_eq!(quant.len(), 1);
    assert!(build_index < quant[0].0 && quant[0].0 < calls[0].0);
    assert_eq!(quant[0].1[1], Arg::Ptr(arena.expert_down_out()));
    assert_eq!(
        quant[0].1[2],
        Arg::Ptr(arena.expert_down_out().offset(rows * 2048))
    );
    for (pair, (index, h, grid, block, s, args)) in calls.iter().enumerate() {
        assert_eq!(*h, handles[expected].0);
        assert_eq!(*s, stream);
        assert_eq!(*block, [128, 1, 1]);
        assert_eq!(*grid, [capacity as u32, if separate { 1 } else { 2 }, 1]);
        assert!(quant[0].0 < *index);
        assert_eq!(args[0], Arg::Ptr(arena.expert_down_out()));
        assert_eq!(
            args[1],
            Arg::Ptr(arena.expert_down_out().offset(rows * 2048))
        );
        assert_eq!(
            args[5],
            Arg::Ptr(if pair == 0 {
                arena.expert_gate_out()
            } else {
                arena.expert_up_out()
            })
        );
        let base = if separate { 6 } else { 10 };
        assert_eq!(args[base], Arg::Ptr(offsets));
        assert_eq!(args[base + 1], Arg::Ptr(arena.gate_logits()));
        assert_eq!(args[base + 5], Arg::Ptr(work));
        assert_eq!(args[base + 6], Arg::Ptr(total));
        assert_eq!(args[base + 7], u32_arg(capacity as u32));
        if !separate {
            assert_eq!(args[9], Arg::Ptr(arena.expert_up_out()));
        }
    }
}
fn assert_no_host_work(trace: &[Event], stream: u64) {
    for event in trace {
        match event {
            Event::Read(..)
            | Event::Sync(..)
            | Event::Alloc(..)
            | Event::Free(..)
            | Event::H2d(..) => panic!("reader host work: {event:?}"),
            Event::Launch(_, _, _, _, actual, _) | Event::Copy(_, _, _, actual) => {
                assert_eq!(*actual, stream)
            }
            _ => {}
        }
    }
}
