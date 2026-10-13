// SPDX-License-Identifier: AGPL-3.0-only
//! Actual legacy-T MoE calls; sparse byte backend proves dispatch, not numerics.
use super::{
    arena_tests::ContextResources,
    recording::{Arg, Event, Gpu},
    resident_tests,
};
use crate::{
    layer::AttnMetadataDev,
    layers::{FfnComponent, ops},
};
use anyhow::Result;
use spark_comm::CommBackend;
use spark_runtime::{
    buffers::BufferArena,
    gpu::{DevicePtr, GpuBackend},
};
use std::sync::{Mutex, atomic::Ordering};

#[path = "decode_m16_tests.rs"]
mod decode_m16;
#[path = "forward_independent_tests.rs"]
mod independent;

struct Comm<'a> {
    gpu: &'a Gpu,
    rank: usize,
    reductions: Mutex<Vec<(usize, Option<u64>, usize)>>,
}
impl Comm<'_> {
    fn reduce(&self, bytes: usize, stream: Option<u64>) -> Result<()> {
        self.reductions
            .lock()
            .unwrap()
            .push((bytes, stream, self.gpu.trace().len()));
        Ok(())
    }
}
impl CommBackend for Comm<'_> {
    fn rank(&self) -> usize {
        self.rank
    }
    fn world_size(&self) -> usize {
        2
    }
    fn all_reduce(&self, _: u64, n: usize) -> Result<()> {
        self.reduce(n, None)
    }
    fn all_reduce_async(&self, _: u64, n: usize, s: u64) -> Result<()> {
        self.reduce(n, Some(s))
    }
    fn all_gather(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected gather")
    }
    fn reduce_scatter(&self, _: u64, _: u64, _: usize) -> Result<()> {
        anyhow::bail!("unexpected scatter")
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
}

#[test]
fn actual_c2_compact_entry_and_literal_off_control() {
    const SENTINEL: &str = "ATLAS_TEST_C2_COMPACT";
    let Ok(mode) = std::env::var(SENTINEL) else {
        let name = concat!(
            module_path!(),
            "::actual_c2_compact_entry_and_literal_off_control"
        );
        let name = name.split_once("::").unwrap().1;
        for mode in ["0", "1"] {
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args(["--exact", name, "--nocapture"])
                .env(SENTINEL, mode)
                .env("ATLAS_GLM_C2_COMPACT_MOE", mode);
            for flag in [
                "ATLAS_HOST_TRANSPOSE",
                "ATLAS_GLM_MOE_GATE_UP_M16",
                "ATLAS_GLM_MOE_GATE_UP_M16_VERIFY",
                "ATLAS_GLM_MOE_DECODE_M16",
                "ATLAS_MOE_SHARED_REDUCE_OVERLAP",
                "ATLAS_HOLO_MOE_GROUPED_CUTLASS",
                "ATLAS_HOLO_MOE_GROUPED_DOWN",
                "ATLAS_GLM_SHARED_FP8_CACHE",
                "ATLAS_GLM_C3_GROUPED_MOE",
                "ATLAS_GLM_C4_GROUPED_MOE",
            ] {
                cmd.env(flag, "0");
            }
            cmd.env("ATLAS_NVFP4_FUSED_SILU_QUANT", "1")
                .env("ATLAS_NVFP4_PREQUANT_MOE", "1");
            let output = cmd.output().unwrap();
            assert!(
                output.status.success(),
                "mode{mode}\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(String::from_utf8_lossy(&output.stdout).contains("1 passed"));
        }
        return;
    };
    for rank in 0..2 {
        for vector in [false, true] {
            for capture in [false, true] {
                let gpu = Gpu::new();
                let (_store, config, mut layer) = resident_tests::setup(&gpu, rank);
                layer
                    .transpose_for_prefill_unified_keep_shared(&gpu, &config)
                    .unwrap();
                layer.unified_layout = true;
                layer.nvfp4_vecscale = vector;
                layer.nvfp4_fused_silu_quant = true;
                let arena = BufferArena::new(&config, 4, 2048, 16, 4, &gpu).unwrap();
                let metadata = gpu.alloc(128).unwrap();
                gpu.copy_h2d(&[3u32.to_le_bytes(), 1u32.to_le_bytes()].concat(), metadata)
                    .unwrap();
                let meta = AttnMetadataDev {
                    positions: metadata,
                    positions_h: metadata,
                    positions_w: metadata,
                    slot: metadata.offset(16),
                    seq_len: metadata.offset(32),
                    block_table: metadata.offset(48),
                    max_blocks_per_seq: 1,
                    num_seqs: 2,
                    seq_slot: metadata,
                    moe_row_adapter: DevicePtr::NULL,
                };
                let resources = ContextResources::new();
                let mut levers = ops::ModelLevers::defaults();
                levers.max_decode_seqs = 4;
                let comm = Comm {
                    gpu: &gpu,
                    rank,
                    reductions: Mutex::new(vec![]),
                };
                let mut ctx = resources.view(&arena, &config, &gpu);
                ctx.comm = Some(&comm);
                ctx.attn_metadata = Some(meta);
                ctx.levers = &levers;
                ctx.graph_capture = capture;
                gpu.capture.store(capture, Ordering::Relaxed);
                let router = layer.dense_gemv_batchm.0;
                let shared = layer.w4a16_gemv_batch2.0;
                let builder = layer.moe_build_tile_worklist_k.0;
                let fused = if vector {
                    layer.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0
                } else {
                    layer.moe_w4a4_prequant_t_k64_compact_gate_up.0
                };
                if mode == "1" {
                    let reject = |layer: &crate::layers::moe::MoeLayer,
                                  ctx: &crate::layer::ForwardContext,
                                  input| {
                        gpu.clear();
                        assert!(layer.forward_c2_compact(input, ctx, 91).is_err());
                        assert!(gpu.trace().is_empty(), "invalid profile performed GPU work");
                    };
                    reject(&layer, &ctx, arena.norm_output().offset(8192));
                    ctx.attn_metadata = None;
                    reject(&layer, &ctx, arena.norm_output());
                    ctx.attn_metadata = Some(AttnMetadataDev {
                        num_seqs: 3,
                        ..meta
                    });
                    reject(&layer, &ctx, arena.norm_output());
                    ctx.attn_metadata = Some(meta);
                    ctx.comm = None;
                    reject(&layer, &ctx, arena.norm_output());
                    ctx.comm = Some(&comm);
                    let original = layer.w4a16_gemv_batch2;
                    layer.w4a16_gemv_batch2.0 = 0;
                    reject(&layer, &ctx, arena.norm_output());
                    layer.w4a16_gemv_batch2 = original;
                    let original = layer.moe_unpermute_reduce_ep;
                    layer.moe_unpermute_reduce_ep.0 = 0;
                    reject(&layer, &ctx, arena.norm_output());
                    layer.moe_unpermute_reduce_ep = original;
                    let original = if vector {
                        layer.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up
                    } else {
                        layer.moe_w4a4_prequant_t_k64_compact_gate_up
                    };
                    if vector {
                        layer.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0 = 0;
                    } else {
                        layer.moe_w4a4_prequant_t_k64_compact_gate_up.0 = 0;
                    }
                    reject(&layer, &ctx, arena.norm_output());
                    if vector {
                        layer.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up = original;
                    } else {
                        layer.moe_w4a4_prequant_t_k64_compact_gate_up = original;
                    }
                    assert!(!layer.glm_c2_grouped(&ctx, 1));
                    assert!(!layer.glm_c2_grouped(&ctx, 3));
                    assert!(!layer.glm_c2_grouped(&ctx, 4));
                    assert!(!layer.glm_c2_grouped(&ctx, 5));
                }
                let ffn = FfnComponent::Moe(layer);
                gpu.clear();
                let result = ffn
                    .try_forward_c2_compact(arena.norm_output(), &ctx, 91)
                    .unwrap();
                if mode == "0" {
                    assert_eq!(result, None);
                    assert!(gpu.trace().is_empty());
                    // The adapter does not replace the caller's scalar loop when off.
                    for row in 0..2 {
                        ffn.forward(arena.norm_output().offset(row * 8192), &ctx, 91)
                            .unwrap();
                    }
                    assert_eq!(
                        comm.reductions
                            .lock()
                            .unwrap()
                            .iter()
                            .map(|r| r.0)
                            .collect::<Vec<_>>(),
                        vec![8192, 8192]
                    );
                    continue;
                }
                assert_eq!(result, Some(arena.moe_output()));
                let calls = gpu.trace();
                let kernels: Vec<_> = calls
                    .iter()
                    .filter_map(|e| {
                        if let Event::Launch(h, g, b, _, s, a) = e {
                            Some((*h, *g, *b, *s, a))
                        } else {
                            None
                        }
                    })
                    .collect();
                // One batched router pass covers both rows.
                let routers: Vec<_> = kernels.iter().filter(|k| k.0 == router).collect();
                assert_eq!(routers.len(), 1);
                assert_eq!(routers[0].4[0], Arg::Ptr(arena.norm_output()));
                assert_eq!(routers[0].4[3], Arg::Bytes(2u32.to_ne_bytes().to_vec()));
                assert_eq!(kernels.iter().filter(|k| k.0 == shared).count(), 3);
                for shared_call in kernels.iter().filter(|k| k.0 == shared) {
                    assert_eq!(shared_call.4.len(), 7, "batch2 ABI has no runtime M");
                }
                assert_eq!(kernels.iter().filter(|k| k.0 == builder).count(), 1);
                assert_eq!(kernels.iter().filter(|k| k.0 == fused).count(), 1);
                let reductions = comm.reductions.lock().unwrap();
                assert_eq!(reductions.len(), 1);
                assert_eq!(
                    (reductions[0].0, reductions[0].1),
                    (16384, if capture { None } else { Some(91) })
                );
                assert!(
                    calls[reductions[0].2..]
                        .iter()
                        .any(|e| matches!(e, Event::Launch(..))),
                    "post-reduce shared blend missing"
                );
                assert!(!calls.iter().any(|e| matches!(
                    e,
                    Event::Read(..) | Event::Alloc(..) | Event::Free(..) | Event::Sync(..)
                )));
            }
        }
    }
}

#[test]
fn c2_toggle_is_strict_and_default_off() {
    use crate::layers::moe::forward_c2::parse_toggle;
    assert!(!parse_toggle(None).unwrap());
    assert!(!parse_toggle(Some("0")).unwrap());
    assert!(parse_toggle(Some("1")).unwrap());
    for bad in ["", "true", "2", " 1"] {
        assert!(parse_toggle(Some(bad)).is_err());
    }
}

#[test]
fn two_row_capacity_proof_rejects_each_undersized_arena() {
    use spark_runtime::buffers::BufferSizes;
    let gpu = Gpu::new();
    let (_, config, _) = resident_tests::setup(&gpu, 0);
    let validate = crate::layers::moe::forward_c4::validate_independent_moe_arenas;
    validate(
        &config,
        &BufferSizes::from_config(&config, 4, 2048, 16, 4),
        2,
    )
    .unwrap();
    let changes: [fn(&mut BufferSizes); 12] = [
        |s| s.norm_output = 16383,
        |s| s.moe_output = 16383,
        |s| s.gate_logits = 1347,
        |s| s.moe_router_in_f32 = 2063,
        |s| s.expert_gate_out = 65535,
        |s| s.expert_up_out = 65535,
        |s| s.expert_down_out = 131071,
        |s| s.ssm_deinterleaved = 8191,
        |s| s.ssm_qkvz = 8191,
        |s| s.attn_output = 16383,
        |s| s.logits = 4095,
        |s| s.scratch = 127,
    ];
    for change in changes {
        let mut sizes = BufferSizes::from_config(&config, 4, 2048, 16, 4);
        change(&mut sizes);
        assert!(validate(&config, &sizes, 2).is_err());
    }
}
