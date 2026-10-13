// SPDX-License-Identifier: AGPL-3.0-only
//! Actual FFN/MoE dispatch, not CUDA numerical equivalence.
use super::*;
#[path = "independent_kda_tests.rs"]
mod kda;

#[test]
fn actual_independent_width_entry() {
    const SENTINEL: &str = "ATLAS_TEST_INDEPENDENT_FFN";
    if std::env::var_os(SENTINEL).is_none() {
        let name = concat!(module_path!(), "::actual_independent_width_entry");
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name.split_once("::").unwrap().1, "--nocapture"])
            .env(SENTINEL, "1")
            .env("ATLAS_GLM_INDEPENDENT_DECODE", "1")
            .env("ATLAS_NVFP4_PREQUANT_MOE", "0")
            .env("ATLAS_NVFP4_FUSED_SILU_QUANT", "0")
            .env("ATLAS_GLM_MOE_GATE_UP_M16", "0")
            .env("ATLAS_GLM_MOE_GATE_UP_M16_VERIFY", "0")
            .env("ATLAS_GLM_MOE_DECODE_M16", "0")
            .env("ATLAS_GLM_C2_COMPACT_MOE", "0")
            .env("ATLAS_GLM_C4_GROUPED_MOE", "0")
            .env("ATLAS_GLM_C3_GROUPED_MOE", "0")
            .env("ATLAS_GLM_K5_COMPACT_MOE", "0")
            .env("ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP", "0")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .env("ATLAS_GLM_K5_HC_CUBLAS", "0")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }
    for rank in 0..2 {
        for vector in [false, true] {
            for rows in 2..=8 {
                let gpu = Gpu::new();
                let (_store, config, mut layer) = resident_tests::setup(&gpu, rank);
                layer
                    .transpose_for_prefill_unified_keep_shared(&gpu, &config)
                    .unwrap();
                layer.unified_layout = true;
                layer.nvfp4_vecscale = vector;
                layer.nvfp4_fused_silu_quant = true;
                let arena = BufferArena::new(&config, 8, 2048, 16, 8, &gpu).unwrap();
                let metadata = gpu.alloc(512).unwrap();
                let meta = AttnMetadataDev {
                    positions: metadata,
                    positions_h: metadata,
                    positions_w: metadata,
                    slot: metadata.offset(32),
                    seq_len: metadata.offset(64),
                    block_table: metadata.offset(96),
                    max_blocks_per_seq: 1,
                    num_seqs: rows as u32,
                    seq_slot: metadata,
                    moe_row_adapter: DevicePtr::NULL,
                };
                let resources = ContextResources::new();
                let mut levers = ops::ModelLevers::defaults();
                levers.max_decode_seqs = 8;
                let comm = Comm {
                    gpu: &gpu,
                    rank,
                    reductions: Mutex::new(vec![]),
                };
                let mut ctx = resources.view(&arena, &config, &gpu);
                ctx.levers = &levers;
                ctx.comm = Some(&comm);
                ctx.attn_metadata = Some(meta);
                let h = [gpu.alloc(8 * 2097152).unwrap()];
                let conv = [gpu.alloc(8 * 196608).unwrap()];
                let pool = crate::layer::ssm_batch::SsmPoolView::new(
                    &h, &conv, 2097152, 2097152, 196608, 8,
                )
                .unwrap();
                let slots = [7, 0, 6, 1, 5, 2, 4, 3];
                ctx.ssm_batch = Some(
                    crate::layer::ssm_batch::SsmBatchView::new(
                        pool,
                        metadata.offset(256),
                        &slots[..rows],
                    )
                    .unwrap(),
                );
                let router = layer.dense_gemv_batchm.0;
                let shared = match rows {
                    2 => layer.w4a16_gemv_batch2.0,
                    3 => layer.w4a16_gemv_batch3.0,
                    _ => layer.w4a16_batchm.kernel(rows as u32).0,
                };
                let builder = layer.moe_build_tile_worklist_k.0;
                let fused = if vector {
                    layer.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0
                } else {
                    layer.moe_w4a4_prequant_t_k64_compact_gate_up.0
                };
                // Every selected drain handle is checked, not only this call's width.
                let good = layer.w4a16_gemv_batch3;
                layer.w4a16_gemv_batch3.0 = 0;
                gpu.clear();
                assert!(
                    layer
                        .forward_independent(arena.norm_output(), rows, &ctx, 91)
                        .is_err()
                );
                assert!(gpu.trace().is_empty());
                layer.w4a16_gemv_batch3 = good;
                let good = layer.moe_w4a4_prequant_t_k64_compact;
                layer.moe_w4a4_prequant_t_k64_compact.0 = 0;
                assert!(layer.validate_independent_handles().is_err());
                layer.moe_w4a4_prequant_t_k64_compact = good;
                let view = ctx.ssm_batch;
                ctx.ssm_batch = None;
                gpu.clear();
                assert!(
                    layer
                        .forward_independent(arena.norm_output(), rows, &ctx, 91)
                        .is_err()
                );
                assert!(gpu.trace().is_empty());
                ctx.ssm_batch = view;
                let ffn = FfnComponent::Moe(layer);
                gpu.clear();
                assert_eq!(
                    ffn.forward_independent(arena.norm_output(), rows, &ctx, 91)
                        .unwrap(),
                    arena.moe_output()
                );
                let calls = gpu.trace();
                let kernels: Vec<_> = calls
                    .iter()
                    .filter_map(|e| {
                        if let Event::Launch(h, _, _, _, _, a) = e {
                            Some((*h, a))
                        } else {
                            None
                        }
                    })
                    .collect();
                // One batched router pass covers every row (bit-identical per
                // row to the scalar GEMV).
                let routers: Vec<_> = kernels.iter().filter(|k| k.0 == router).collect();
                assert_eq!(routers.len(), 1);
                assert_eq!(routers[0].1[0], Arg::Ptr(arena.norm_output()));
                assert_eq!(
                    routers[0].1[3],
                    Arg::Bytes((rows as u32).to_ne_bytes().to_vec())
                );
                let shared_calls: Vec<_> = kernels.iter().filter(|k| k.0 == shared).collect();
                assert_eq!(shared_calls.len(), 3);
                for call in shared_calls {
                    assert_eq!(call.1.len(), if rows < 4 { 7 } else { 8 });
                    if rows >= 4 {
                        assert_eq!(call.1[5], Arg::Bytes((rows as u32).to_ne_bytes().to_vec()));
                    }
                }
                assert_eq!(kernels.iter().filter(|k| k.0 == builder).count(), 1);
                assert_eq!(kernels.iter().filter(|k| k.0 == fused).count(), 1);
                assert_eq!(
                    comm.reductions
                        .lock()
                        .unwrap()
                        .iter()
                        .map(|r| r.0)
                        .collect::<Vec<_>>(),
                    vec![rows * 8192]
                );
                assert!(
                    !calls
                        .iter()
                        .any(|e| matches!(e, Event::Alloc(..) | Event::Free(..) | Event::Read(..)))
                );
            }
        }
    }
    kda::actual_kda_rows();
}

/// The GLM router (288 experts: a 72-CTA grid) resolves the load-ahead batchm
/// tier. A GLM kernel set without it, a grid too wide for it, or another
/// model family (whose kernel set is never asked for the GLM symbol) keeps
/// the generic kernel.
#[test]
fn glm_router_resolves_the_load_ahead_batchm_tier() {
    use crate::layers::moe::init_optional::router_gemv_batchm;
    const MODULE: &str = "dense_gemv_bf16_batchm";
    const AHEAD: &str = "dense_gemv_bf16_batchm_ahead";
    let gpu = Gpu::new();
    let (_store, mut config, layer) = resident_tests::setup(&gpu, 0);
    let ahead = gpu.kernel(MODULE, AHEAD).unwrap().0;
    let generic = gpu.kernel(MODULE, MODULE).unwrap().0;
    assert_ne!(ahead, generic);
    assert_eq!(layer.dense_gemv_batchm.0, ahead);

    gpu.clear();
    gpu.lookup_zero.store(true, Ordering::Relaxed);
    gpu.lookup_failure.store(1, Ordering::Relaxed);
    assert_eq!(router_gemv_batchm(&gpu, &config).0, generic);
    gpu.lookup_failure.store(usize::MAX, Ordering::Relaxed);

    let only_generic = [Event::Lookup(MODULE.into(), MODULE.into())];
    config.num_experts = 388;
    gpu.clear();
    assert_eq!(router_gemv_batchm(&gpu, &config).0, generic);
    assert_eq!(gpu.trace(), only_generic);

    config.num_experts = 288;
    config.model_type = "qwen3_next".into();
    gpu.clear();
    assert_eq!(router_gemv_batchm(&gpu, &config).0, generic);
    assert_eq!(gpu.trace(), only_generic);
}
