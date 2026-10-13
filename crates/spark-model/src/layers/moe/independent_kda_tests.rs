// SPDX-License-Identifier: AGPL-3.0-only
//! Real KDA constructor and independent entry; backend records launches only.
use super::*;
use crate::layer::{LayerState, SsmLayerState, TransformerLayer};
use crate::layers::glm5_kda::{Glm5KdaLayer, Glm5KdaWeights, Glm5Projection};
use crate::layers::qwen3_attention::{HcSiteWeights, HcWeights};
use crate::weight_map::{DenseWeight, QuantizedWeight};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

fn with_kda(rank: usize, run: impl FnOnce(&Gpu, &atlas_core::config::ModelConfig, &Glm5KdaLayer)) {
    with_kda_ffn(rank, false, run);
}

fn with_kda_ffn(
    rank: usize,
    pair: bool,
    run: impl FnOnce(&Gpu, &atlas_core::config::ModelConfig, &Glm5KdaLayer),
) {
    let gpu = Gpu::new();
    let (_store, mut config, mut ffn) = resident_tests::setup(&gpu, rank);
    let ffn = if pair {
        ffn.transpose_for_prefill_unified_keep_shared(&gpu, &config)
            .unwrap();
        ffn.unified_layout = true;
        ffn.nvfp4_fused_silu_quant = true;
        FfnComponent::Moe(ffn)
    } else {
        FfnComponent::None
    };
    config.num_hidden_layers = 1;
    config.layer_types = vec![atlas_core::config::LayerType::LinearAttention];
    config.linear_num_key_heads = 32;
    config.linear_num_value_heads = 32;
    config.linear_key_head_dim = 128;
    config.linear_value_head_dim = 128;
    config.linear_conv_kernel_dim = 4;
    config.kda_gate_lower_bound = -5.0;
    config.num_attention_heads = 32;
    config.q_lora_rank = 1536;
    config.kv_lora_rank = 512;
    config.qk_nope_head_dim = 256;
    config.qk_rope_head_dim = 0;
    config.v_head_dim = 256;
    config.index_topk = 2048;
    config.hc_mult = 4;
    let dense = |bytes| DenseWeight {
        weight: gpu.alloc(bytes).unwrap(),
    };
    let projection = || Glm5Projection {
        nvfp4: QuantizedWeight {
            weight: gpu.alloc(4096 * 4096 / 2).unwrap(),
            weight_scale: gpu.alloc(4096 * 4096 / 16).unwrap(),
            weight_scale_2: 1.0,
            input_scale: DevicePtr::NULL,
            weight_scale_2_vec: DevicePtr::NULL,
        },
        prefill_nvfp4_t: None,
    };
    let site = || HcSiteWeights {
        hc_fn: gpu.alloc(24 * 4 * 4096 * 4).unwrap(),
        hc_fn_bf16: DevicePtr::NULL,
        hc_base: gpu.alloc(24 * 4).unwrap(),
        hc_scale: gpu.alloc(12).unwrap(),
        lowrank: None,
    };
    let weights = Glm5KdaWeights {
        q_proj: projection(),
        k_proj: projection(),
        v_proj: projection(),
        o_proj: projection(),
        b_proj: dense(32 * 4096 * 2),
        f_a_proj: dense(128 * 4096 * 2),
        f_b_proj: dense(4096 * 128 * 2),
        g_a_proj: dense(128 * 4096 * 2),
        g_b_proj: dense(4096 * 128 * 2),
        conv: dense(12288 * 4 * 2),
        a_log: dense(32 * 4),
        dt_bias: dense(4096 * 4),
        o_norm: dense(4096 * 2),
    };
    let layer = Glm5KdaLayer::new(
        dense(8192),
        dense(8192),
        weights,
        ffn,
        HcWeights {
            attn: site(),
            ffn: site(),
            head: None,
            hc_mult: 4,
            sinkhorn_iters: 1,
            hc_eps: 1e-6,
            is_first_model_layer: true,
            is_last_model_layer: true,
        },
        0,
        &config,
        &gpu,
    )
    .unwrap();
    run(&gpu, &config, &layer);
}

pub(super) fn actual_kda_rows() {
    for rank in 0..2 {
        with_kda(rank, |gpu, config, layer| {
            let arena = BufferArena::new(config, 8, 2048, 16, 8, gpu).unwrap();
            let h = [gpu.alloc(8 * 2097152).unwrap()];
            let conv = [gpu.alloc(8 * 196608).unwrap()];
            let metadata = gpu.alloc(512).unwrap();
            let pool =
                crate::layer::ssm_batch::SsmPoolView::new(&h, &conv, 2097152, 2097152, 196608, 8)
                    .unwrap();
            let ids = [7, 0, 6, 1, 5, 2, 4, 3];
            let resources = ContextResources::new();
            let mut levers = ops::ModelLevers::defaults();
            levers.max_decode_seqs = 8;
            let comm = Comm {
                gpu,
                rank,
                reductions: Mutex::new(vec![]),
            };
            let conv_kernel = gpu
                .kernel("causal_conv1d", "glm_kda_conv_indexed")
                .unwrap()
                .0;
            let recurrent = gpu.kernel("kda", "glm_kda_recurrent_indexed").unwrap().0;
            let mut cache = PagedKvCache::new(
                KvCacheConfig {
                    block_size: 16,
                    num_kv_heads: 1,
                    head_dim: 512,
                    num_layers: 1,
                    dtype: KvCacheDtype::Bf16,
                    layer_dtypes: vec![],
                    layer_dims: vec![],
                    cache_blocks_per_seq: None,
                },
                8,
                gpu,
            )
            .unwrap();
            for rows in 2..=8 {
                let mut states: Vec<Box<dyn LayerState>> = ids[..rows]
                    .iter()
                    .map(|&i| {
                        Box::new(SsmLayerState {
                            h_state: h[0].offset(i as usize * 2097152),
                            conv_state: conv[0].offset(i as usize * 196608),
                            h_is_f16: false,
                            h_state_checkpoint: None,
                            conv_state_checkpoint: None,
                            h_state_intermediates: vec![],
                            kda_records: spark_runtime::gpu::DevicePtr::NULL,
                            gdn_commit_qkv: DevicePtr(0),
                            gdn_commit_gb: DevicePtr(0),
                            gdn_commit_pending: false,
                            conv_state_intermediates: vec![],
                            h_prefill_stage: None,
                            ple: None,
                        }) as Box<dyn LayerState>
                    })
                    .collect();
                let mut refs: Vec<&mut dyn LayerState> = states
                    .iter_mut()
                    .map(|s| s.as_mut() as &mut dyn LayerState)
                    .collect();
                let mut ctx = resources.view(&arena, config, gpu);
                ctx.levers = &levers;
                ctx.comm = Some(&comm);
                ctx.ssm_batch = Some(
                    crate::layer::ssm_batch::SsmBatchView::new(
                        pool,
                        metadata.offset(256),
                        &ids[..rows],
                    )
                    .unwrap(),
                );
                ctx.attn_metadata = Some(AttnMetadataDev {
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
                });
                let positions: Vec<_> = (0..rows).map(|i| i * 17).collect();
                let blocks = vec![vec![]; rows];
                gpu.clear();
                layer
                    .decode_multi_seq(
                        arena.hidden_states(),
                        arena.hidden_states(),
                        rows,
                        rows,
                        &mut refs,
                        &mut cache,
                        &positions,
                        &blocks,
                        &ctx,
                        91,
                    )
                    .unwrap();
                let trace = gpu.trace();
                for (kernel, grid) in [
                    (conv_kernel, [48, rows as u32, 1]),
                    (recurrent, [32, rows as u32, 1]),
                ] {
                    let calls: Vec<_> = trace
                        .iter()
                        .filter_map(|e| {
                            if let Event::Launch(k, g, _, _, _, args) = e {
                                (*k == kernel).then_some((*g, args))
                            } else {
                                None
                            }
                        })
                        .collect();
                    assert_eq!(calls.len(), 1);
                    assert_eq!(calls[0].0, grid);
                    assert!(calls[0].1.contains(&Arg::Ptr(metadata.offset(256))));
                }
                assert!(
                    !trace
                        .iter()
                        .any(|e| matches!(e, Event::Alloc(..) | Event::Free(..) | Event::Read(..)))
                );
            }
        });
    }
}

const K128_DUAL: (&str, &str) = ("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm_dual_k128");

#[test]
fn kda_constructor_resolves_the_k128_dual_before_the_boot_seal() {
    with_kda(0, |gpu, _, _| {
        let lookup = Event::Lookup(K128_DUAL.0.into(), K128_DUAL.1.into());
        assert!(gpu.trace().contains(&lookup));
    });
}

/// `(func, grid)` of the f_b/g_b dual at K = 128 on `gpu`.
fn k128_dual_launch(gpu: &Gpu) -> (u64, [u32; 3]) {
    let w = [0x100, 0x200].map(|p| DenseWeight {
        weight: DevicePtr(p),
    });
    let io = |p| [DevicePtr(p), DevicePtr(p + 0x100)];
    ops::dense_gemv_batchm_dual(
        gpu,
        spark_runtime::gpu::KernelHandle(7),
        io(0x300),
        [&w[0], &w[1]],
        io(0x500),
        8,
        4096,
        128,
        0,
    )
    .unwrap();
    match gpu.trace().last() {
        Some(Event::Launch(k, grid, ..)) => (*k, *grid),
        e => panic!("expected the dual launch, got {e:?}"),
    }
}

#[test]
fn k128_dual_handle_belongs_to_its_backend() {
    let (func, grid) = k128_dual_launch(&Gpu::new());
    assert_ne!(func, 7);
    assert_eq!(grid, [4096 / 16, 1, 2]);
    // A second backend (another model's registry) that lacks the tier must
    // not launch the first backend's handle: it keeps the generic dual.
    let without = Gpu::new();
    without.lookup_failure.store(1, Ordering::Relaxed);
    assert_eq!(k128_dual_launch(&without), (7, [4096 / 4, 1, 2]));
}

/// Non-verify prefill rows run beta | f_a | g_a as the one-grid triple, in
/// plane order; `ATLAS_GLM_KDA_FUSED_SMALL_PREFILL=0` (re-run in a child
/// process) restores the three pipelined launches.
#[test]
fn kda_prefill_small_projections_fuse_unless_killed() {
    const KILL: &str = "ATLAS_GLM_KDA_FUSED_SMALL_PREFILL";
    let killed = std::env::var(KILL).as_deref() == Ok("0");
    if !killed {
        let name = concat!(
            module_path!(),
            "::kda_prefill_small_projections_fuse_unless_killed"
        );
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name.split_once("::").unwrap().1, "--nocapture"])
            .env(KILL, "0")
            .output()
            .unwrap();
        assert!(
            output.status.success() && String::from_utf8_lossy(&output.stdout).contains("1 passed"),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    with_kda(0, |gpu, config, layer| {
        const M: usize = 300;
        let arena = BufferArena::new(config, M, 2048, 16, 1, gpu).unwrap();
        let resources = ContextResources::new();
        let ctx = resources.view(&arena, config, gpu);
        let mut state = layer.alloc_state(gpu).unwrap();
        let mut cache = PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 512,
                num_layers: 1,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            8,
            gpu,
        )
        .unwrap();
        // The fixture allocates b, f_a, f_b, g_a, g_b back to back.
        let allocs: Vec<_> = gpu
            .trace()
            .iter()
            .filter_map(|e| match e {
                Event::Alloc(p, bytes) => Some((*p, *bytes)),
                _ => None,
            })
            .collect();
        let b = allocs
            .iter()
            .position(|&(_, bytes)| bytes == 32 * 4096 * 2)
            .unwrap();
        let [b, f_a, g_a] = [b, b + 1, b + 3].map(|i| Arg::Ptr(allocs[i].0));
        let kernel = |name| gpu.kernel("gemm", name).unwrap().0;
        let triple = kernel("dense_gemm_bf16_pipelined_triple_n");
        let pipelined = kernel("dense_gemm_bf16_pipelined");
        let hidden = arena.hidden_states();
        let (mut blocks, mut disk, mut offloaded) = (vec![], vec![], vec![]);
        gpu.clear();
        layer
            .prefill(
                hidden,
                hidden,
                M,
                state.as_mut(),
                &mut cache,
                0,
                &mut blocks,
                &mut disk,
                &mut offloaded,
                0,
                &ctx,
                91,
            )
            .unwrap();
        let trace = gpu.trace();
        let launches = |kernel| -> Vec<_> {
            trace
                .iter()
                .filter_map(|e| match e {
                    Event::Launch(k, grid, _, _, _, args) if *k == kernel => Some((*grid, args)),
                    _ => None,
                })
                .collect()
        };
        let u32_arg = |v: u32| Arg::Bytes(v.to_ne_bytes().to_vec());
        let normed = Arg::Ptr(arena.norm_output());
        let beta = arena.qkv_output().offset(3 * M * 4096 * 2);
        let fa = beta.offset(M * 32 * 2);
        let planes = [beta, fa, fa.offset(M * 128 * 2)].map(Arg::Ptr);
        let (fused, split) = (launches(triple), launches(pipelined));
        // (weight, output, N, K) of each pipelined launch, in launch order.
        let split: Vec<_> = split
            .iter()
            .map(|(_, a)| (&a[1], &a[2], &a[4], &a[5]))
            .collect();
        let small = [
            (&b, &planes[0], 32),
            (&f_a, &planes[1], 128),
            (&g_a, &planes[2], 128),
        ];
        if killed {
            assert!(fused.is_empty());
            assert_eq!(split.len(), 5);
            for ((weight, plane, n), got) in small.into_iter().zip(&split) {
                assert_eq!(*got, (weight, plane, &u32_arg(n), &u32_arg(4096)));
            }
        } else {
            assert_eq!(fused.len(), 1);
            let (grid, args) = &fused[0];
            assert_eq!(*grid, [3, 3, 1]);
            let mut want = vec![normed, b.clone(), f_a.clone(), g_a.clone()];
            want.extend(planes.iter().cloned());
            want.extend([M as u32, 32, 128, 4096].map(u32_arg));
            assert_eq!(**args, want);
            assert_eq!(split.len(), 2);
        }
        // f_b / g_b (K = 128) follow on the pipelined kernel (CUTLASS is off here).
        for got in &split[split.len() - 2..] {
            assert_eq!((got.2, got.3), (&u32_arg(4096), &u32_arg(128)));
        }
        // Verify rows and single-token decode never take the prefill triple.
        let ssm = state.as_any_mut().downcast_mut::<SsmLayerState>().unwrap();
        ssm.h_state_intermediates = (0..4).map(|_| gpu.alloc(2097152).unwrap()).collect();
        ssm.conv_state_intermediates = (0..5).map(|_| gpu.alloc(196608).unwrap()).collect();
        gpu.clear();
        layer
            .decode_batched(
                hidden,
                hidden,
                5,
                state.as_mut(),
                &mut cache,
                0,
                &mut blocks,
                &mut disk,
                &mut offloaded,
                &ctx,
                91,
            )
            .unwrap();
        layer
            .decode(
                hidden,
                hidden,
                state.as_mut(),
                &mut cache,
                0,
                &mut blocks,
                &mut disk,
                &mut offloaded,
                &ctx,
                91,
            )
            .unwrap();
        let trace = gpu.trace();
        assert!(trace.iter().any(|e| matches!(e, Event::Launch(..))));
        assert!(
            !trace
                .iter()
                .any(|e| matches!(e, Event::Launch(k, ..) if *k == triple))
        );
    });
}
