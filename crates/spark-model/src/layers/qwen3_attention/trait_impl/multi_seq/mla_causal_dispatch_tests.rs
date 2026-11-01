// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

#[test]
fn actual_prompt_index_and_causal_verify_dispatch() {
    const CHILD: &str = "ATLAS_TEST_LONG_MTP_ATTENTION";
    if std::env::var_os(CHILD).is_none() {
        let name = concat!(
            module_path!(),
            "::actual_prompt_index_and_causal_verify_dispatch"
        );
        for (tc, o_batch, o_compare, q_batch, q_compare) in [
            ("0", "0", "0", "0", "0"),
            ("1", "0", "0", "0", "0"),
            ("0", "1", "0", "0", "0"),
            ("1", "1", "0", "0", "0"),
            ("0", "1", "1", "0", "0"),
            ("1", "1", "1", "0", "0"),
            // Query batching is checked with O batching disabled so each
            // projection's handle and output slice remain unambiguous.
            ("0", "0", "0", "1", "0"),
            ("0", "0", "0", "1", "1"),
            ("1", "1", "0", "1", "0"),
            ("1", "1", "1", "1", "1"),
        ] {
            let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
            cmd.args(["--exact", name.split_once("::").unwrap().1, "--nocapture"]);
            for (k, _) in std::env::vars_os() {
                if k.to_string_lossy().starts_with("ATLAS_") {
                    cmd.env_remove(k);
                }
            }
            let out = cmd
                .env(CHILD, "1")
                .env("ATLAS_GLM_MTP_LONG_CONTEXT", "1")
                .env("ATLAS_GLM_MTP_REPAIR", "1")
                .env("ATLAS_GLM_SPARSE_DECODE_TC", tc)
                .env("ATLAS_GLM_K3_MLA_O_BATCHM", o_batch)
                .env("ATLAS_GLM_K3_MLA_O_COMPARE", o_compare)
                .env("ATLAS_GLM_K3_MLA_QUERY_BATCHM", q_batch)
                .env("ATLAS_GLM_K3_MLA_QUERY_COMPARE", q_compare)
                .output()
                .unwrap();
            assert!(String::from_utf8_lossy(&out.stdout).contains("running 1 test"));
            assert!(
                out.status.success(),
                "TC={tc}, O batchm={o_batch}, O compare={o_compare}, \
                 Q batchm={q_batch}, Q compare={q_compare}: {}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        return;
    }
    fixture(|gpu, config, layer| {
        if std::env::var("ATLAS_GLM_K3_MLA_O_BATCHM").as_deref() == Ok("1") {
            use crate::layers::qwen3_attention::glm_k3_mla_o::initialize;
            let ptr = layer.mla.as_ref().unwrap().wo.weight;
            initialize(gpu, config, KvCacheDtype::Bf16, ptr, 4096, 8192).unwrap();
            let mut wrong = config.clone();
            wrong.num_attention_heads = 64;
            assert!(initialize(gpu, &wrong, KvCacheDtype::Bf16, ptr, 4096, 8192).is_err());
            assert!(initialize(gpu, config, KvCacheDtype::Fp8, ptr, 4096, 8192).is_err());
        }
        let sparse_kernel = if ops::glm_sparse_decode_tc_enabled(&config.model_type).unwrap() {
            gpu.kernel(
                "glm_sparse_prefill_kv_reuse",
                "glm_sparse_mla_prefill_bf16_head32_tc_kv_pad",
            )
            .unwrap()
            .0
        } else {
            804
        };
        let arena = BufferArena::new(config, 16, 32768, 16, 1, gpu).unwrap();
        let mut dispatch = ops::GemmDispatch::defaults();
        dispatch.cublas_gemm = false;
        let derived = ops::DerivedWeights::new();
        let levers = ops::ModelLevers::defaults();
        let stats = ops::ModelStats::new();
        let ptr = gpu.alloc(3 * 2049 * 4 + 1024).unwrap();
        let meta = AttnMetadataDev {
            positions: ptr,
            positions_h: ptr,
            positions_w: ptr,
            slot: ptr.offset(256),
            seq_len: ptr.offset(512),
            block_table: ptr.offset(768),
            max_blocks_per_seq: 2049,
            num_seqs: 3,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let ctx = ForwardContext {
            buffers: &arena,
            gpu,
            config,
            dispatch: &dispatch,
            derived: &derived,
            levers: &levers,
            stats: &stats,
            ssm_batch: None,
            attn_metadata: Some(meta),
            profile: false,
            comm: None,
            graph_capture: false,
            gdn_exact_replay: false,
            token_ids: None,
            host_token_ids: None,
            routed_lora_layers: None,
            midchunk_capture: None,
            moe_lora_route: crate::layer::MoeLoraRoute::Skip,
        };
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
            2050,
            gpu,
        )
        .unwrap();
        cache
            .attach_sparse_index(SparseIndexCacheConfig::bf16(4, 128), gpu)
            .unwrap();
        let before = gpu.launch_count();
        layer
            .prefill_mla_kv_only_impl(arena.hidden_states(), 3, &mut cache, meta.slot, &ctx, 0)
            .unwrap();
        let launches = gpu.launches_snapshot();
        let index: Vec<_> = launches[before..]
            .iter()
            .filter(|x| (801..=804).contains(&x.func))
            .map(|x| (x.func, x.grid[0]))
            .collect();
        assert_eq!(
            index,
            vec![(801, 3), (802, 3)],
            "KV-only prompt/repair must populate raw and pooled index"
        );
        for positions in [[2046, 2048, 2049], [32767, 32768, 32769]] {
            let c = MultiSeqCtx::new(
                layer,
                &ctx,
                arena.hidden_states(),
                arena.residual(),
                3,
                &positions,
                16,
                0,
            );
            let before = gpu.launch_count();
            assert!(layer.ms_mla_decode(&c, &mut cache, meta).is_err());
            assert_eq!(
                gpu.launch_count(),
                before,
                "bad row extents must refuse before kernels"
            );
        }
        let graph_ctx = ForwardContext {
            graph_capture: true,
            midchunk_capture: None,
            ..ctx
        };
        let positions = [2046, 2047, 2048];
        let c = MultiSeqCtx::new(
            layer,
            &graph_ctx,
            arena.hidden_states(),
            arena.residual(),
            3,
            &positions,
            16,
            0,
        );
        let before = gpu.launch_count();
        assert!(layer.ms_mla_decode(&c, &mut cache, meta).is_err());
        assert_eq!(gpu.launch_count(), before);
        if std::env::var("ATLAS_GLM_K3_MLA_QUERY_BATCHM").as_deref() == Ok("1") {
            // Exercise the full FP32 residual span: placing normed in its
            // upper half must be rejected before any projection or cache
            // launch. A BF16-sized residual liveness check would miss this
            // overlap.
            let positions = [2046, 2047, 2048];
            let c = MultiSeqCtx::new(
                layer,
                &ctx,
                arena.hidden_states(),
                arena.residual(),
                3,
                &positions,
                16,
                0,
            );
            let residual_bytes = arena.sizes().ssm_conv_out_f32;
            let input_bytes = 3 * 4096 * 2;
            assert!(residual_bytes >= input_bytes);
            let mut rejected = c;
            rejected.residual = arena.ssm_conv_out_f32();
            rejected.normed = arena
                .ssm_conv_out_f32()
                .offset(residual_bytes - input_bytes);
            let before = gpu.launch_count();
            assert!(layer.ms_mla_decode(&rejected, &mut cache, meta).is_err());
            assert_eq!(
                gpu.launch_count(),
                before,
                "FP32 residual upper-half alias must fail before query work"
            );
        }
        for positions in [
            [2045, 2046, 2047], // all short: query flag must fall back cleanly
            [2046, 2047, 2048], // exact dense/sparse boundary
            [32764, 32765, 32766],
        ] {
            let c = MultiSeqCtx::new(
                layer,
                &ctx,
                arena.hidden_states(),
                arena.residual(),
                3,
                &positions,
                16,
                0,
            );
            if std::env::var("ATLAS_GLM_K3_MLA_O_BATCHM").as_deref() == Ok("1") {
                let compare = std::env::var("ATLAS_GLM_K3_MLA_O_COMPARE").as_deref() == Ok("1");
                for bad in 0..if compare { 3 } else { 2 } {
                    let mut rejected = MultiSeqCtx::new(
                        layer,
                        &ctx,
                        arena.hidden_states(),
                        arena.residual(),
                        3,
                        &positions,
                        16,
                        0,
                    );
                    if bad == 0 {
                        rejected.nq = 64;
                    } else if bad == 1 {
                        rejected.normed = arena.ssm_qkvz().offset(512);
                    } else {
                        rejected.normed = arena.attn_output();
                    }
                    let count = gpu.launch_count();
                    assert!(layer.ms_mla_decode(&rejected, &mut cache, meta).is_err());
                    assert_eq!(
                        gpu.launch_count(),
                        count,
                        "bad O plan must reject before cache/index writes"
                    );
                }
            }
            let before = gpu.launch_count();
            let args_before = gpu.1.lock().unwrap().len();
            let alloc_before = gpu.alloc_count();
            let copies_before = gpu.d2h_blocking_count();
            layer.ms_mla_decode(&c, &mut cache, meta).unwrap();
            assert_eq!(gpu.alloc_count(), alloc_before, "no forward allocation");
            let args = gpu.1.lock().unwrap();
            let calls = &args[args_before..];
            let batched = std::env::var("ATLAS_GLM_K3_MLA_O_BATCHM").as_deref() == Ok("1");
            let compare = std::env::var("ATLAS_GLM_K3_MLA_O_COMPARE").as_deref() == Ok("1");
            let query_batched =
                std::env::var("ATLAS_GLM_K3_MLA_QUERY_BATCHM").as_deref() == Ok("1");
            let query_compare =
                std::env::var("ATLAS_GLM_K3_MLA_QUERY_COMPARE").as_deref() == Ok("1");
            let query_active = query_batched && positions.iter().any(|&pos| pos >= 2048);
            assert_eq!(
                gpu.d2h_blocking_count() - copies_before,
                (if compare { 2 } else { 0 }) + (if query_compare && query_active { 8 } else { 0 }),
                "only diagnostic modes read full comparison buffers"
            );
            let mla = layer.mla.as_ref().unwrap();
            let q_a_weight = mla.wq_a.weight.0.to_ne_bytes().to_vec();
            let q_b_weight = mla.wq_b.weight.0.to_ne_bytes().to_vec();
            let index_q_weight = mla.glm_indexer.as_ref().unwrap().wq_b.weight;
            let index_q_weight = index_q_weight.0.to_ne_bytes().to_vec();
            let query_batches: Vec<_> = calls
                .iter()
                .enumerate()
                .filter(|(_, (k, a))| {
                    *k == 807
                        && (a[1] == q_a_weight || a[1] == q_b_weight || a[1] == index_q_weight)
                })
                .collect();
            assert_eq!(
                query_batches.len(),
                if query_active { 3 } else { 0 },
                "query batchm must be disabled for short or opt-out calls"
            );
            if query_active {
                let first_cache_write = calls
                    .iter()
                    .position(|(k, _)| *k == 820)
                    .expect("scalar MLA cache assembly must be recorded");
                for (index, call) in query_batches.iter().enumerate() {
                    let launch_index = call.0;
                    let args = &call.1.1;
                    assert!(
                        launch_index < first_cache_write,
                        "all query projections must precede the first cache write"
                    );
                    let (weight, output, dims) = match index {
                        0 => (
                            q_a_weight.clone(),
                            arena.ssm_ba(),
                            [3_u32, 1536, 4096, 1536],
                        ),
                        1 => (
                            q_b_weight.clone(),
                            arena.ssm_deinterleaved(),
                            [3_u32, 8192, 1536, 8192],
                        ),
                        2 => (
                            index_q_weight.clone(),
                            arena.ssm_deinterleaved().offset(3 * 8192 * 2),
                            [3_u32, 4096, 1536, 4096],
                        ),
                        _ => unreachable!(),
                    };
                    assert_eq!(args[1], weight);
                    assert_eq!(args[2], output.0.to_ne_bytes().to_vec());
                    assert_eq!(&args[3..], &dims.map(|value| value.to_ne_bytes().to_vec()));
                }
                let scalar_for = |weight: &[u8]| {
                    calls
                        .iter()
                        .filter(|(k, a)| *k == 806 && a[1] == weight)
                        .count()
                };
                assert_eq!(scalar_for(&q_a_weight), if query_compare { 3 } else { 0 });
                assert_eq!(scalar_for(&q_b_weight), if query_compare { 3 } else { 0 });
                assert_eq!(
                    scalar_for(&index_q_weight),
                    if query_compare { 3 } else { 0 }
                );

                // Selection must consume the retained per-row index-Q slice;
                // it must not fall back to the scalar base projection or
                // overwrite another row's tail before its logits launch.
                let index_logits: Vec<_> = calls
                    .iter()
                    .enumerate()
                    .filter(|(_, (k, _))| *k == 812)
                    .collect();
                let expected_index_logits = positions.iter().filter(|&&pos| pos >= 2048).count();
                assert_eq!(index_logits.len(), expected_index_logits);
                let mut row = 0;
                for (position, (_, args)) in index_logits {
                    while positions[row] < 2048 {
                        row += 1;
                    }
                    assert_eq!(
                        args[0],
                        arena
                            .ssm_deinterleaved()
                            .offset(3 * 8192 * 2 + row * 4096 * 2)
                            .0
                            .to_ne_bytes()
                            .to_vec(),
                        "index logits must use row {row}'s retained index-Q"
                    );
                    assert!(position > first_cache_write);
                    row += 1;
                }
            }
            let o_weight = layer
                .mla
                .as_ref()
                .unwrap()
                .wo
                .weight
                .0
                .to_ne_bytes()
                .to_vec();
            let o_calls: Vec<_> = calls
                .iter()
                .enumerate()
                .filter(|(_, (k, a))| (*k == 806 || *k == 807) && a[1] == o_weight)
                .collect();
            assert_eq!(
                o_calls.len(),
                if compare {
                    4
                } else if batched {
                    1
                } else {
                    3
                }
            );
            let extracted: Vec<_> = calls
                .iter()
                .enumerate()
                .filter(|(_, (k, a))| *k == 808 && a[3] == 256_u32.to_ne_bytes())
                .collect();
            assert_eq!(extracted.len(), 3);
            let prefix_writes: Vec<_> = calls
                .iter()
                .filter(|(k, a)| {
                    *k == 806
                        && (a[2] == arena.ssm_qkvz().0.to_ne_bytes()
                            || a[2] == arena.ssm_qkvz().offset(256).0.to_ne_bytes())
                        && a[3] == 128_u32.to_ne_bytes()
                })
                .collect();
            assert_eq!(
                prefix_writes.len(),
                6,
                "three serial key/gate writes stay in512-byte prefix"
            );
            for (i, (_, (_, a))) in extracted.iter().enumerate() {
                let offset = if batched { 512 + i * 8192 * 2 } else { 0 };
                assert_eq!(a[2], arena.ssm_qkvz().offset(offset).0.to_ne_bytes());
            }
            if compare {
                let scalar: Vec<_> = o_calls.iter().filter(|(_, (k, _))| *k == 806).collect();
                assert_eq!(scalar.len(), 3);
                for i in 0..3 {
                    assert_eq!(
                        scalar[i].0,
                        extracted[i].0 + 1,
                        "scalar baseline must immediately follow its V extract"
                    );
                    let a = &scalar[i].1.1;
                    assert_eq!(
                        a[0],
                        arena.ssm_qkvz().offset(512 + i * 8192 * 2).0.to_ne_bytes()
                    );
                    assert_eq!(
                        a[2],
                        arena.moe_output().offset(i * 4096 * 2).0.to_ne_bytes()
                    );
                }
            }
            if batched {
                let batched_o = o_calls.iter().find(|(_, (k, _))| *k == 807).unwrap();
                assert!(
                    batched_o.0 > extracted[2].0,
                    "O must follow third V extraction"
                );
                assert_eq!(calls.last().unwrap().0, 807);
                let a = &batched_o.1.1;
                assert_eq!(a[0], arena.ssm_qkvz().offset(512).0.to_ne_bytes());
                let expected_output = if compare {
                    arena.attn_output()
                } else {
                    arena.moe_output()
                };
                assert_eq!(a[2], expected_output.0.to_ne_bytes());
                assert_eq!(
                    &a[3..],
                    &[3_u32, 4096, 8192, 4096].map(|v| v.to_ne_bytes().to_vec())
                );
            }
            drop(args);
            let launches = gpu.launches_snapshot();
            let index: Vec<_> = launches[before..]
                .iter()
                .filter(|x| (801..=804).contains(&x.func) || x.func == sparse_kernel)
                .map(|x| x.func)
                .collect();
            let mut expected = vec![];
            for pos in positions {
                expected.extend([801, 802]);
                if pos + 1 > 2048 {
                    expected.extend([803, sparse_kernel]);
                }
            }
            assert_eq!(
                index, expected,
                "causal rows must maintain index before selecting sparse attention"
            );
        }
    });
}
