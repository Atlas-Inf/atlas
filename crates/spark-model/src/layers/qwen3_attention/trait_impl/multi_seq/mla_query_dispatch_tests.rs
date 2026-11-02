// SPDX-License-Identifier: AGPL-3.0-only

// Focused child module for mla_long_context_tests.rs.
//
// Required inclusion in mla_long_context_tests.rs:
//   #[path = "mla_query_dispatch_tests.rs"]
//   mod query_dispatch_tests;
//
// This deliberately runs with O staging disabled.  It isolates the query
// stage's three batchm launches and covers the guards that an O-enabled test
// can otherwise satisfy before reaching query planning.
use super::*;

#[test]
fn query_batch_dispatch_boundary_fallback_and_guards() {
    const CHILD: &str = "ATLAS_TEST_GLM_K3_QUERY_DISPATCH";
    if std::env::var_os(CHILD).is_none() {
        let name = concat!(
            module_path!(),
            "::query_batch_dispatch_boundary_fallback_and_guards"
        );
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.args(["--exact", name.split_once("::").unwrap().1, "--nocapture"]);
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("ATLAS_") {
                cmd.env_remove(key);
            }
        }
        let output = cmd
            .env(CHILD, "1")
            .env("ATLAS_GLM_MTP_LONG_CONTEXT", "1")
            .env("ATLAS_GLM_MTP_REPAIR", "1")
            .env("ATLAS_GLM_SPARSE_DECODE_TC", "0")
            .env("ATLAS_GLM_K3_MLA_O_BATCHM", "0")
            .env("ATLAS_GLM_K3_MLA_O_COMPARE", "0")
            .env("ATLAS_GLM_K3_MLA_QUERY_BATCHM", "1")
            .env("ATLAS_GLM_K3_MLA_QUERY_COMPARE", "0")
            .output()
            .unwrap();
        assert!(String::from_utf8_lossy(&output.stdout).contains("running 1 test"));
        assert!(
            output.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        return;
    }

    fixture(|gpu, config, layer| {
        let arena = BufferArena::new(config, 16, 32768, 16, 1, gpu).unwrap();
        let mut dispatch = ops::GemmDispatch::defaults();
        dispatch.cublas_gemm = false;
        let derived = ops::DerivedWeights::new();
        let levers = ops::ModelLevers::defaults();
        let stats = ops::ModelStats::new();
        let metadata = gpu.alloc(3 * 2049 * 4 + 1024).unwrap();
        let meta = AttnMetadataDev {
            positions: metadata,
            positions_h: metadata,
            positions_w: metadata,
            slot: metadata.offset(256),
            seq_len: metadata.offset(512),
            block_table: metadata.offset(768),
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
        layer
            .prefill_mla_kv_only_impl(arena.hidden_states(), 3, &mut cache, meta.slot, &ctx, 0)
            .unwrap();

        let q_a = layer
            .mla
            .as_ref()
            .unwrap()
            .wq_a
            .weight
            .0
            .to_ne_bytes()
            .to_vec();
        let q_b = layer
            .mla
            .as_ref()
            .unwrap()
            .wq_b
            .weight
            .0
            .to_ne_bytes()
            .to_vec();
        let index_q = layer
            .mla
            .as_ref()
            .unwrap()
            .glm_indexer
            .as_ref()
            .unwrap()
            .wq_b
            .weight
            .0
            .to_ne_bytes()
            .to_vec();
        let args_for = |calls: &[(u64, Vec<Vec<u8>>)], kernel: u64, weight: &[u8]| {
            calls
                .iter()
                .filter(|(func, args)| *func == kernel && args.get(1).is_some_and(|x| x == weight))
                .count()
        };

        // Position 2048 is both the first sparse row (seq_len > topk) and a
        // fresh four-token pool. All three stateless projections must finish
        // before row 0 can mutate the semantic cache.
        let boundary = [2047, 2048, 2049];
        let c = MultiSeqCtx::new(
            layer,
            &ctx,
            arena.hidden_states(),
            arena.residual(),
            3,
            &boundary,
            16,
            0,
        );
        let launch_before = gpu.launch_count();
        let record_before = gpu.1.lock().unwrap().len();
        layer.ms_mla_decode(&c, &mut cache, meta).unwrap();
        let calls = gpu.1.lock().unwrap()[record_before..].to_vec();
        let query_batches: Vec<_> = calls
            .iter()
            .enumerate()
            .filter(|(_, (func, args))| {
                *func == 807
                    && [q_a.as_slice(), q_b.as_slice(), index_q.as_slice()]
                        .iter()
                        .any(|weight| args.get(1).is_some_and(|x| x == *weight))
            })
            .collect();
        assert_eq!(query_batches.len(), 3);
        let first_cache = calls
            .iter()
            .position(|(func, _)| *func == 820)
            .expect("cache assembly must be recorded");
        assert!(query_batches.iter().all(|(i, _)| *i < first_cache));
        assert_eq!(args_for(&calls, 807, &q_a), 1);
        assert_eq!(args_for(&calls, 807, &q_b), 1);
        assert_eq!(args_for(&calls, 807, &index_q), 1);

        // The retained index-Q must follow the query layout row by row. The
        // first row remains dense, while rows 1 and 2 select sparse attention
        // only after their own key/gate update and top-k expansion.
        let logits: Vec<_> = calls.iter().filter(|(func, _)| *func == 812).collect();
        assert_eq!(logits.len(), 2);
        for (row, (_, args)) in logits.iter().enumerate() {
            let row = row + 1;
            assert_eq!(
                args[0],
                arena
                    .ssm_deinterleaved()
                    .offset(3 * 8192 * 2 + row * 4096 * 2)
                    .0
                    .to_ne_bytes()
                    .to_vec()
            );
        }
        let causal: Vec<_> = gpu.launches_snapshot()[launch_before..]
            .iter()
            .filter(|launch| matches!(launch.func, 801 | 802 | 803 | 804 | 812))
            .map(|launch| launch.func)
            .collect();
        assert_eq!(
            causal,
            vec![801, 802, 801, 802, 812, 803, 804, 801, 802, 812, 803, 804]
        );

        // Fully short K3 must keep the scalar path. The query flag is enabled
        // globally, but no batchm launch is permitted when every row is <=topk.
        let short = [2045, 2046, 2047];
        let c = MultiSeqCtx::new(
            layer,
            &ctx,
            arena.hidden_states(),
            arena.residual(),
            3,
            &short,
            16,
            0,
        );
        let record_before = gpu.1.lock().unwrap().len();
        layer.ms_mla_decode(&c, &mut cache, meta).unwrap();
        let calls = gpu.1.lock().unwrap()[record_before..].to_vec();
        assert_eq!(args_for(&calls, 807, &q_a), 0);
        assert_eq!(args_for(&calls, 807, &q_b), 0);
        assert_eq!(args_for(&calls, 807, &index_q), 0);
        assert_eq!(args_for(&calls, 806, &q_a), 3);
        assert_eq!(args_for(&calls, 806, &q_b), 3);
        assert_eq!(args_for(&calls, 806, &index_q), 0);
        assert!(
            !calls
                .iter()
                .any(|(func, _)| matches!(*func, 812 | 803 | 804))
        );

        // Query-only guards must fail before the first cache write. These
        // cases are intentionally run with O staging disabled, so an O plan
        // cannot mask a query geometry or liveness failure.
        for bad in ["nq", "normed_alias", "latent_alias"] {
            let positions = [2047, 2048, 2049];
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
            match bad {
                "nq" => rejected.nq = 64,
                "normed_alias" => rejected.normed = arena.ssm_qkvz().offset(512),
                "latent_alias" => rejected.normed = arena.ssm_ba(),
                _ => unreachable!(),
            }
            let before = gpu.launch_count();
            assert!(
                layer.ms_mla_decode(&rejected, &mut cache, meta).is_err(),
                "{bad}"
            );
            assert_eq!(
                gpu.launch_count(),
                before,
                "{bad} must reject before cache writes"
            );
        }
    });
}
