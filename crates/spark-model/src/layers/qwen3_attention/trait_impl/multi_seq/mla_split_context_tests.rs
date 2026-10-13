// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
#[test]
fn actual_prompt_index_and_causal_verify_dispatch_split() {
    const CHILD: &str = "ATLAS_TEST_SPLIT_K3";
    if std::env::var_os(CHILD).is_none() {
        let name = concat!(
            module_path!(),
            "::actual_prompt_index_and_causal_verify_dispatch_split"
        );
        for o in ["0", "1"] {
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
                .env("ATLAS_GLM_SPARSE_DECODE_TC", "1")
                .env("ATLAS_GLM_SPARSE_DECODE_SPLIT", "1")
                .env("ATLAS_GLM_K3_MLA_O_BATCHM", o)
                .output()
                .unwrap();
            assert!(String::from_utf8_lossy(&out.stdout).contains("running 1 test"));
            assert!(
                out.status.success(),
                "{}\n{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
        }
        return;
    }
    fixture(|gpu, source, layer| {
        let mut config = source.clone();
        config.moe_intermediate_size = 2048;
        config.num_experts = 288;
        config.num_experts_per_tok = 8;
        let arena = BufferArena::new(&config, 32, 32768, 16, 1, gpu).unwrap();
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
            config: &config,
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
        for positions in [[2046, 2047, 2048], [32764, 32765, 32766]] {
            let mut c = MultiSeqCtx::new(
                layer,
                &ctx,
                arena.hidden_states(),
                arena.residual(),
                3,
                &positions,
                16,
                0,
            );
            let saved = c.normed;
            c.normed = arena.expert_gate_out().offset(32);
            let before = gpu.launch_count();
            assert!(layer.ms_mla_decode(&c, &mut cache, meta).is_err());
            assert_eq!(
                before,
                gpu.launch_count(),
                "scratch alias rejects before cache writes"
            );
            c.normed = saved;
            let args_before = gpu.1.lock().unwrap().len();
            let alloc_before = gpu.alloc_count();
            layer.ms_mla_decode(&c, &mut cache, meta).unwrap();
            assert_eq!(gpu.alloc_count(), alloc_before);
            let records = gpu.1.lock().unwrap();
            let calls = &records[args_before..];
            let selected: Vec<_> = calls
                .iter()
                .filter(|(k, _)| matches!(k, 803 | 804 | 805 | 809 | 810))
                .collect();
            let count = if positions[0] > 2048 { 3 } else { 1 };
            assert_eq!(selected.len(), count * 3);
            for triple in selected.chunks_exact(3) {
                assert_eq!([triple[0].0, triple[1].0, triple[2].0], [803, 809, 810]);
                let split = &triple[1].1;
                let merge = &triple[2].1;
                assert_eq!(split[0], arena.expert_up_out().0.to_ne_bytes());
                assert_eq!(split[3], arena.qkv_output().0.to_ne_bytes());
                assert_eq!(split[4], arena.expert_gate_out().0.to_ne_bytes());
                assert_eq!(split[6], 1u32.to_ne_bytes());
                assert_eq!(split[13], 8u32.to_ne_bytes());
                assert_eq!(
                    split[12],
                    arena.expert_gate_out().offset(524288).0.to_ne_bytes()
                );
                assert_eq!(merge[2], arena.attn_output().0.to_ne_bytes());
                assert_eq!(
                    merge[3],
                    arena.expert_gate_out().offset(525312).0.to_ne_bytes()
                );
            }
            let extracts: Vec<_> = calls
                .iter()
                .filter(|(k, a)| *k == 808 && a[3] == 256u32.to_ne_bytes())
                .collect();
            assert_eq!(extracts.len(), 3);
            let staged = std::env::var("ATLAS_GLM_K3_MLA_O_BATCHM").as_deref() == Ok("1");
            for (i, (_, a)) in extracts.iter().enumerate() {
                assert_eq!(
                    a[2],
                    arena
                        .ssm_qkvz()
                        .offset(if staged { 512 + i * 16384 } else { 0 })
                        .0
                        .to_ne_bytes()
                );
            }
        }
    });
}
