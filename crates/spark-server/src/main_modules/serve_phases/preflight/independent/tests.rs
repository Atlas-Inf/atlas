// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use clap::Parser;

#[path = "long_context_tests.rs"]
mod long_context;

// Supply only the I/O boundary; all topology/admission/accounting is production.
fn prepare_reserve(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
    free_mem: usize,
) -> Result<(Option<Topology>, ReservePreflight)> {
    let (_, _, topology, reserve) = super::prepare_reserve(args, config, || Ok(((), free_mem)))?;
    Ok((topology, reserve))
}

fn args(cap: usize, rank: usize) -> cli::ServeArgs {
    let cli::Command::Serve(mut args) = cli::Cli::try_parse_from([
        "spark",
        "serve",
        "unused",
        "--world-size",
        "2",
        "--tp-size",
        "2",
        "--ep-size",
        "2",
        "--kv-cache-dtype",
        "bf16",
        "--max-seq-len",
        "2048",
        "--max-prefill-tokens",
        "1024",
        "--swap-space-gb",
        "0",
    ])
    .unwrap()
    .command
    else {
        panic!("serve args")
    };
    args.max_batch_size = cap;
    args.max_num_seqs = cap;
    args.rank = rank;
    args.num_drafts = Some(4);
    args
}

fn config() -> ModelConfig {
    atlas_core::config::parse_config(&serde_json::json!({
        "model_type":"glm5_next", "text_config": {
            "model_type":"glm5_next_text", "hidden_size":4096,
            "num_hidden_layers":45, "num_nextn_predict_layers":1,
            "intermediate_size":12288, "vocab_size":154880,
            "max_position_embeddings":1048576, "rms_norm_eps":1e-5,
            "num_attention_heads":64, "num_key_value_heads":64,
            "kv_lora_rank":512, "q_lora_rank":1536,
            "qk_nope_head_dim":256, "qk_rope_head_dim":0, "v_head_dim":256,
            "linear_attn_config":{"num_heads":64,"head_dim":128,
                "short_conv_kernel_size":4,"gate_lower_bound":-5.0},
            "n_routed_experts":288, "num_experts_per_tok":8,
            "moe_intermediate_size":2048,"n_shared_experts":1,
            "norm_topk_prob":true,"scoring_func":"sigmoid","topk_method":"noaux_tc",
            "routed_scaling_factor":2.5,"first_k_dense_replace":3,
            "layer_types":(0..45).map(|i|if (i+1)%4==0 {"deepseek_sparse_attention"} else {"linear_attention"}).collect::<Vec<_>>(),
            "hc_mult":4,"hc_sinkhorn_iters":20,"hc_eps":1e-6,
            "index_n_heads":32,"index_head_dim":128,"index_topk":2048,
            "index_kpool":4,"index_kpool_compress":true,
            "index_kpool_always_select_tail":true,"indexer_types":vec!["full";45]
        }, "quantization_config":{"quant_method":"modelopt","quant_algo":"NVFP4"}
    }).to_string()).unwrap()
}

fn isolated(name: &str, overrides: &[(&str, &str)]) -> bool {
    const CHILD: &str = "ATLAS_SERVER_LONG_CONTEXT_TEST_CHILD";
    if std::env::var_os(CHILD).is_none() {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        for (key, _) in std::env::vars_os() {
            if key.to_string_lossy().starts_with("ATLAS_") {
                child.env_remove(key);
            }
        }
        let result = child
            .arg("--exact")
            .arg(format!(
                "main_modules::serve_phases::preflight::independent::tests::{name}"
            ))
            .arg("--nocapture")
            .env(CHILD, "1")
            .env("ATLAS_EP_PROTOCOL", "v2")
            .envs(overrides.iter().copied())
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}\n{}",
            String::from_utf8_lossy(&result.stdout),
            String::from_utf8_lossy(&result.stderr)
        );
        assert!(
            String::from_utf8_lossy(&result.stdout).contains("1 passed"),
            "child did not run requested test"
        );
        return true;
    }
    false
}

fn assert_allocation(
    a: &cli::ServeArgs,
    cfg: &ModelConfig,
    reserve: &ReservePreflight,
    rows: usize,
) {
    assert_eq!(cfg.linear_num_key_heads, 32);
    assert_eq!(cfg.linear_num_value_heads, 32);
    assert_eq!(cfg.num_attention_heads, 32);
    assert_eq!(cfg.num_ssm_layers(), 34);
    assert_eq!(reserve.max_batch_tokens_pre, rows);
    assert_eq!(
        reserve.resolved_prefill.as_ref().unwrap().max_batch_tokens,
        rows
    );
    assert_eq!(
        reserve.resolved_prefill.as_ref().unwrap().prefill_budget,
        1024
    );
    assert_eq!(
        reserve.buffer_arena_bytes,
        spark_runtime::buffers::BufferSizes::from_config(
            cfg,
            rows,
            a.max_seq_len,
            a.block_size,
            a.max_batch_size
        )
        .total_bytes()
    );
    let blob = cfg.num_ssm_layers() * (cfg.ssm_h_state_bytes() + cfg.ssm_conv_state_bytes());
    assert_eq!(blob, 77987840); // Actual TP-local FP32 H + conv across34 KDA layers.
    let ring =
        spark_model::ssm_reserve::decode_rollback_ring_slots(cfg.num_ssm_layers(), false).slots;
    assert_eq!(ring, 8);
    let gdn = rows.min(a.max_seq_len) * (12288 * 2 + 32 * 2 * 4 + 4096 * 2 * 2);
    assert_eq!(reserve.gdn_two_phase_bytes, gdn);
    assert_eq!(
        reserve.inference_reserve,
        (a.max_batch_size + 1 + a.ssm_cache_slots + ring * a.max_batch_size) * blob
            + gdn
            + (512 << 20)
    );
}

#[test]
fn default_glm_reserve_keeps_late_topology() {
    if isolated(
        "default_glm_reserve_keeps_late_topology",
        &[
            ("ATLAS_GLM_C4_DECODE", "1"),
            ("ATLAS_GLM_KDA_MULTI_SEQ", "1"),
            ("ATLAS_GLM_MLA_MULTI_SEQ", "1"),
        ],
    ) {
        return;
    }
    for rank in 0..2 {
        let a = args(4, rank);
        let mut cfg = config();
        let (topology, reserve) = prepare_reserve(&a, &mut cfg, 128usize << 30).unwrap();
        assert!(
            topology.is_none(),
            "only the long-context lane resolves early"
        );
        assert!(reserve.resolved_prefill.is_none());
        assert_eq!(
            cfg.linear_num_key_heads, 64,
            "heads stay global until resolution"
        );
        assert_eq!(reserve.max_batch_tokens_pre, 1024);
        // SSM pools are allocated before the KV snapshot, so the reserve is the
        // GDN two-phase scratch plus the non-speculative CUDA headroom.
        assert_eq!(
            reserve.inference_reserve,
            reserve.gdn_two_phase_bytes + (512usize << 20)
        );
        assert_eq!(
            reserve.buffer_arena_bytes,
            spark_runtime::buffers::BufferSizes::from_config(&cfg, 1024, 2048, 16, 4).total_bytes()
        );
        let topology = resolve_topology(&a, &mut cfg).unwrap();
        assert_eq!(topology.tp_rank, rank);
        assert_eq!(cfg.linear_num_key_heads, 32);
    }
}
