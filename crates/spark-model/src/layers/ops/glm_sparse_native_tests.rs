// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.hidden_size = 4096;
    c.hc_mult = 4;
    c.max_batch_tokens = 4100;
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.num_attention_heads = 32;
    c.num_key_value_heads = 32;
    c.head_dim = 256;
    c.kv_lora_rank = 512;
    c.qk_rope_head_dim = 0;
    c.index_topk = 2048;
    c.index_kpool = 4;
    c.index_n_heads = 32;
    c.index_head_dim = 128;
    c.linear_num_key_heads = 32;
    c.linear_num_value_heads = 32;
    c.linear_key_head_dim = 128;
    c.linear_value_head_dim = 128;
    c.num_experts_per_tok = 8;
    c.moe_intermediate_size = 2048;
    c.intermediate_size = 12288;
    c
}

#[test]
fn native_sparse_opt_in_is_strict() {
    for value in [None, Some("0")] {
        assert!(!dispatch::parse(value).unwrap());
    }
    assert!(dispatch::parse(Some("1")).unwrap());
    for value in ["", "true", "yes", " 1", "2"] {
        assert!(dispatch::parse(Some(value)).is_err());
    }
}

#[test]
fn native_sparse_startup_uses_real_arena_formula_and_rejects_wrong_profile() {
    let check = |c: &ModelConfig| validate_startup(c, 4100, 32768, 16, 4, KvCacheDtype::Bf16, &[]);
    check(&config()).unwrap();
    for mutate in [
        (|c: &mut ModelConfig| c.model_type = "qwen3_next".into()) as fn(&mut ModelConfig),
        |c| c.hidden_size = 8192,
        |c| c.hc_mult = 8,
        |c| c.tp_world_size = 1,
        |c| c.ep_world_size = 1,
        |c| c.num_attention_heads = 64,
        |c| c.num_key_value_heads = 64,
        |c| c.kv_lora_rank = 256,
        |c| c.qk_rope_head_dim = 64,
        |c| c.index_topk = 1024,
        |c| c.index_kpool = 8,
        |c| c.max_batch_tokens = 4096,
        |c| {
            c.linear_num_key_heads = 1;
            c.linear_num_value_heads = 1;
        },
    ] {
        let mut c = config();
        mutate(&mut c);
        assert!(check(&c).is_err());
    }
    for (rows, seq, block, active) in [
        (4096, 32768, 16, 4),
        (4100, 16384, 16, 4),
        (4100, 32768, 64, 4),
        (4100, 32768, 16, 0),
        (4100, 32768, 16, 9),
    ] {
        assert!(
            validate_startup(&config(), rows, seq, block, active, KvCacheDtype::Bf16, &[]).is_err()
        );
    }
    assert!(validate_startup(&config(), 4100, 32768, 16, 4, KvCacheDtype::Nvfp4, &[]).is_err());
    assert!(
        validate_startup(
            &config(),
            4100,
            32768,
            16,
            4,
            KvCacheDtype::Bf16,
            &[KvCacheDtype::Nvfp4]
        )
        .is_err()
    );
}

#[test]
fn native_sparse_context_limit_is_a_fallback_boundary() {
    assert_eq!(plan::MAX_CONTEXT, 32768);
    assert_eq!(qualified_context(32768), 32768);
    assert_eq!(qualified_context(36864), 32768);
    assert!(plan::admit(4096, 28672, false, false, false).is_some());
    assert!(plan::admit(4096, 32768, false, false, false).is_none());
}

#[path = "glm_sparse_native_dispatch_tests.rs"]
mod dispatch_tests;
