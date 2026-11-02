// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn config() -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "glm5_next".into();
    c.tp_world_size = 2;
    c.ep_world_size = 2;
    c.num_attention_heads = 32;
    c
}

fn policy(c: &ModelConfig) -> BuildPolicy<'_> {
    BuildPolicy {
        config: c,
        self_speculative: false,
        context: 32768,
        block_size: 16,
        kv_dtype: KvCacheDtype::Bf16,
        layer_dtypes: &[],
        dflash: true,
    }
}

#[test]
fn glm_sparse_decode_build_accepts_only_the_long_context_dflash_lane() {
    let c = config();
    assert!(policy(&c).validate(true, true).is_ok());
    let mut g128 = policy(&c);
    g128.kv_dtype = KvCacheDtype::Fp8G128;
    assert!(g128.validate(true, true).is_ok());
    for variant in 0..9 {
        let mut p = policy(&c);
        let fp8 = [KvCacheDtype::Fp8];
        match variant {
            0 => p.dflash = false,
            1 => p.self_speculative = true,
            2 => p.context = 2048,
            3 => p.block_size = 32,
            4 => p.kv_dtype = KvCacheDtype::Fp8,
            5 => p.layer_dtypes = &fp8,
            6 => {
                assert!(p.validate(false, true).is_err(), "lane disabled");
                continue;
            }
            7 => {
                assert!(p.validate(true, false).is_err(), "long context disabled");
                continue;
            }
            _ => p.context = crate::speculative::glm_repair_policy::max_long_context() + 1,
        }
        assert!(p.validate(true, true).is_err(), "variant {variant}");
    }
}

#[test]
fn glm_sparse_decode_build_rejects_wrong_rank_geometry() {
    for variant in 0..4 {
        let mut c = config();
        match variant {
            0 => c.model_type = "other".into(),
            1 => c.tp_world_size = 1,
            2 => c.ep_world_size = 1,
            _ => c.num_attention_heads = 64,
        }
        assert!(policy(&c).validate(true, true).is_err());
    }
}

#[test]
fn larger_served_context_is_capped_to_the_verifier_domain() {
    let max = crate::speculative::glm_repair_policy::MAX_LONG_CONTEXT;
    assert_eq!(repair_context(2048), 2048);
    assert_eq!(repair_context(max), max);
    assert_eq!(repair_context(max + 4096), max);
}
