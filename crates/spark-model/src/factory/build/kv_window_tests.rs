// SPDX-License-Identifier: AGPL-3.0-only

//! Which configs take window-bounded KV, and the budget split.

use super::{Exclusions, plan, primary_blocks};
use atlas_core::config::ModelConfig;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype};

const NONE: Exclusions = Exclusions {
    high_speed_swap: false,
    disk_tier: false,
    multi_rank: false,
    glm_plan: false,
};

/// Inkling 2 Mini's 48 visits: 8 global (5, 11 x3, 17 x3, 23), 40 local.
fn inkling() -> (ModelConfig, KvCacheConfig) {
    let globals = [5, 11, 23, 29, 35, 41, 47, 17];
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "inkling2_loop".into();
    config.sliding_window = 1024;
    config.sliding_attention_layers = (0..48).filter(|v| !globals.contains(v)).collect();
    let kv = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 8,
        head_dim: 128,
        num_layers: 48,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    (config, kv)
}

#[test]
fn inkling_keeps_forty_window_layers_in_a_bounded_pool() {
    let (config, kv) = inkling();
    let plan = plan(&config, &kv, NONE, 32, 4128).unwrap();
    assert_eq!(plan.layers.len(), 40);
    assert_eq!((plan.window, plan.pins), (1024, 32));
    // 40 layers x K+V x 16 tokens x 8 heads x 128 x 2 bytes
    assert_eq!(plan.local_block_bytes(&kv), 40 * 2 * 16 * 8 * 128 * 2);
    assert_eq!(plan.primary_block_bytes(&kv), 8 * 2 * 16 * 8 * 128 * 2);
}

#[test]
fn anything_that_assumes_one_block_id_per_row_keeps_full_pools() {
    let (config, kv) = inkling();
    for excluded in [
        Exclusions {
            high_speed_swap: true,
            ..NONE
        },
        Exclusions {
            disk_tier: true,
            ..NONE
        },
        Exclusions {
            multi_rank: true,
            ..NONE
        },
        Exclusions {
            glm_plan: true,
            ..NONE
        },
    ] {
        assert!(
            plan(&config, &kv, excluded, 8, 4104).is_none(),
            "{excluded:?}"
        );
    }
    let mut other = config;
    other.model_type = "ricursive".into();
    assert!(plan(&other, &kv, NONE, 8, 4104).is_none());
}

#[test]
fn the_budget_left_after_the_window_pools_buys_primary_blocks() {
    let (config, kv) = inkling();
    let plan = plan(&config, &kv, NONE, 8, 4104).unwrap();
    let window = plan.blocks * plan.local_block_bytes(&kv);
    let budget = window + 1000 * (plan.primary_block_bytes(&kv) + 4);
    assert_eq!(primary_blocks(&plan, &kv, budget, 0).unwrap(), 1000);
    assert!(primary_blocks(&plan, &kv, window, 0).is_err());
}
