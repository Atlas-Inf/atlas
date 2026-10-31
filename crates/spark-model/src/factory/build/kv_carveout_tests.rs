// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use spark_runtime::gpu::carveout::CarveoutArena;
use spark_runtime::kv_cache::KvCacheDtype;

const MIB: usize = 1 << 20;
/// The GB10 DISPLAY_FRM carveout as the driver reports it.
const GB10: usize = 2046 * MIB;

fn config(layers: usize, dtype: KvCacheDtype) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: layers,
        dtype,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

/// SparkGLM's TP2 serving cache: 11 fp8_g128 latent layers, the BF16
/// semantic index, and slot-mapped tails for 4 sequences at 8192-token
/// prefill chunks (`glm::cache_plan`).
fn sparkglm(config: &KvCacheConfig) -> KvShape<'_> {
    KvShape {
        auxiliary_block_bytes: 0,
        config,
        v_aliases_k: true,
        index: Some(SparseIndexCacheConfig::bf16(4, 128)),
        tail_slots: Some(TailSlotPlan {
            lag_blocks: 8192usize.div_ceil(16) + 2,
            sequences: 5,
        }),
    }
}

fn carveout_footprint(shape: KvShape, blocks: usize, placement: &KvPlacement) -> usize {
    shape
        .buffers(blocks)
        .iter()
        .filter(|&&(buffer, _)| placement.contains(buffer))
        .map(|&(_, bytes)| CarveoutArena::footprint(bytes))
        .sum()
}

#[test]
fn sparkglm_gains_about_a_fifth_more_blocks_at_no_system_memory_cost() {
    let cfg = config(11, KvCacheDtype::Fp8G128);
    let shape = sparkglm(&cfg);
    let budgeted = 95_535;
    let (n, placement) = plan(shape, budgeted, GB10);
    // 2046 MiB over ~104 KB per block is ~20.6K blocks. The default order
    // places latent pools only (`ATLAS_KV_CARVEOUT_ORDER=latent`), so whole
    // ~0.9 GB K pools strand up to one pool's worth: ~18.5K blocks (+19.4%,
    // as on the pair before the KV shard).
    assert!(n >= budgeted + 18_000, "{n}");
    assert!(n <= budgeted + GB10 / 104_192 + 1, "{n}");
    assert!(
        shape.system_bytes(n, &placement) <= shape.system_bytes(budgeted, &KvPlacement::default())
    );
    assert!(carveout_footprint(shape, n, &placement) <= GB10);
}

#[test]
fn any_agreed_count_up_to_the_planned_one_still_fits() {
    let cfg = config(11, KvCacheDtype::Fp8G128);
    let shape = sparkglm(&cfg);
    let budgeted = 95_535;
    let (n, placement) = plan(shape, budgeted, GB10);
    let system = shape.system_bytes(budgeted, &KvPlacement::default());
    for agreed in [budgeted - 5_000, budgeted, (budgeted + n) / 2, n - 1, n] {
        assert!(shape.system_bytes(agreed, &placement) <= system, "{agreed}");
        assert!(
            carveout_footprint(shape, agreed, &placement) <= GB10,
            "{agreed}"
        );
    }
}

#[test]
fn without_a_carveout_the_budgeted_pool_is_unchanged() {
    let cfg = config(11, KvCacheDtype::Fp8G128);
    let (n, placement) = plan(sparkglm(&cfg), 95_535, 0);
    assert_eq!(n, 95_535);
    assert!(placement.is_empty());
}

#[test]
fn a_generic_cache_with_separate_k_and_v_grows_too() {
    let cfg = config(4, KvCacheDtype::Bf16);
    let shape = KvShape {
        auxiliary_block_bytes: 0,
        config: &cfg,
        v_aliases_k: false,
        index: None,
        tail_slots: None,
    };
    let (n, placement) = plan(shape, 1_000, 64 * MIB);
    assert!(n > 1_000);
    assert!(
        shape.system_bytes(n, &placement) <= shape.system_bytes(1_000, &KvPlacement::default())
    );
}

#[test]
fn qwen_carveout_growth_reserves_indexer_state_for_every_added_token() {
    let mut cfg = config(12, KvCacheDtype::Bf16);
    cfg.head_dim = 256;
    let bare = KvShape {
        config: &cfg,
        v_aliases_k: false,
        index: None,
        tail_slots: None,
        auxiliary_block_bytes: 0,
    };
    let shape = bare.with_per_token_aux(768).unwrap();
    let budgeted = 208_676;
    let (n, placement) = plan(shape, budgeted, GB10);
    assert!(n > budgeted);
    let actual = |blocks, placement: &KvPlacement| {
        bare.system_bytes(blocks, placement) + blocks * cfg.block_size * 768
    };
    let budget = actual(budgeted, &KvPlacement::default());
    assert!(actual(n, &placement) <= budget);
    assert!(carveout_footprint(shape, n, &placement) <= GB10);
    let over = KvPlacement::plan(&shape.buffers(n + 1), GB10);
    assert!(actual(n + 1, &over) > budget);
    let (old_n, old_placement) = plan(bare, budgeted, GB10);
    assert!(actual(old_n, &old_placement) > budget);
    assert!(bare.with_per_token_aux(usize::MAX).is_err());
}
