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
        config,
        v_aliases_k: true,
        index: Some(SparseIndexCacheConfig::bf16(4, 128)),
        tail_slots: Some(TailSlotPlan {
            lag_blocks: 8192usize.div_ceil(16) + 2,
            sequences: 5,
        }),
        latent_shard: None,
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
    // places latent pools only (`ATLAS_KV_CARVEOUT_ORDER=latent`, the
    // carveout is L2-uncached), so whole ~0.9 GB K pools strand up to one
    // pool's worth: ~18.5K blocks (+19.4%, as on the pair).
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
        config: &cfg,
        v_aliases_k: false,
        index: None,
        tail_slots: None,
        latent_shard: None,
    };
    let (n, placement) = plan(shape, 1_000, 64 * MIB);
    assert!(n > 1_000);
    assert!(
        shape.system_bytes(n, &placement) <= shape.system_bytes(1_000, &KvPlacement::default())
    );
}

/// SparkGLM's shape under `ATLAS_GLM_KV_SHARD=1`, with the ~0.3 GiB scratch
/// and 32,772-block view of `--max-seq-len 524288`.
fn sharded(config: &KvCacheConfig, rank: usize) -> KvShape<'_> {
    KvShape {
        latent_shard: Some(LatentShardSpec {
            rank,
            world: 2,
            scratch_bytes: 300 * MIB,
            view_blocks: 32_772,
            write_rows: 8192,
            lane: false,
        }),
        ..sparkglm(config)
    }
}

#[test]
fn a_latent_shard_grows_by_its_half_size_latent_pools() {
    let cfg = config(11, KvCacheDtype::Fp8G128);
    // About 1.8x the unsharded budget at the same memory setting.
    let budgeted = 170_000;
    let (r0, r1) = (sharded(&cfg, 0), sharded(&cfg, 1));
    let (n, placement) = plan(r0, budgeted, GB10);
    // Both ranks size the same buffers, so they plan the same pool.
    assert_eq!(plan(r1, budgeted, GB10), (n, placement.clone()));
    // Each local K pool (85K slots x 8448 B, ~685 MiB) is a whole buffer:
    // two fit, and freeing them buys ~29K more blocks of everything else.
    assert!(n >= budgeted + budgeted / 10, "{n}");
    assert!(
        placement.len() >= 2
            && r0
                .buffers(n)
                .iter()
                .all(|&(b, _)| !placement.contains(b) || matches!(b, KvBuffer::K(_)))
    );
    let system = r0.system_bytes(budgeted, &KvPlacement::default());
    for agreed in [budgeted, budgeted + 1, (budgeted + n) / 2, n - 1, n] {
        assert!(r0.system_bytes(agreed, &placement) <= system, "{agreed}");
        assert!(
            carveout_footprint(r0, agreed, &placement) <= GB10,
            "{agreed}"
        );
    }
    // The planned count is the largest that fits.
    let over = KvPlacement::plan(&r0.buffers(n + 1), GB10);
    assert!(r0.system_bytes(n + 1, &over) > system);
}

#[test]
fn a_shards_pool_is_sized_at_its_local_slots() {
    let cfg = config(11, KvCacheDtype::Fp8G128);
    let k = |shape: KvShape, n| {
        shape
            .buffers(n)
            .into_iter()
            .find(|&(b, _)| b == KvBuffer::K(0))
            .unwrap()
            .1
    };
    assert_eq!(k(sharded(&cfg, 0), 1001), 501 * 16 * 528);
    assert_eq!(k(sparkglm(&cfg), 1001), 1001 * 16 * 528);
    // The scratch's identity table grows with the pool once it outgrows the view.
    let shape = sharded(&cfg, 1);
    assert_eq!(shape.side_bytes(1000), shape.side_bytes(2000));
    assert_eq!(shape.side_bytes(100_002) - shape.side_bytes(100_000), 4);
    assert_eq!(sparkglm(&cfg).side_bytes(100_000), 0);
}
