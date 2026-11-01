// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_KV_SHARD=1` pool accounting: halved latent bytes, and a plan
//! that always covers what `PagedKvCache::new_latent_sharded` allocates.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

fn config(layers: usize) -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: layers,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

/// The production plan (V aliases K, lent index tails) with and without
/// the latent shard; `slotted = false` keeps per-block tails.
fn plans(layers: usize, spec: LatentShardSpec, slotted: bool) -> (GlmCachePlan, GlmCachePlan) {
    let shape = GlmMlaShape::new(512, 0).unwrap();
    let cfg = config(layers);
    let index = shape.bf16_index(4, 128).unwrap();
    let mut base = GlmCachePlan::new(shape, &cfg, Some(index))
        .unwrap()
        .aliased_v(&cfg);
    if slotted {
        let slots = TailSlotPlan {
            lag_blocks: 515,
            sequences: 5,
        };
        base = base.slotted_tails(&cfg, index, slots);
    }
    (base, base.latent_sharded(&cfg, spec))
}

#[test]
fn a_sharded_block_costs_half_its_latents_and_keeps_its_index() {
    let spec = crate::layers::glm_kv_shard::spec(0, &config(11), 524_288, 8256);
    let (base, sharded) = plans(11, spec, true);
    // Per 16-token block: 11 layers x (16 x 528 fp8_g128 latent + 16 / 4
    // pooled keys x 128 x 2 B) + a u32 tail-slot entry. Per token 6,512.25 B
    // before, 3,608.4 B after (half a u32 identity entry per block).
    assert_eq!(base.block_bytes_all_layers(), 11 * 8448 + 11 * 1024 + 4);
    assert_eq!(
        sharded.block_bytes_all_layers(),
        11 * 8448 / 2 + 11 * 1024 + 4 + 2
    );
    assert_eq!(sharded.shard(), Some(spec));
    assert_eq!(base.shard(), None);
    // The same budget buys ~1.8x the blocks, less the fixed shard scratch.
    let budget = 8usize << 30;
    let (b, s) = (
        base.num_blocks_for_budget(budget),
        sharded.num_blocks_for_budget(budget),
    );
    assert!(s * 10 > b * 17 && s < b * 2, "{b} -> {s}");
}

#[test]
fn the_sharded_plan_covers_every_allocation() {
    let shape = GlmMlaShape::new(512, 0).unwrap();
    let index = shape.bf16_index(4, 128).unwrap();
    for rank in 0..2 {
        let spec = crate::layers::glm_kv_shard::spec(rank, &config(3), 2048, 64);
        let (_, plan) = plans(3, spec, false);
        for blocks in [1, 2, 7, 64, 129, 130] {
            let gpu = MockGpuBackend::new();
            let mut cache =
                PagedKvCache::new_latent_sharded(config(3), blocks, &gpu, spec).unwrap();
            cache.attach_sparse_index(index, &gpu).unwrap();
            let shard = cache.latent_shard().unwrap();
            let size = |p| gpu.read_alloc(p).map_or(0, |a| a.len());
            let allocated: usize = size(shard.scratch)
                + (0..3)
                    .map(|l| {
                        size(cache.latent_pool_ptr(l))
                            + size(cache.sparse_index_pool_ptr(l))
                            + size(cache.sparse_index_tail_pool_ptr(l))
                    })
                    .sum::<usize>();
            let accounted = plan.bytes_for_blocks(blocks).unwrap();
            assert!(
                allocated <= accounted,
                "{blocks}: {allocated} > {accounted}"
            );
            // Slack: the odd latent slot and half an identity entry a block.
            assert!(
                accounted - allocated <= 3 * 8448 + 2 * blocks + 4,
                "{blocks}"
            );
        }
    }
}
