// SPDX-License-Identifier: AGPL-3.0-only

//! Latent-shard ownership, allocation, zeroing and fail-closed tests.

use super::*;
use atlas_core::scope::ModelResource;

fn glm_config() -> KvCacheConfig {
    KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: 3,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    }
}

fn spec(rank: usize) -> LatentShardSpec {
    LatentShardSpec {
        rank,
        world: 2,
        scratch_bytes: 4096,
        view_blocks: 8,
        write_rows: 4,
        lane: false,
    }
}

fn shard(rank: usize) -> LatentShard {
    LatentShard {
        spec: spec(rank),
        local_blocks: 0,
        scratch: DevicePtr::NULL,
        identity: DevicePtr::NULL,
        lane: None,
    }
}

#[test]
fn every_block_has_exactly_one_owner_and_a_distinct_local_slot() {
    let (r0, r1) = (shard(0), shard(1));
    for block in 0..64u32 {
        let slots = [r0.local_slot(block), r1.local_slot(block)];
        assert_eq!(slots.iter().filter(|s| s.is_some()).count(), 1, "{block}");
        assert_eq!(slots[r0.owner(block)], Some(block / 2));
    }
    assert_eq!(LatentShard::local_blocks_for(7, 2), 4);
    assert_eq!(LatentShard::local_blocks_for(8, 2), 4);
}

#[test]
fn owned_run_matches_a_per_block_filter() {
    for rank in 0..2 {
        let s = shard(rank);
        for first in 0..12u32 {
            for count in 0..12usize {
                let slots: Vec<u32> = (first..first + count as u32)
                    .filter_map(|b| s.local_slot(b))
                    .collect();
                let (start, n) = s.owned_run(first, count);
                assert_eq!(n, slots.len(), "rank {rank} [{first}, +{count})");
                if let Some(&head) = slots.first() {
                    assert_eq!(start, head as usize);
                    assert!(slots.windows(2).all(|w| w[1] == w[0] + 1));
                }
            }
        }
    }
}

#[test]
fn plan_partitions_logical_blocks_in_order() {
    let table = [4u32, 7, 2, 9, 0];
    let p0 = shard(0).plan(&table).unwrap();
    assert_eq!(p0.mine_logical, vec![0, 2, 4]);
    assert_eq!(p0.mine_slot, vec![2, 1, 0]);
    assert_eq!(p0.peer_logical, vec![1, 3]);
    let p1 = shard(1).plan(&table).unwrap();
    assert_eq!(p1.mine_logical, p0.peer_logical);
    assert_eq!(p1.mine_slot, vec![3, 4]);
    assert_eq!(p1.peer_logical, p0.mine_logical);
}

#[test]
fn a_block_off_its_logical_residue_is_rejected() {
    // Logical block 1 holds an even block: the ranks would disagree.
    for rank in 0..2 {
        let err = shard(rank).plan(&[4, 6, 2]).unwrap_err().to_string();
        assert!(err.contains("logical block 1"), "{err}");
    }
    assert!(shard(0).check_table(&[]).is_ok());
}

/// One rank's pool driven through a shared script, freeing in its own order.
fn run_rank(rank: usize, reverse_frees: bool) -> Vec<Vec<u32>> {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new_latent_sharded(glm_config(), 24, &gpu, spec(rank)).unwrap();
    let alloc = |cache: &mut PagedKvCache, n: usize| -> Vec<u32> {
        (0..n).map(|l| cache.alloc_block_at(l).unwrap()).collect()
    };
    let a = alloc(&mut cache, 5);
    let b = alloc(&mut cache, 3);
    let mut order = a.clone();
    if reverse_frees {
        order.reverse();
    }
    cache.free_blocks(&order);
    let c = alloc(&mut cache, 7);
    cache.free_block(b[1]);
    let mut d = alloc(&mut cache, 2);
    // A sequence extended one block at a time after a prefix of two.
    d.extend((2..6).map(|l| cache.alloc_block_at(l).unwrap()));
    vec![b, c, d]
}

#[test]
fn ranks_freeing_in_different_orders_agree_on_every_owner() {
    let (r0, r1) = (run_rank(0, false), run_rank(1, true));
    assert_ne!(r0, r1, "the script must make the ranks' ids diverge");
    for (t0, t1) in r0.iter().zip(&r1) {
        assert_eq!(t0.len(), t1.len());
        let (p0, p1) = (shard(0).plan(t0).unwrap(), shard(1).plan(t1).unwrap());
        assert_eq!(p0.mine_logical, p1.peer_logical);
        assert_eq!(p0.peer_logical, p1.mine_logical);
    }
}

#[test]
fn free_count_is_what_any_logical_mix_can_draw() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new_latent_sharded(glm_config(), 9, &gpu, spec(0)).unwrap();
    // Residue 0 holds 5 blocks, residue 1 holds 4.
    assert_eq!(cache.num_free_blocks(), 8);
    let even: Vec<u32> = (0..4).map(|_| cache.alloc_block_at(2).unwrap()).collect();
    assert!(even.iter().all(|b| b % 2 == 0));
    assert_eq!(cache.num_free_blocks(), 2);
    cache.alloc_block_at(0).unwrap();
    assert_eq!(cache.num_free_blocks(), 0);
    let err = cache.alloc_block_at(4).unwrap_err().to_string();
    assert!(
        err.contains("logical block 4") && err.contains("4 free"),
        "{err}"
    );
    let odd = cache.alloc_block_at(1).unwrap();
    assert_eq!(odd % 2, 1);
    cache.free_block(even[0]);
    assert_eq!(cache.num_free_blocks(), 2);
    // A block freed into the richer residue draws nothing more, and still
    // counts as reclaimed.
    let in_all = cache.num_free_in_all();
    cache.free_block(odd);
    assert_eq!(cache.num_free_blocks(), 2);
    assert_eq!(cache.num_free_in_all(), in_all + 1);
}

#[test]
fn unsharded_allocation_ignores_the_logical_index() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(glm_config(), 4, &gpu).unwrap();
    assert_eq!(cache.alloc_block_at(1).unwrap(), 0);
    assert_eq!(cache.alloc_block().unwrap(), 1);
    assert_eq!(cache.try_alloc_block(), Some(2));
    assert_eq!(cache.num_free_blocks(), 1);
    assert_eq!(cache.num_free_in_all(), 1);
}

#[test]
fn allocation_without_a_logical_index_is_refused_when_sharded() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new_latent_sharded(glm_config(), 4, &gpu, spec(0)).unwrap();
    assert!(cache.alloc_block().is_err());
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cache.try_alloc_block()));
    assert!(caught.is_err());
}

#[test]
fn sharded_pool_allocates_half_the_latent_slots_plus_scratch() {
    let gpu = MockGpuBackend::new();
    let before = gpu.alloc_count();
    let mut cache = PagedKvCache::new_latent_sharded(glm_config(), 7, &gpu, spec(1)).unwrap();
    let s = cache.latent_shard().unwrap();
    assert_eq!(s.local_blocks, 4);
    let stride = glm_config().k_block_bytes_for_layer(0);
    assert_eq!(stride, 16 * 528);
    for layer in 0..3 {
        let pool = gpu.read_alloc(cache.latent_pool_ptr(layer)).unwrap();
        assert_eq!(pool.len(), 4 * stride);
    }
    // Model scratch, then the identity table over max(4 local, 8 view) blocks.
    let scratch = gpu.read_alloc(s.scratch).unwrap();
    assert_eq!(scratch.len(), 4096 + 8 * 4);
    assert_eq!(s.identity, s.scratch.offset(4096));
    let identity: Vec<u32> = scratch[4096..]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    assert_eq!(identity, (0..8).collect::<Vec<u32>>());
    // Three layer pools (V aliases K) and the scratch.
    assert_eq!(gpu.alloc_count() - before, 4);
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.alloc_count(), before);
}

#[test]
fn the_exchange_lane_lives_exactly_as_long_as_a_cache_that_asked_for_it() {
    let gpu = MockGpuBackend::new();
    let plain = PagedKvCache::new_latent_sharded(glm_config(), 7, &gpu, spec(0)).unwrap();
    assert_eq!(plain.latent_shard().unwrap().lane, None);
    assert_eq!(gpu.live_streams_and_events(), (0, 0));
    let with_lane = LatentShardSpec {
        lane: true,
        ..spec(1)
    };
    let mut cache = PagedKvCache::new_latent_sharded(glm_config(), 7, &gpu, with_lane).unwrap();
    assert!(cache.latent_shard().unwrap().lane.is_some());
    assert_eq!(gpu.live_streams_and_events(), (1, 2));
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.live_streams_and_events(), (0, 0));
}

#[test]
fn zeroing_touches_only_owned_local_slots() {
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new_latent_sharded(glm_config(), 12, &gpu, spec(1)).unwrap();
    let stride = glm_config().k_block_bytes_for_layer(0);
    let pool = cache.latent_pool_ptr(2);
    gpu.memset(pool, 0xAB, 6 * stride).unwrap();
    // Rank 1 owns 3 and 5 of these (local slots 1 and 2) and 9 (slot 4).
    cache.zero_blocks(&[2, 3, 4, 5, 9], &gpu, 0).unwrap();
    let bytes = gpu.read_alloc(pool).unwrap();
    for slot in 0..6 {
        let want = if [1, 2, 4].contains(&slot) { 0 } else { 0xAB };
        let block = &bytes[slot * stride..(slot + 1) * stride];
        assert!(block.iter().all(|&b| b == want), "slot {slot}");
    }
}

#[test]
#[should_panic(expected = "ATLAS_GLM_KV_SHARD")]
fn global_pool_access_fails_closed_when_sharded() {
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new_latent_sharded(glm_config(), 4, &gpu, spec(0)).unwrap();
    let _ = cache.k_pool_ptr(0);
}

#[test]
fn block_io_is_refused_when_sharded() {
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new_latent_sharded(glm_config(), 4, &gpu, spec(0)).unwrap();
    assert!(cache.read_block(0, 0, &gpu).is_err());
    let data = vec![0u8; 16 * 528];
    assert!(cache.write_block(0, 0, &data, &data, &gpu).is_err());
    assert!(
        PagedKvCache::new_latent_sharded(
            glm_config(),
            4,
            &gpu,
            LatentShardSpec {
                world: 3,
                ..spec(0)
            }
        )
        .is_err()
    );
}
