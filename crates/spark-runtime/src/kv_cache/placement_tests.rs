// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::carveout::CarveoutArena;
use crate::gpu::mock::MockGpuBackend;
use crate::kv_cache::KvCacheDtype;
use atlas_core::scope::ModelResource;

const MIB: usize = 1 << 20;

/// GLM-shaped: one 512-wide fp8_g128 latent per layer (8448 B/block) and a
/// BF16 four-token semantic index (1024 B values + 8192 B tail per block).
fn glm(layers: usize) -> (KvCacheConfig, SparseIndexCacheConfig) {
    let config = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 512,
        num_layers: layers,
        dtype: KvCacheDtype::Fp8G128,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    (config, SparseIndexCacheConfig::bf16(4, 128))
}

/// The real constructors, as `glm::new_kv_cache` and the factory call them.
fn build(
    gpu: &MockGpuBackend,
    layers: usize,
    blocks: usize,
    placement: KvPlacement,
) -> PagedKvCache {
    build_slotted(gpu, layers, blocks, placement, None)
}

fn build_slotted(
    gpu: &MockGpuBackend,
    layers: usize,
    blocks: usize,
    placement: KvPlacement,
    tail_slots: Option<TailSlotPlan>,
) -> PagedKvCache {
    let (config, index) = glm(layers);
    let mut cache = PagedKvCache::new_placed(config, blocks, gpu, true, placement).unwrap();
    cache
        .attach_sparse_index_with_tail_slots(index, tail_slots, gpu)
        .unwrap();
    cache
}

#[test]
fn plan_takes_the_largest_buffers_that_fit() {
    let (config, index) = glm(3);
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), None);
    // K 844800 B and tails 819200 B occupy 13 × 64 KiB each, values 2 × 64 KiB.
    let placement = KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Size);
    let expected: BTreeSet<_> = [
        KvBuffer::K(0),
        KvBuffer::K(1),
        KvBuffer::IndexValues(0),
        KvBuffer::IndexValues(1),
        KvBuffer::IndexValues(2),
    ]
    .into();
    assert_eq!(placement.carveout, expected);
    assert_eq!(placement.carved_bytes(&sizes), 2 * 844_800 + 3 * 102_400);
    // Zero-byte buffers (BF16 index scales) are never placed.
    assert!(
        sizes
            .iter()
            .any(|&(b, n)| b == KvBuffer::IndexScales(0) && n == 0)
    );
}

#[test]
fn latent_order_places_only_latent_pools() {
    let (config, index) = glm(3);
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), None);
    // 32 units of 64 KiB hold two 13-unit K pools; the index buffers that
    // the size order would add after them stay in system memory.
    let placement = KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Latent);
    let expected: BTreeSet<_> = [KvBuffer::K(0), KvBuffer::K(1)].into();
    assert_eq!(placement.carveout, expected);
    // Room for everything still places no index buffer.
    let all = KvPlacement::plan_ordered(&sizes, 64 * MIB, CarveoutOrder::Latent);
    let expected: BTreeSet<_> = [KvBuffer::K(0), KvBuffer::K(1), KvBuffer::K(2)].into();
    assert_eq!(all.carveout, expected);
}

#[test]
fn buffer_sizes_match_what_the_constructors_allocate() {
    let (config, index) = glm(3);
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), None);
    // Room for everything: every non-empty listed buffer must land in the
    // carveout at exactly its listed size.
    let gpu = MockGpuBackend::with_carveout(64 * MIB);
    let placement = KvPlacement::plan_ordered(&sizes, 64 * MIB, CarveoutOrder::Size);
    let _cache = build(&gpu, 3, 100, placement);
    let mut listed: Vec<usize> = sizes.iter().map(|&(_, n)| n).filter(|&n| n > 0).collect();
    let mut allocated = gpu.carveout_alloc_sizes();
    listed.sort_unstable();
    allocated.sort_unstable();
    assert_eq!(allocated, listed);
}

#[test]
fn buffer_sizes_match_with_slot_mapped_tails() {
    let (config, index) = glm(2);
    let slots = TailSlotPlan {
        lag_blocks: 6,
        sequences: 3,
    };
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), Some(slots));
    assert!(sizes.contains(&(KvBuffer::TailMap, 400)));
    let gpu = MockGpuBackend::with_carveout(64 * MIB);
    let _cache = build_slotted(
        &gpu,
        2,
        100,
        KvPlacement::plan_ordered(&sizes, 64 * MIB, CarveoutOrder::Size),
        Some(slots),
    );
    let mut listed: Vec<usize> = sizes.iter().map(|&(_, n)| n).filter(|&n| n > 0).collect();
    let mut allocated = gpu.carveout_alloc_sizes();
    listed.sort_unstable();
    allocated.sort_unstable();
    assert_eq!(allocated, listed);
}

#[test]
fn only_placed_buffers_use_the_carveout() {
    let (config, index) = glm(3);
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), None);
    let placement = KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Size);
    let gpu = MockGpuBackend::with_carveout(2 * MIB);
    let _cache = build(&gpu, 3, 100, placement.clone());
    let footprints: usize = sizes
        .iter()
        .filter(|(b, _)| placement.carveout.contains(b))
        .map(|&(_, n)| CarveoutArena::footprint(n))
        .sum();
    assert_eq!(gpu.carveout_used(), footprints);
    assert_eq!(gpu.carveout_alloc_sizes().len(), placement.len());
}

#[test]
fn a_placement_planned_at_more_blocks_fits_fewer() {
    // Ranks agree on a block count no larger than the one each planned at.
    let (config, index) = glm(3);
    let planned = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), None);
    let placement = KvPlacement::plan_ordered(&planned, 2 * MIB, CarveoutOrder::Size);
    let gpu = MockGpuBackend::with_carveout(2 * MIB);
    let _cache = build(&gpu, 3, 61, placement.clone());
    let fewer = PagedKvCache::buffer_sizes(&config, 61, true, Some(index), None);
    assert!(placement.carved_bytes(&fewer) < placement.carved_bytes(&planned));
}

#[test]
fn release_returns_carveout_memory() {
    let (config, index) = glm(3);
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), None);
    let gpu = MockGpuBackend::with_carveout(2 * MIB);
    let mut cache = build(
        &gpu,
        3,
        100,
        KvPlacement::plan_ordered(&sizes, 2 * MIB, CarveoutOrder::Size),
    );
    assert!(gpu.carveout_used() > 0);
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.carveout_used(), 0);
}

#[test]
fn without_a_carveout_nothing_is_placed() {
    let (config, index) = glm(2);
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, Some(index), None);
    assert!(KvPlacement::plan_ordered(&sizes, 0, CarveoutOrder::Size).is_empty());
    let gpu = MockGpuBackend::new();
    let _cache = build(&gpu, 2, 100, KvPlacement::default());
    assert!(gpu.carveout_alloc_sizes().is_empty());
}

#[test]
fn a_latent_shard_places_its_local_slot_pools() {
    // `ATLAS_GLM_KV_SHARD=1`: each rank's K pool holds ceil(101 / 2) = 51
    // slots, and the placement is planned over exactly those sizes.
    let (config, index) = glm(3);
    let local = crate::kv_cache::LatentShard::local_blocks_for(101, 2);
    let sizes =
        PagedKvCache::buffer_sizes_with_k_slots(&config, 101, local, true, Some(index), None);
    assert!(sizes.contains(&(KvBuffer::K(0), 51 * 8448)));
    for rank in 0..2 {
        let gpu = MockGpuBackend::with_carveout(64 * MIB);
        let spec = crate::kv_cache::LatentShardSpec {
            rank,
            world: 2,
            scratch_bytes: 4096,
            view_blocks: 8,
            write_rows: 4,
            lane: false,
        };
        let placement = KvPlacement::plan_ordered(&sizes, 64 * MIB, CarveoutOrder::Size);
        let mut cache =
            PagedKvCache::new_latent_sharded_placed(glm(3).0, 101, &gpu, spec, placement).unwrap();
        cache.attach_sparse_index(index, &gpu).unwrap();
        let mut listed: Vec<usize> = sizes.iter().map(|&(_, n)| n).filter(|&n| n > 0).collect();
        let mut allocated = gpu.carveout_alloc_sizes();
        listed.sort_unstable();
        allocated.sort_unstable();
        assert_eq!(allocated, listed, "rank {rank}");
        // The latent order keeps the index in system memory here too.
        let latent = KvPlacement::plan_ordered(&sizes, 64 * MIB, CarveoutOrder::Latent);
        assert_eq!(latent.len(), 3);
        cache.release(&gpu).unwrap();
        assert_eq!(gpu.carveout_used(), 0);
    }
}

#[test]
fn failed_placed_pool_allocation_releases_both_arenas_and_can_retry() {
    let (config, _) = glm(3);
    let sizes = PagedKvCache::buffer_sizes(&config, 100, true, None, None);
    // One latent pool fits; the other two allocations use system memory.
    let capacity = CarveoutArena::footprint(sizes[0].1);
    let placement = KvPlacement::plan_ordered(&sizes, capacity, CarveoutOrder::Latent);
    assert_eq!(placement.len(), 1);
    for fail_after in 0..config.num_layers {
        let gpu = MockGpuBackend::with_carveout(capacity);
        gpu.fail_alloc_after(fail_after);
        assert!(
            PagedKvCache::new_placed(config.clone(), 100, &gpu, true, placement.clone()).is_err()
        );
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.carveout_used(), 0);
        assert_eq!(gpu.invalid_free_count(), 0);
        let mut cache =
            PagedKvCache::new_placed(config.clone(), 100, &gpu, true, placement.clone()).unwrap();
        cache.release(&gpu).unwrap();
        assert_eq!(gpu.alloc_count(), 0);
        assert_eq!(gpu.carveout_used(), 0);
        assert_eq!(gpu.invalid_free_count(), 0);
    }
}
