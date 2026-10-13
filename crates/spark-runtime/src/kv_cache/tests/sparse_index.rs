// SPDX-License-Identifier: AGPL-3.0-only

//! Sparse-index pool tests: geometry, per-layer attach, and slotted tails.

use super::*;

#[test]
fn test_sparse_index_cache_geometry_and_budgeting() {
    let cfg = test_config();
    let index = SparseIndexCacheConfig::bf16(4, 128);

    // Four pooled keys plus raw key+gate staging for all 16 token offsets.
    assert_eq!(index.block_bytes(cfg.block_size).unwrap(), 9216);
    assert_eq!(
        PagedKvCache::compute_num_blocks_with_sparse_index(&cfg, index, 1_000_000).unwrap(),
        1_000_000 / (196_608 + 12 * 9216)
    );
}

#[test]
fn test_sparse_index_cache_attaches_to_each_layer() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 10, &gpu).unwrap();
    let index = SparseIndexCacheConfig::bf16(4, 128);

    cache.attach_sparse_index(index, &gpu).unwrap();
    assert!(!cache.sparse_index_pool_ptr(0).is_null());
    assert_ne!(
        cache.sparse_index_pool_ptr(0),
        cache.sparse_index_pool_ptr(1)
    );
    assert_eq!(cache.sparse_index_block_stride_bytes(0), 1024);
    assert!(!cache.sparse_index_tail_pool_ptr(0).is_null());
    assert_eq!(cache.sparse_index_tail_block_stride_bytes(0), 8192);
    assert_eq!(cache.sparse_index_config(), Some(index));
}

fn device_tail_map(cache: &PagedKvCache, gpu: &MockGpuBackend) -> Vec<u32> {
    let mut bytes = vec![0u8; cache.num_blocks() * 4];
    gpu.copy_d2h(cache.sparse_index_tail_map_ptr(), &mut bytes)
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn slotted_tails_are_lent_to_fresh_blocks_and_published() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 64, &gpu).unwrap();
    let plan = TailSlotPlan {
        lag_blocks: 2,
        sequences: 2,
    };
    cache
        .attach_sparse_index_with_tail_slots(SparseIndexCacheConfig::bf16(4, 128), Some(plan), &gpu)
        .unwrap();
    // The tail pool holds capacity() slots, not one per block.
    assert_eq!(plan.capacity(), 8);
    assert!(device_tail_map(&cache, &gpu).iter().all(|&s| s == NO_TAIL));

    let blocks: Vec<u32> = (0..3).map(|_| cache.alloc_block().unwrap()).collect();
    cache.lend_tail_slots(&blocks, &gpu, 0).unwrap();
    let map = device_tail_map(&cache, &gpu);
    let mut slots: Vec<u32> = blocks.iter().map(|&b| map[b as usize]).collect();
    slots.sort_unstable();
    slots.dedup();
    assert_eq!(slots.len(), 3, "each fresh block owns a distinct slot");
    assert!(slots.iter().all(|&s| (s as usize) < plan.capacity()));
    // Lending again is idempotent.
    cache.lend_tail_slots(&blocks, &gpu, 0).unwrap();
    assert_eq!(device_tail_map(&cache, &gpu), map);
}

#[test]
fn slotted_tails_release_only_blocks_beyond_the_lag() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 64, &gpu).unwrap();
    let plan = TailSlotPlan {
        lag_blocks: 2,
        sequences: 1,
    };
    cache
        .attach_sparse_index_with_tail_slots(SparseIndexCacheConfig::bf16(4, 128), Some(plan), &gpu)
        .unwrap();
    let mut table = Vec::new();
    // Decode-like growth: each step opens one block and releases what lags.
    for newest in 0..10usize {
        cache.release_lagging_tail_slots(&table, newest);
        let block = cache.alloc_block().unwrap();
        table.push(block);
        cache.lend_tail_slots(&[block], &gpu, 0).unwrap();
        let map = device_tail_map(&cache, &gpu);
        for (logical, &b) in table.iter().enumerate() {
            let lent = map[b as usize] != NO_TAIL;
            assert_eq!(
                lent,
                logical + plan.lag_blocks >= newest,
                "newest={newest} logical={logical}"
            );
        }
    }
}

#[test]
fn slotted_tails_return_on_free_and_refuse_exhaustion() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 64, &gpu).unwrap();
    let plan = TailSlotPlan {
        lag_blocks: 0,
        sequences: 1,
    };
    cache
        .attach_sparse_index_with_tail_slots(SparseIndexCacheConfig::bf16(4, 128), Some(plan), &gpu)
        .unwrap();
    let a = cache.alloc_block().unwrap();
    let b = cache.alloc_block().unwrap();
    cache.lend_tail_slots(&[a, b], &gpu, 0).unwrap();
    let c = cache.alloc_block().unwrap();
    let err = cache.lend_tail_slots(&[c], &gpu, 0).unwrap_err();
    assert!(err.to_string().contains("tail slots exhausted"), "{err}");
    // Freeing a block returns its slot; the freed entry is unpublished in the
    // same flush that lends the slot again.
    cache.free_block(a);
    cache.lend_tail_slots(&[c], &gpu, 0).unwrap();
    let map = device_tail_map(&cache, &gpu);
    assert_eq!(map[a as usize], NO_TAIL);
    assert_ne!(map[c as usize], NO_TAIL);
    assert_ne!(map[c as usize], map[b as usize]);
}

#[test]
fn tail_slot_missing_only_for_unlent_slotted_blocks() {
    let gpu = MockGpuBackend::new();
    let index = SparseIndexCacheConfig::bf16(4, 128);
    let mut per_block = PagedKvCache::new(test_config(), 8, &gpu).unwrap();
    per_block.attach_sparse_index(index, &gpu).unwrap();
    let block = per_block.alloc_block().unwrap();
    assert!(!per_block.tail_slot_missing(block));

    let mut cache = PagedKvCache::new(test_config(), 8, &gpu).unwrap();
    let plan = TailSlotPlan {
        lag_blocks: 1,
        sequences: 1,
    };
    cache
        .attach_sparse_index_with_tail_slots(index, Some(plan), &gpu)
        .unwrap();
    let block = cache.alloc_block().unwrap();
    assert!(cache.tail_slot_missing(block));
    cache.lend_tail_slots(&[block], &gpu, 0).unwrap();
    assert!(!cache.tail_slot_missing(block));
}

#[test]
fn slotted_tails_are_not_zeroed_by_block() {
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new(test_config(), 64, &gpu).unwrap();
    cache
        .attach_sparse_index_with_tail_slots(
            SparseIndexCacheConfig::bf16(4, 128),
            Some(TailSlotPlan {
                lag_blocks: 1,
                sequences: 1,
            }),
            &gpu,
        )
        .unwrap();
    let tail = cache.sparse_index_tail_pool_ptr(0);
    gpu.memset(tail, 0xAB, 8192).unwrap();
    // Block 40 lies far outside the 3-slot tail pool: zeroing it must not
    // address the tail pool by block id.
    cache.zero_blocks(&[40], &gpu, 0).unwrap();
    let mut first = vec![0u8; 8192];
    gpu.copy_d2h(tail, &mut first).unwrap();
    assert!(first.iter().all(|&b| b == 0xAB));
}
