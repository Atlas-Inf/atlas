// SPDX-License-Identifier: AGPL-3.0-only

//! Slotted sliding-window pools: map publication, lag release, exhaustion.

use super::*;

fn swa_cache(gpu: &MockGpuBackend, lag_blocks: usize) -> PagedKvCache {
    let plan = SwaSlotPlan {
        layers: vec![1, 2],
        lag_blocks,
        sequences: 1,
        window_blocks: 2,
        prefix_tails: 0,
        tail_blocks: 0,
    };
    PagedKvCache::new_with_swa_slots(test_config(), 64, gpu, KvPlacement::default(), &plan).unwrap()
}

fn device_swa_map(cache: &PagedKvCache, gpu: &MockGpuBackend) -> Vec<u32> {
    let mut bytes = vec![0u8; cache.num_blocks() * 4];
    gpu.copy_d2h(cache.swa_slot_map_ptr(1), &mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn only_planned_layers_read_a_slot_map() {
    let gpu = MockGpuBackend::new();
    let cache = swa_cache(&gpu, 2);
    assert!(cache.swa_slot_map_ptr(0).is_null());
    assert!(cache.swa_slot_map_ptr(3).is_null());
    assert!(!cache.swa_slot_map_ptr(1).is_null());
    assert_eq!(cache.swa_slot_map_ptr(1), cache.swa_slot_map_ptr(2));
    let plain = PagedKvCache::new(test_config(), 64, &gpu).unwrap();
    assert!(plain.swa_slot_map_ptr(1).is_null());
}

#[test]
fn slots_release_only_blocks_beyond_the_lag() {
    let gpu = MockGpuBackend::new();
    let mut cache = swa_cache(&gpu, 2);
    let mut table = Vec::new();
    for newest in 0..12usize {
        cache.release_lagging_swa_slots(&table, newest, 0);
        let block = cache.alloc_block().unwrap();
        table.push(block);
        cache.lend_swa_slots(&[block], &gpu, 0).unwrap();
        let map = device_swa_map(&cache, &gpu);
        for (logical, &b) in table.iter().enumerate() {
            assert_eq!(
                map[b as usize] != NO_TAIL,
                logical + 2 >= newest,
                "newest={newest} logical={logical}"
            );
        }
    }
}

#[test]
fn slots_return_on_free_and_refuse_exhaustion() {
    let gpu = MockGpuBackend::new();
    let mut cache = swa_cache(&gpu, 0);
    let a = cache.alloc_block().unwrap();
    let b = cache.alloc_block().unwrap();
    cache.lend_swa_slots(&[a, b], &gpu, 0).unwrap();
    let c = cache.alloc_block().unwrap();
    let err = cache.lend_swa_slots(&[c], &gpu, 0).unwrap_err();
    assert!(
        err.to_string()
            .contains("sliding-window KV slots exhausted"),
        "{err}"
    );
    cache.free_block(a);
    cache.lend_swa_slots(&[c], &gpu, 0).unwrap();
    let map = device_swa_map(&cache, &gpu);
    assert_eq!(map[a as usize], NO_TAIL);
    assert_ne!(map[c as usize], NO_TAIL);
}

fn pinned_cache(gpu: &MockGpuBackend, sequences: usize, prefix_tails: usize) -> PagedKvCache {
    pinned_cache_lag(gpu, sequences, prefix_tails, 2)
}

fn pinned_cache_lag(
    gpu: &MockGpuBackend,
    sequences: usize,
    prefix_tails: usize,
    lag_blocks: usize,
) -> PagedKvCache {
    let plan = SwaSlotPlan {
        layers: vec![1, 2],
        lag_blocks,
        sequences,
        window_blocks: 2,
        prefix_tails,
        tail_blocks: 3,
    };
    PagedKvCache::new_with_swa_slots(test_config(), 64, gpu, KvPlacement::default(), &plan).unwrap()
}

/// Decode-style growth: release what lags, then lend one fresh block.
fn grow(
    cache: &mut PagedKvCache,
    gpu: &MockGpuBackend,
    table: &mut Vec<u32>,
    to: usize,
    owned: usize,
) {
    for newest in table.len()..to {
        cache.release_lagging_swa_slots(table, newest, owned);
        let block = cache.alloc_block().unwrap();
        table.push(block);
        cache.lend_swa_slots(&[block], gpu, 0).unwrap();
    }
}

#[test]
fn pinned_prompt_tail_outlives_its_sequence_until_evicted() {
    let gpu = MockGpuBackend::new();
    let mut cache = pinned_cache(&gpu, 1, 1);
    assert!(cache.swa_prefix_tails_enabled());
    let mut table = Vec::new();
    grow(&mut cache, &gpu, &mut table, 6, 0);
    let prompt = table.clone();
    cache.pin_swa_tail(&prompt);
    assert_eq!(cache.ref_count(prompt[5]), 2);
    // Decode moves on; then the sequence leaves.
    grow(&mut cache, &gpu, &mut table, 10, 0);
    cache.release_swa_slots(&table);
    cache.free_blocks(&table);
    cache.lend_swa_slots(&[], &gpu, 0).unwrap();
    let map = device_swa_map(&cache, &gpu);
    assert!(prompt[3..].iter().all(|&b| map[b as usize] != NO_TAIL));
    assert!(table[6..].iter().all(|&b| map[b as usize] == NO_TAIL));
    assert_eq!(cache.swa_adoptable_blocks(&prompt), Some(6));
    assert_eq!(cache.swa_adoptable_blocks(&prompt[..5]), Some(5));
    assert_eq!(cache.swa_adoptable_blocks(&prompt[..4]), Some(0));
    // A second prompt's tail takes the only pin: the first is released.
    let mut other = Vec::new();
    grow(&mut cache, &gpu, &mut other, 3, 0);
    cache.pin_swa_tail(&other);
    assert!(prompt[3..].iter().all(|&b| cache.ref_count(b) == 0));
    assert_eq!(cache.swa_adoptable_blocks(&prompt), Some(0));
    assert_eq!(cache.swa_adoptable_blocks(&other), Some(3));
}

#[test]
fn repinning_a_tail_refreshes_it_without_more_references() {
    let gpu = MockGpuBackend::new();
    let mut cache = pinned_cache(&gpu, 2, 2);
    let (mut a, mut b) = (Vec::new(), Vec::new());
    grow(&mut cache, &gpu, &mut a, 3, 0);
    cache.pin_swa_tail(&a);
    grow(&mut cache, &gpu, &mut b, 3, 0);
    cache.pin_swa_tail(&b);
    cache.pin_swa_tail(&a);
    assert_eq!(cache.ref_count(a[0]), 2);
    // `b` is now least recent: a third tail evicts it, not `a`.
    let mut c = Vec::new();
    cache.release_swa_slots(&a);
    cache.release_swa_slots(&b);
    grow(&mut cache, &gpu, &mut c, 3, 0);
    cache.pin_swa_tail(&c);
    assert_eq!(cache.ref_count(b[0]), 1);
    assert_eq!(cache.ref_count(a[0]), 2);
    assert_eq!(cache.swa_adoptable_blocks(&b), Some(0));
    assert_eq!(cache.swa_adoptable_blocks(&a), Some(3));
}

#[test]
fn evicting_a_tail_shared_with_the_new_one_keeps_the_shared_blocks() {
    let gpu = MockGpuBackend::new();
    let mut cache = pinned_cache_lag(&gpu, 1, 1, 100);
    let mut table = Vec::new();
    grow(&mut cache, &gpu, &mut table, 7, 0);
    // Tail of `table[..6]` is blocks [3, 4, 5].
    cache.pin_swa_tail(&table[..6]);
    cache.release_swa_slots(&table);
    cache.free_blocks(&table);
    cache.lend_swa_slots(&[], &gpu, 0).unwrap();
    // A second prompt whose held tail [3, 4] overlaps the first: evicting the
    // first must not drop the pins (or free the blocks) the second needs.
    cache.pin_swa_tail(&[table[2], table[3], table[4]]);
    cache.lend_swa_slots(&[], &gpu, 0).unwrap();
    assert_eq!(cache.ref_count(table[3]), 1);
    assert_eq!(cache.ref_count(table[4]), 1);
    assert_eq!(cache.ref_count(table[5]), 0);
    let map = device_swa_map(&cache, &gpu);
    assert_ne!(map[table[3] as usize], NO_TAIL);
    assert_ne!(map[table[4] as usize], NO_TAIL);
    assert_eq!(map[table[5] as usize], NO_TAIL);
}

#[test]
fn lag_release_skips_blocks_taken_from_the_cache() {
    let gpu = MockGpuBackend::new();
    let mut cache = pinned_cache(&gpu, 2, 0);
    assert!(!cache.swa_prefix_tails_enabled());
    let mut lender = Vec::new();
    grow(&mut cache, &gpu, &mut lender, 2, 0);
    let mut table = lender.clone();
    grow(&mut cache, &gpu, &mut table, 9, 2);
    cache.lend_swa_slots(&[], &gpu, 0).unwrap();
    let map = device_swa_map(&cache, &gpu);
    assert!(lender.iter().all(|&b| map[b as usize] != NO_TAIL));
    assert!(table[2..6].iter().all(|&b| map[b as usize] == NO_TAIL));
    assert!(table[6..].iter().all(|&b| map[b as usize] != NO_TAIL));
}
