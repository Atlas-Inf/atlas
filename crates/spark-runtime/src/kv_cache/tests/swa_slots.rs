// SPDX-License-Identifier: AGPL-3.0-only

//! Slotted sliding-window pools: map publication, lag release, exhaustion.

use super::*;

fn swa_cache(gpu: &MockGpuBackend, lag_blocks: usize) -> PagedKvCache {
    let plan = SwaSlotPlan {
        layers: vec![1, 2],
        lag_blocks,
        sequences: 1,
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
        cache.release_lagging_swa_slots(&table, newest);
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
