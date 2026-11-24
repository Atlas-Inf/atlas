// SPDX-License-Identifier: AGPL-3.0-only

//! Window-bounded KV pools for sliding-window layers (`window.rs`).

use super::*;

const WINDOW: usize = 1024;

/// Four attention layers, the odd ones sliding-window, 16-token blocks.
fn windowed(blocks: usize, local: usize, pins: usize, gpu: &MockGpuBackend) -> PagedKvCache {
    let config = KvCacheConfig {
        num_layers: 4,
        dtype: KvCacheDtype::Bf16,
        ..test_config()
    };
    let plan = WindowPlan {
        layers: vec![1, 3],
        window: WINDOW,
        blocks: local,
        pins,
    };
    PagedKvCache::new_windowed(config, blocks, plan, gpu).unwrap()
}

fn local_of(cache: &PagedKvCache, gpu: &MockGpuBackend, block: u32) -> u32 {
    let map = gpu.read_alloc(cache.window_map_ptr()).unwrap();
    let at = block as usize * 4;
    u32::from_le_bytes(map[at..at + 4].try_into().unwrap())
}

#[test]
fn first_block_starts_at_the_key_tile_below_the_window() {
    let plan = WindowPlan {
        layers: vec![1],
        window: WINDOW,
        blocks: 2,
        pins: 0,
    };
    // the first row at p reads keys from p + 1 - W, loaded from the 32-key
    // tile that holds it (the prefill kernels' tile start)
    for (first_pos, block) in [(0, 0), (1023, 0), (1024, 0), (1056, 2), (5000, 248)] {
        assert_eq!(
            plan.first_block(first_pos, 16),
            block,
            "first row {first_pos}"
        );
    }
}

#[test]
fn plan_holds_every_live_span_the_batch_transient_and_the_pins() {
    // W + 31 tile keys + 16 verify rows over 16-token blocks, plus the
    // block a row run can straddle
    assert_eq!(WindowPlan::span_blocks(WINDOW, 16, 16), 68);
    assert_eq!(WindowPlan::span_blocks(WINDOW, 16, 0), 67);
    let blocks = WindowPlan::size_blocks(WINDOW, 16, 8, 16, 4104, 8);
    // null block + 8 spans + the batch's rows (and a block per sequence it
    // can straddle) + 8 pinned windows
    assert_eq!(blocks, 1 + 8 * 68 + 4104usize.div_ceil(16) + 8 + 8 * 67);
}

#[test]
fn window_layers_live_outside_the_primary_pools() {
    let gpu = MockGpuBackend::new();
    let cache = windowed(64, 8, 0, &gpu);
    for layer in [1, 3] {
        assert!(cache.is_window_layer(layer));
        assert!(!cache.k_pool_ptr(layer).is_null());
        assert_ne!(cache.k_pool_ptr(layer), cache.v_pool_ptr(layer));
    }
    assert!(!cache.is_window_layer(0) && !cache.is_window_layer(2));
    // zeroing a primary block touches the two primary layers' K and V only
    let before = gpu.memset_count();
    cache.zero_blocks(&[5], &gpu, 0).unwrap();
    assert_eq!(gpu.memset_count() - before, 2 * 2);
}

#[test]
fn hold_lends_a_zeroed_local_block_and_publishes_the_map() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    assert_eq!(local_of(&cache, &gpu, 5), NO_WINDOW);
    let junk = vec![0xABu8; 8 * cache.k_block_stride_bytes_for_layer(1)];
    gpu.copy_h2d(&junk, cache.k_pool_ptr(1)).unwrap();
    cache.window_hold(&[5, 6], &gpu, 0).unwrap();
    let (l5, l6) = (local_of(&cache, &gpu, 5), local_of(&cache, &gpu, 6));
    assert!(l5 != NO_WINDOW && l6 != NO_WINDOW && l5 != l6);
    // block 0 of the window pool is the null block, never lent
    assert!(l5 != 0 && l6 != 0);
    let stride = cache.k_block_stride_bytes_for_layer(1);
    let pool = gpu.read_alloc(cache.k_pool_ptr(1)).unwrap();
    for l in [l5, l6] {
        let at = l as usize * stride;
        assert!(pool[at..at + stride].iter().all(|&b| b == 0), "local {l}");
    }
    assert!(cache.window_mapped(5) && cache.window_mapped(6));
}

#[test]
fn holders_of_one_primary_block_share_its_window_rows() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    cache.window_hold(&[5], &gpu, 0).unwrap();
    let local = local_of(&cache, &gpu, 5);
    cache.window_hold(&[5], &gpu, 0).unwrap();
    assert_eq!(local_of(&cache, &gpu, 5), local);
    assert_eq!(cache.window_free_blocks(), 6);
    cache.window_release(&[5], &gpu, 0).unwrap();
    assert!(cache.window_mapped(5));
    cache.window_release(&[5], &gpu, 0).unwrap();
    assert!(!cache.window_mapped(5));
    assert_eq!(cache.window_free_blocks(), 7);
}

#[test]
fn an_exhausted_window_pool_refuses_the_whole_hold() {
    let gpu = MockGpuBackend::new();
    // null block + two lendable blocks
    let mut cache = windowed(64, 3, 0, &gpu);
    cache.window_hold(&[1], &gpu, 0).unwrap();
    let err = cache.window_hold(&[2, 3], &gpu, 0).unwrap_err();
    // the scheduler reads this as an exhausted pool (preempt and retry)
    assert!(format!("{err:#}").contains("KV cache exhausted"), "{err:#}");
    // nothing of the refused hold stays lent
    assert!(!cache.window_mapped(2) && !cache.window_mapped(3));
    assert_eq!(cache.window_free_blocks(), 1);
}

#[test]
fn pins_keep_a_snapshot_window_until_the_pool_needs_it() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 5, 4, &gpu);
    let (a, b) = (cache.alloc_block().unwrap(), cache.alloc_block().unwrap());
    cache.window_hold(&[a, b], &gpu, 0).unwrap();
    assert!(cache.window_pin(&[a, b]));
    // the pin holds the primary blocks too
    assert_eq!((cache.ref_count(a), cache.ref_count(b)), (2, 2));
    // the sequence leaves; the pin keeps the window
    cache.window_release(&[a, b], &gpu, 0).unwrap();
    cache.free_blocks(&[a, b]);
    assert!(cache.window_mapped(a) && cache.window_mapped(b));
    // two blocks left free; a third evicts the pin
    cache.window_hold(&[10, 11], &gpu, 0).unwrap();
    assert!(cache.window_mapped(a));
    cache.window_hold(&[12], &gpu, 0).unwrap();
    assert!(!cache.window_mapped(a) && !cache.window_mapped(b));
    assert_eq!((cache.ref_count(a), cache.ref_count(b)), (0, 0));
    assert_eq!(cache.window_pins(), 0);
}

#[test]
fn pins_past_their_count_evict_the_oldest() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 1, &gpu);
    let (a, b) = (cache.alloc_block().unwrap(), cache.alloc_block().unwrap());
    cache.window_hold(&[a, b], &gpu, 0).unwrap();
    assert!(cache.window_pin(&[a]));
    assert!(cache.window_pin(&[b]));
    assert_eq!(cache.window_pins(), 1);
    assert_eq!((cache.ref_count(a), cache.ref_count(b)), (1, 2));
    cache.window_release(&[a, b], &gpu, 0).unwrap();
    assert!(!cache.window_mapped(a) && cache.window_mapped(b));
}

#[test]
fn a_window_with_unmapped_rows_is_not_pinned() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 4, &gpu);
    cache.window_hold(&[1], &gpu, 0).unwrap();
    assert!(!cache.window_pin(&[1, 2]));
    assert_eq!(cache.window_pins(), 0);
    assert_eq!(cache.ref_count(1), 0);
}

#[test]
fn a_freed_primary_block_drops_its_window_rows() {
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    let block = cache.alloc_block().unwrap();
    cache.window_hold(&[block], &gpu, 0).unwrap();
    // a holder without a primary ref is a bug; the free still reclaims the
    // rows rather than leave them mapped to a block another sequence gets
    cache.free_block(block);
    assert!(!cache.window_mapped(block));
    assert_eq!(cache.window_free_blocks(), 7);
}

#[test]
fn release_returns_the_window_pools() {
    use atlas_core::scope::ModelResource;
    let gpu = MockGpuBackend::new();
    let mut cache = windowed(64, 8, 0, &gpu);
    let live = gpu.alloc_count();
    cache.release(&gpu).unwrap();
    assert_eq!(gpu.invalid_free_count(), 0);
    assert!(gpu.alloc_count() < live);
}
