// SPDX-License-Identifier: AGPL-3.0-only

//! Block-run zeroing and V-aliases-K pool tests.

use super::*;

#[test]
fn zero_blocks_zeroes_exactly_the_given_blocks_one_memset_per_run() {
    let gpu = MockGpuBackend::new();
    let cache = PagedKvCache::new(test_config(), 10, &gpu).unwrap();
    let stride = cache.block_stride_bytes();
    let fill = vec![0xABu8; stride];
    for layer in 0..12 {
        for blk in 0..10 {
            cache.write_block(layer, blk, &fill, &fill, &gpu).unwrap();
        }
    }
    let before = gpu.memset_count();
    // Unsorted ids forming two contiguous runs: 3..=5 and 7.
    cache.zero_blocks(&[5, 4, 3, 7], &gpu, 0).unwrap();
    assert_eq!(gpu.memset_count() - before, 12 * 2 * 2);
    for layer in 0..12 {
        for blk in 0..10 {
            let (k, v) = cache.read_block(layer, blk, &gpu).unwrap();
            let want = if [3, 4, 5, 7].contains(&blk) { 0 } else { 0xAB };
            assert!(
                k.iter().chain(&v).all(|&b| b == want),
                "layer {layer} block {blk}"
            );
        }
    }
}

#[test]
fn aliased_v_pool_shares_k_storage_and_frees_once() {
    use atlas_core::scope::ModelResource;
    let gpu = MockGpuBackend::new();
    let mut cache = PagedKvCache::new_with_v_alias(test_config(), 4, &gpu, true).unwrap();
    for layer in 0..12 {
        assert_eq!(cache.k_pool_ptr(layer), cache.v_pool_ptr(layer));
    }
    let stride = cache.block_stride_bytes();
    cache
        .write_block(3, 2, &vec![7u8; stride], &vec![7u8; stride], &gpu)
        .unwrap();
    let (k, v) = cache.read_block(3, 2, &gpu).unwrap();
    assert!(k.iter().chain(&v).all(|&b| b == 7));
    cache.zero_blocks(&[2], &gpu, 0).unwrap();
    assert!(
        cache
            .read_block(3, 2, &gpu)
            .unwrap()
            .0
            .iter()
            .all(|&b| b == 0)
    );
    cache.release(&gpu).unwrap();
}
