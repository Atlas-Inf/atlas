// SPDX-License-Identifier: AGPL-3.0-only

//! The drafter's KV geometry (`drafter_kv_config`).

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

/// The written slot keeps a full layer's strides; the twelve slots in
/// front of it cost next to nothing.
#[test]
fn only_the_written_slot_is_a_full_layer() {
    let cfg = drafter_kv_config(12, 2, 256);
    let full = 16 * 2 * 256 * 2;
    assert_eq!(cfg.k_block_bytes_for_layer(12), full);
    assert_eq!(cfg.v_block_bytes_for_layer(12), full);
    assert_eq!(cfg.cache_stride_elements(), 16 * 2 * 256);
    assert_eq!(cfg.block_bytes_kv_all_layers(), 2 * full + 12 * 2 * 32);

    let gpu = MockGpuBackend::new();
    let blocks = 64;
    let kv = PagedKvCache::new(cfg, blocks, &gpu).unwrap();
    assert_eq!(kv.k_block_stride_bytes_for_layer(12), full);
    assert_eq!(kv.v_block_stride_bytes_for_layer(12), full);
    assert_eq!(kv.dtype_for_layer(12), KvCacheDtype::Bf16);
    let k12 = gpu.read_alloc(kv.k_pool_ptr(12)).unwrap();
    assert_eq!(k12.len(), blocks * full);
    assert_eq!(gpu.read_alloc(kv.k_pool_ptr(0)).unwrap().len(), blocks * 32);
}
