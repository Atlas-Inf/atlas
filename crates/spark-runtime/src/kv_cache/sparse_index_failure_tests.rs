// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;
use crate::kv_cache::{KvCacheDtype, SparseIndexCacheDtype, TailSlotPlan};
use atlas_core::scope::ModelResource;

#[test]
fn failed_sparse_attach_releases_tail_map_and_all_new_pools_before_retry() {
    for dtype in [
        SparseIndexCacheDtype::Bf16,
        SparseIndexCacheDtype::Fp8E4m3Scaled,
    ] {
        for slotted in [false, true] {
            let index = SparseIndexCacheConfig {
                tokens_per_pool: 4,
                head_dim: 128,
                dtype,
            };
            let plan = slotted.then_some(TailSlotPlan {
                lag_blocks: 2,
                sequences: 1,
            });
            let config = KvCacheConfig {
                block_size: 16,
                num_kv_heads: 1,
                head_dim: 128,
                num_layers: 3,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            };
            let per_layer = if dtype == SparseIndexCacheDtype::Bf16 {
                2
            } else {
                3
            };
            let allocations = usize::from(slotted) + per_layer * config.num_layers;
            for fail_after in 0..allocations {
                let gpu = MockGpuBackend::new();
                let unrelated = gpu.alloc(16).unwrap();
                let mut cache =
                    PagedKvCache::new_with_v_alias(config.clone(), 4, &gpu, true).unwrap();
                let before = gpu.alloc_count();
                gpu.fail_alloc_after(fail_after);
                let error = cache
                    .attach_sparse_index_with_tail_slots(index, plan, &gpu)
                    .unwrap_err();
                assert!(error.to_string().contains("mock allocation failed"));
                assert_eq!(
                    gpu.alloc_count(),
                    before,
                    "dtype={dtype:?} slotted={slotted} fail_after={fail_after}"
                );
                assert_eq!(gpu.invalid_free_count(), 0);
                assert!(gpu.read_alloc(unrelated).is_some());
                assert!(cache.sparse_index_config().is_none());
                assert!(cache.sparse_index_tail_map_ptr().is_null());
                for layer in 0..config.num_layers {
                    assert!(cache.sparse_index_pool_ptr(layer).is_null());
                    assert!(cache.sparse_index_scale_pool_ptr(layer).is_null());
                    assert!(cache.sparse_index_tail_pool_ptr(layer).is_null());
                }
                cache
                    .attach_sparse_index_with_tail_slots(index, plan, &gpu)
                    .unwrap();
                cache.release(&gpu).unwrap();
                assert_eq!(gpu.alloc_count(), 1);
                assert_eq!(gpu.invalid_free_count(), 0);
                gpu.free(unrelated).unwrap();
            }
        }
    }
}
