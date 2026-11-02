// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use atlas_core::scope::ModelResource;

#[test]
fn failed_pool_allocation_releases_every_owned_pointer_once_and_can_retry() {
    for alias in [false, true] {
        let config = KvCacheConfig {
            num_layers: 3,
            ..test_config()
        };
        let allocations = config.num_layers * if alias { 1 } else { 2 };
        for fail_after in 0..allocations {
            let gpu = MockGpuBackend::new();
            let unrelated = gpu.alloc(16).unwrap();
            gpu.fail_alloc_after(fail_after);
            let error = PagedKvCache::new_with_v_alias(config.clone(), 2, &gpu, alias)
                .err()
                .expect("injected allocation must fail");
            assert!(error.to_string().contains("mock allocation failed"));
            assert_eq!(
                gpu.alloc_count(),
                1,
                "leaked pool: alias={alias}, fail_after={fail_after}"
            );
            assert_eq!(
                gpu.invalid_free_count(),
                0,
                "an aliased K/V pointer must only be freed once"
            );
            assert!(gpu.read_alloc(unrelated).is_some());
            let mut cache = PagedKvCache::new_with_v_alias(config.clone(), 2, &gpu, alias).unwrap();
            cache.release(&gpu).unwrap();
            assert_eq!(gpu.alloc_count(), 1);
            assert_eq!(gpu.invalid_free_count(), 0);
            gpu.free(unrelated).unwrap();
        }
    }
}
