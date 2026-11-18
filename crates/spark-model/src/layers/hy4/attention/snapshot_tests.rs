// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

#[test]
fn snapshots_restore_overwritten_window_and_partial_dsa_group_bytes_and_cursors() {
    for layer in [0, 2] {
        let gpu = MockGpuBackend::new();
        let c = config(0, 1);
        let store = store(&gpu, &c, layer);
        let a = Hy4Attention::load(&store, layer, &c, &gpu).unwrap();
        let mut cache = AttentionCache::new(&a, 1024, 4).unwrap();
        cache.len = 515;
        cache.first_group_position = Some([512; 3]);
        cache.stream = Some(17);
        let buffers = [
            (cache.latent, cache.latent_bytes),
            (cache.pool, cache.pool_bytes),
            (
                cache.tail,
                if cache.pool_bytes > 0 { 4 * 128 * 2 } else { 0 },
            ),
        ];
        for (ptr, len) in buffers {
            if len > 0 {
                gpu.copy_h2d(&vec![31; len], ptr).unwrap();
            }
        }
        let snapshot = cache.fork(17).unwrap();
        for (ptr, len) in buffers {
            if len > 0 {
                gpu.copy_h2d(&vec![79; len], ptr).unwrap();
            }
        }
        cache.len = 521;
        cache.first_group_position = Some([520; 3]);
        cache.copy_state_from(&snapshot, 17).unwrap();
        assert_eq!(cache.len, 515);
        assert_eq!(cache.first_group_position, Some([512; 3]));
        for (ptr, len) in buffers {
            if len > 0 {
                let mut bytes = vec![0; len];
                gpu.copy_d2h(ptr, &mut bytes).unwrap();
                assert!(bytes.iter().all(|b| *b == 31));
            }
        }
        assert!(cache.copy_state_from(&snapshot, 18).is_err());
        assert_eq!(cache.len, 515);
    }
}

#[test]
fn dfly_alias_requires_live_contiguous_dsa_cache_on_its_stream() {
    let c = config(0, 1);
    let swa = c
        .hy4
        .as_ref()
        .unwrap()
        .layer_types
        .iter()
        .position(|kind| *kind == Kind::SlidingAttention)
        .unwrap();
    let dsa = c
        .hy4
        .as_ref()
        .unwrap()
        .layer_types
        .iter()
        .position(|kind| *kind != Kind::SlidingAttention)
        .unwrap();
    for layer in [swa, dsa] {
        let gpu = MockGpuBackend::new();
        let c = config(0, 1);
        let store = store(&gpu, &c, layer);
        let a = Hy4Attention::load(&store, layer, &c, &gpu).unwrap();
        let mut cache = AttentionCache::new(&a, 1024, 4).unwrap();
        assert!(cache.dfly_target(17).is_err());
        cache.len = 57;
        cache.stream = Some(17);
        assert!(cache.dfly_target(18).is_err());
        if layer == dsa {
            let view = cache.dfly_target(17).unwrap();
            assert_eq!(view.latent, cache.latent);
            assert_eq!(view.indices, cache.scratch.indices);
            assert_eq!(view.len, 57);
            assert_eq!(view.capacity, 1024);
            cache.layout.table.swap(0, 1);
            assert!(cache.dfly_target(17).is_err());
            cache.layout.table.swap(0, 1);
            cache.poisoned = true;
            assert!(cache.dfly_target(17).is_err());
        } else {
            assert!(cache.dfly_target(17).is_err());
        }
    }
}
