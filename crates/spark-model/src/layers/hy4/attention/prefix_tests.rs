// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

fn append(cache: &mut AttentionCache<'_, '_>, gpu: &MockGpuBackend, value: u8) {
    let p = cache.len;
    gpu.copy_h2d(
        &vec![value; 1152],
        cache.latent.offset(cache.layout.slot(p).unwrap() * 1152),
    )
    .unwrap();
    if cache.pool_bytes > 0 {
        if p.is_multiple_of(4) {
            cache.first_group_position = Some([p as i64; 3]);
        }
        gpu.copy_h2d(&vec![value; 256], cache.tail.offset(p % 4 * 256))
            .unwrap();
        if p % 4 == 3 {
            let group = p / 4;
            gpu.copy_h2d(
                &[value; 128],
                cache.pool.offset(group / 64 * 8448 + group % 64 * 128),
            )
            .unwrap();
            gpu.copy_h2d(
                &[value; 4],
                cache.pool.offset(group / 64 * 8448 + 8192 + group % 64 * 4),
            )
            .unwrap();
        }
        gpu.copy_h2d(&vec![value; 8192], cache.scratch.indices)
            .unwrap();
    }
    cache.len += 1;
    cache.record_prefix(17).unwrap();
}

fn bytes(gpu: &MockGpuBackend, ptr: spark_runtime::gpu::DevicePtr, len: usize) -> Vec<u8> {
    let mut out = vec![0; len];
    if len > 0 {
        gpu.copy_d2h(ptr, &mut out).unwrap();
    }
    out
}

#[test]
fn exact_prefix_restores_rejected_window_rows_pool_tail_indices_and_positions() {
    for layer in [0, 2] {
        for base in [57, 255, 515] {
            for accepted in 1..=6 {
                let gpu = MockGpuBackend::new();
                let c = config(0, 1);
                let store = store(&gpu, &c, layer);
                let a = Hy4Attention::load(&store, layer, &c, &gpu).unwrap();
                let mut cache = AttentionCache::new(&a, 1024, 4).unwrap();
                cache.len = base;
                cache.stream = Some(17);
                cache.first_group_position = Some([(base / 4 * 4) as i64; 3]);
                for (ptr, size) in [
                    (cache.latent, cache.latent_bytes),
                    (cache.pool, cache.pool_bytes),
                    (cache.tail, if cache.pool_bytes > 0 { 1024 } else { 0 }),
                ] {
                    if size > 0 {
                        gpu.copy_h2d(&vec![31; size], ptr).unwrap();
                    }
                }
                let checkpoint = cache.fork(17).unwrap();
                assert!(cache.commit_prefix(&checkpoint, 1, 6, 17).is_err());
                assert!(cache.begin_prefix(0, 17).is_err());
                assert!(cache.begin_prefix(6, 18).is_err());
                cache.begin_prefix(6, 17).unwrap();
                let mut expected = None;
                let mut indices = Vec::new();
                for row in 0..6 {
                    append(&mut cache, &gpu, row as u8 + 70);
                    if row + 1 == accepted {
                        expected = Some(cache.fork(17).unwrap());
                        if cache.pool_bytes > 0 {
                            indices = bytes(&gpu, cache.scratch.indices, 8192);
                        }
                    }
                }
                cache.commit_prefix(&checkpoint, accepted, 6, 17).unwrap();
                let expected = expected.unwrap();
                assert_eq!(cache.len, base + accepted);
                assert_eq!(cache.first_group_position, expected.first_group_position);
                for (actual, expected, size) in [
                    (cache.latent, expected.latent, cache.latent_bytes),
                    (cache.pool, expected.pool, cache.pool_bytes),
                    (
                        cache.tail,
                        expected.tail,
                        if cache.pool_bytes > 0 { 1024 } else { 0 },
                    ),
                ] {
                    assert_eq!(bytes(&gpu, actual, size), bytes(&gpu, expected, size));
                }
                if cache.pool_bytes > 0 {
                    assert_eq!(bytes(&gpu, cache.scratch.indices, 8192), indices);
                }
                assert!(cache.commit_prefix(&checkpoint, 0, 6, 17).is_err());
                assert!(cache.commit_prefix(&checkpoint, 7, 6, 17).is_err());
                assert!(cache.commit_prefix(&checkpoint, accepted, 6, 18).is_err());
            }
        }
    }
}
