// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn cache_dispatch_uses_only_validated_m64_handle_grid_without_activation_allocation() {
    for (n, k) in [(2048, 4096), (4096, 2048)] {
        for rows in [1, 4, 5, 16, 63, 64, 65, 148, 1024] {
            let gpu = MockGpuBackend::new();
            launch_cached_m64(
                &gpu,
                KernelHandle(777),
                DevicePtr(16),
                DevicePtr(32),
                DevicePtr(48),
                rows,
                n,
                k,
                0,
            )
            .unwrap();
            assert_eq!(
                gpu.alloc_count(),
                0,
                "cache dispatch must not allocate generic FP8 activation scratch"
            );
            assert_eq!(gpu.launch_count(), 1);
            let launches = gpu.launches_snapshot();
            assert_eq!(launches[0].func, 777);
            assert_eq!(launches[0].grid, [n.div_ceil(128), rows.div_ceil(64), 1]);
            assert_eq!(launches[0].block, [128, 1, 1]);
        }
    }
}

#[test]
fn cache_dispatch_rejects_unvalidated_geometry_before_any_work() {
    for (handle, rows, n, k) in [
        (0, 5, 2048, 4096),
        (1, 0, 2048, 4096),
        (1, 1025, 2048, 4096),
        (1, 5, 4096, 4096),
        (1, 5, 2048, 2048),
    ] {
        let gpu = MockGpuBackend::new();
        assert!(
            launch_cached_m64(
                &gpu,
                KernelHandle(handle),
                DevicePtr(16),
                DevicePtr(32),
                DevicePtr(48),
                rows,
                n,
                k,
                0
            )
            .is_err()
        );
        assert_eq!((gpu.alloc_count(), gpu.launch_count()), (0, 0));
    }
}

#[test]
fn resident_output_oracle_uses_both_launches_and_leaves_candidate_without_allocation() {
    let gpu = MockGpuBackend::new();
    let output = gpu.alloc(8).unwrap();
    let expected = [0, 0x3f, 0x80, 0xbf, 0, 0, 0, 0x80];
    let launches = std::cell::Cell::new(0);
    let write = || {
        launches.set(launches.get() + 1);
        gpu.copy_h2d(&expected, output)
    };
    verify_output(&gpu, output, 8, 0, write, write).unwrap();
    assert_eq!(launches.get(), 2);
    assert_eq!(gpu.alloc_count(), 1);
    let mut actual = [0; 8];
    gpu.copy_d2h(output, &mut actual).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn resident_output_oracle_rejects_missing_writes_nonfinite_and_last_byte_difference() {
    let gpu = MockGpuBackend::new();
    let output = gpu.alloc(8).unwrap();
    let good = [0, 0x3f, 0x80, 0xbf, 0, 0, 0, 0x80];
    let old = || gpu.copy_h2d(&good, output);
    assert!(verify_output(&gpu, output, 8, 0, old, || Ok(())).is_err());
    assert!(verify_output(&gpu, output, 8, 0, || Ok(()), old).is_err());
    let mut bad = good;
    bad[7] = 0;
    assert!(verify_output(&gpu, output, 8, 0, old, || gpu.copy_h2d(&bad, output)).is_err());
    assert!(
        verify_output(&gpu, output, 8, 0, old, || anyhow::bail!(
            "injected launch failure"
        ))
        .is_err()
    );
}

#[test]
fn resident_output_oracle_rejects_unbounded_or_malformed_before_launch() {
    let gpu = MockGpuBackend::new();
    let output = gpu.alloc(8).unwrap();
    for (ptr, bytes) in [
        (output, 0),
        (output, 3),
        (output, 40962),
        (DevicePtr::NULL, 8),
    ] {
        let called = std::cell::Cell::new(false);
        assert!(
            verify_output(
                &gpu,
                ptr,
                bytes,
                0,
                || {
                    called.set(true);
                    Ok(())
                },
                || Ok(())
            )
            .is_err()
        );
        assert!(!called.get());
    }
}
