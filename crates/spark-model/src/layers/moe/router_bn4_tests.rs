// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
#[test]
fn router_boundaries_are_independent() {
    let a = DevicePtr(0x1000);
    let b = DevicePtr(0x100000);
    let c = DevicePtr(0x400000);
    assert_eq!(checked_output(a, b, c, c, 2880).unwrap(), 2880);
    assert!(checked_output(a, b, c, c, 2879).is_err());
    assert!(checked_output(a, b, c, DevicePtr(c.0 + 2), 2880).is_err());
    assert!(checked_output(a, b, a, a, 2880).is_err());
    assert!(checked_output(a, DevicePtr(u64::MAX - 7), c, c, 2880).is_err());
}
#[test]
fn router_candidate_has_distinct_actual_launch_geometry() {
    use spark_runtime::gpu::mock::MockGpuBackend;
    let gpu = MockGpuBackend::new();
    ops::glm_router_bn4(
        &gpu,
        KernelHandle(3),
        DevicePtr(4096),
        DevicePtr(8192),
        DevicePtr(16384),
        7,
    )
    .unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    assert_eq!(launches[0].grid, [72, 1, 1]);
    assert_eq!(launches[0].block, [32, 1, 1]);
    ops::dense_gemm_router_m5(
        &gpu,
        KernelHandle(4),
        DevicePtr(4096),
        &DenseWeight {
            weight: DevicePtr(8192),
        },
        DevicePtr(16384),
        5,
        288,
        4096,
        7,
    )
    .unwrap();
    let old = gpu.launches_snapshot();
    assert_eq!(old[1].grid, [18, 1, 1]);
    assert_eq!(old[1].block, [16, 5, 1]);
    assert!(
        ops::glm_router_bn4(
            &gpu,
            KernelHandle(3),
            DevicePtr(4098),
            DevicePtr(8192),
            DevicePtr(16384),
            7
        )
        .is_err()
    );
    assert_eq!(gpu.launch_count(), 2, "reject before launch");
}
