// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

// The MoE tests load the same recording backend under their own module.
#[allow(dead_code, clippy::duplicate_mod)]
#[path = "../moe/gate_up_btile_test_gpu.rs"]
mod recording;

const TIERS: [KernelHandle; 2] = [KernelHandle(8), KernelHandle(16)];
/// W_uv: 32 heads of [n = 256, k = 512], token rows packed.
const W_UV: [u32; 5] = [32, 512, 256, 32 * 512, 32 * 256];

/// Grouped launch of `m` rows with `[g, k, n, a_stride, c_stride]`.
fn grouped(gpu: &dyn GpuBackend, m: u32, [g, k, n, a_stride, c_stride]: [u32; 5]) -> Result<()> {
    mxfp8_gemv_grouped(
        gpu,
        &TIERS,
        DevicePtr(0x100),
        DevicePtr(0x200),
        DevicePtr(0x300),
        DevicePtr(0x400),
        m,
        g,
        k,
        n,
        a_stride,
        c_stride,
        0,
    )
}

#[test]
fn grouped_gemv_launches_one_head_plane_per_z_on_the_row_tier() {
    for (m, tier) in [(1, 8), (8, 8), (9, 16), (16, 16)] {
        let gpu = MockGpuBackend::new();
        grouped(&gpu, m, W_UV).unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].func, tier, "m={m}");
        assert_eq!(launches[0].grid, [256 / 16, 1, 32]);
        assert_eq!(launches[0].block, [256, 1, 1]);
    }
}

#[test]
fn grouped_gemv_passes_the_kernel_abi_in_order() {
    use recording::{Arg, Event};
    let gpu = recording::Gpu::new();
    grouped(&gpu, 5, W_UV).unwrap();
    let ptr = |p| Arg::Ptr(DevicePtr(p));
    let u32 = |v: u32| Arg::Bytes(v.to_le_bytes().to_vec());
    // (A, W, S, C, M, N, K, lda, out_stride)
    let abi = vec![
        ptr(0x100),
        ptr(0x200),
        ptr(0x300),
        ptr(0x400),
        u32(5),
        u32(256),
        u32(512),
        u32(32 * 512),
        u32(32 * 256),
    ];
    assert_eq!(
        gpu.trace(),
        [Event::Launch(8, [16, 1, 32], [256, 1, 1], 0, 0, abi)]
    );
}

#[test]
fn grouped_gemv_rejects_unservable_shapes() {
    let gpu = MockGpuBackend::new();
    let [g, k, n, a, c] = W_UV;
    for (m, shape) in [
        (0, W_UV),
        (17, W_UV),
        (8, [g, 500, n, a, c]),
        (8, [0, k, n, a, c]),
        (8, [g, k, n, a - 8, c]),
        (8, [g, k, n, a, c - 1]),
        (8, [g, k, n, a + 4, c]),
    ] {
        assert!(grouped(&gpu, m, shape).is_err(), "m={m} {shape:?}");
    }
    assert_eq!(gpu.launch_count(), 0);
}
