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

const SWIGLU: [KernelHandle; 3] = [KernelHandle(81), KernelHandle(82), KernelHandle(83)];

fn swiglu(gpu: &dyn GpuBackend, rows: usize) -> Result<()> {
    let w = crate::weight_map::Mxfp8Weight {
        data: DevicePtr(0x200),
        scales: DevicePtr(0x300),
        tiled: false,
    };
    mxfp8_swiglu_rows(
        gpu,
        &SWIGLU,
        &w,
        DevicePtr(0x100),
        DevicePtr(0x400),
        rows,
        320,
        2560,
        0,
    )
}

#[test]
fn swiglu_rows_run_32_row_launches_on_the_row_tier() {
    for (rows, want) in [
        (1usize, vec![81u64]),
        (9, vec![82]),
        (32, vec![83]),
        (40, vec![83, 81]),
    ] {
        let gpu = MockGpuBackend::new();
        swiglu(&gpu, rows).unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.iter().map(|l| l.func).collect::<Vec<_>>(), want);
        // A CTA writes 8 outputs of the [rows, inter] activation.
        assert!(launches.iter().all(|l| l.grid == [320 / 8, 1, 1]));
    }
}

#[test]
fn tiled_switch_is_explicit() {
    assert!(!parse_tiled(None).unwrap());
    assert!(!parse_tiled(Some("0")).unwrap());
    assert!(parse_tiled(Some("1")).unwrap());
    for value in ["", "true", "2", " 1"] {
        assert!(parse_tiled(Some(value)).is_err(), "{value:?}");
    }
}

#[test]
fn swiglu_rows_pass_the_kernel_abi_in_order() {
    use recording::{Arg, Event};
    let gpu = recording::Gpu::new();
    swiglu(&gpu, 40).unwrap();
    let ptr = |p| Arg::Ptr(DevicePtr(p));
    let u32 = |v: u32| Arg::Bytes(v.to_le_bytes().to_vec());
    // (A, W, S, C, M, N = inter, K, out_stride = inter); rows 32.. second.
    let abi = |a: u64, c: u64, m: u32| {
        vec![
            ptr(a),
            ptr(0x200),
            ptr(0x300),
            ptr(c),
            u32(m),
            u32(320),
            u32(2560),
            u32(320),
        ]
    };
    assert_eq!(
        gpu.trace(),
        [
            Event::Launch(83, [40, 1, 1], [256, 1, 1], 0, 0, abi(0x100, 0x400, 32)),
            Event::Launch(
                81,
                [40, 1, 1],
                [256, 1, 1],
                0,
                0,
                abi(0x100 + 32 * 2560 * 2, 0x400 + 32 * 320 * 2, 8)
            ),
        ]
    );
}

#[test]
fn swiglu_rows_refuse_a_k_off_the_scale_grid() {
    let gpu = MockGpuBackend::new();
    let w = crate::weight_map::Mxfp8Weight {
        data: DevicePtr(0x200),
        scales: DevicePtr(0x300),
        tiled: false,
    };
    let r = mxfp8_swiglu_rows(
        &gpu,
        &SWIGLU,
        &w,
        DevicePtr(1),
        DevicePtr(2),
        4,
        320,
        2550,
        0,
    );
    assert!(r.is_err());
    assert_eq!(gpu.launch_count(), 0);
}

#[test]
fn plain_tiers_split_at_8_and_16_rows() {
    for (m, tier) in [(1, 0), (8, 0), (9, 1), (16, 1), (17, 2), (32, 2)] {
        assert_eq!(mxfp8_tier(m), tier, "m={m}");
    }
}
