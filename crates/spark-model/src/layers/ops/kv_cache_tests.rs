// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

const Q_HEADS: u32 = 64;

fn launch_nvfp4(gpu: &MockGpuBackend, num_seqs: u32) -> Result<()> {
    let p = DevicePtr(0x1000);
    mla_paged_decode_nvfp4(
        gpu,
        KernelHandle(7),
        p,
        p,
        p,
        p,
        p,
        p,
        4,
        Q_HEADS,
        1,
        512,
        576,
        16,
        0.044,
        0,
        0,
        num_seqs,
        0,
    )
}

fn launch_fp8(gpu: &MockGpuBackend, num_seqs: u32) -> Result<()> {
    let p = DevicePtr(0x1000);
    mla_paged_decode_fp8(
        gpu,
        KernelHandle(7),
        p,
        p,
        p,
        p,
        p,
        p,
        4,
        Q_HEADS,
        1,
        512,
        576,
        16,
        0.044,
        1.0,
        1.0,
        576,
        num_seqs,
        0,
        DevicePtr::NULL,
        DevicePtr::NULL,
        0,
        0,
    )
}

/// The MLA paged-decode kernels index Q and O by head only, so a second
/// sequence row in the grid would write the same O elements as the first.
#[test]
fn mla_paged_decode_refuses_more_than_one_sequence() {
    for launch in [launch_nvfp4, launch_fp8] {
        for num_seqs in [0, 2, 5] {
            let gpu = MockGpuBackend::new();
            let err = launch(&gpu, num_seqs).unwrap_err().to_string();
            assert!(err.contains("num_seqs"), "{err}");
            assert!(gpu.launches_snapshot().is_empty());
        }
    }
}

#[test]
fn mla_paged_decode_single_sequence_keeps_its_grid() {
    for launch in [launch_nvfp4, launch_fp8] {
        let gpu = MockGpuBackend::new();
        launch(&gpu, 1).unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].func, 7);
        assert_eq!(launches[0].grid, [Q_HEADS, 1, 1]);
        assert_eq!(launches[0].block, [256, 1, 1]);
    }
}
