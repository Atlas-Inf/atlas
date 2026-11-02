// SPDX-License-Identifier: AGPL-3.0-only
//! Backend ABI regression: preserve current main's CUDA and HIP selector contracts.
use sha2::{Digest, Sha256};
use spark_model::layers::ops;
#[path = "dflash2_selector_abi/recording_gpu.rs"]
mod recording_gpu;
use recording_gpu::RecordingGpu;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

#[test]
fn hip_kernel_is_the_unmodified_upstream_implementation() {
    let source = include_bytes!("../../../kernels/strix-hip/common/dflash2_candidate_selector.cu");
    // Exact blob from Atlas e29809db; changing the AMD algorithm needs its own qualification.
    assert_eq!(
        format!("{:x}", Sha256::digest(source)),
        "89ba53435de29dabb1bebe3428a2eece9348e2a9c7a6e07b3efbb6e7ce29a9d1"
    );
}

#[test]
fn single_selector_dispatch_matches_the_backend_abi_without_host_readback() {
    assert_eq!(ops::DFLASH2_SELECTOR_SYMBOL, "dflash2_candidate_selector");
    let gpu = RecordingGpu::default();
    // Both backends read the anchor and ban depth on-device during replay.
    ops::dflash2_candidate_selector(
        &gpu,
        KernelHandle(7),
        DevicePtr(1),
        DevicePtr(2),
        DevicePtr(3),
        DevicePtr(4),
        DevicePtr(5),
        DevicePtr(6),
        DevicePtr(7),
        [8, 9, 10, 11],
        8,
        154880,
        256,
        16,
        DevicePtr(12),
        0,
    )
    .unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    let launch = &launches[0];
    assert_eq!(launch.func, 7);
    assert_eq!(
        (launch.grid, launch.block, launch.args),
        ([6, 8, 1], [512, 1, 1], 16)
    );
    assert_eq!(ops::DFLASH2_SELECTOR_BANS_END_TOKENS, launch.args == 16);
    let expected = vec![
        (true, 1),
        (true, 2),
        (true, 3),
        (true, 4),
        (true, 5),
        (true, 6),
        (true, 7),
        (false, 8),
        (false, 154880),
        (false, 256),
        (false, 16),
        (false, 8),
        (false, 9),
        (false, 10),
        (false, 11),
        (true, 12),
    ];
    assert_eq!(*gpu.arguments.lock().unwrap(), vec![expected]);
    assert_eq!(gpu.sync_count(), 0);
}

#[test]
fn batched_selector_preserves_upstream_launch_contract() {
    for count in [1, 4, 16] {
        let gpu = RecordingGpu::default();
        ops::dflash2_candidate_selector_batched(
            &gpu,
            KernelHandle(9),
            DevicePtr(1),
            DevicePtr(2),
            DevicePtr(3),
            DevicePtr(4),
            DevicePtr(5),
            DevicePtr(6),
            count,
            8,
            154880,
            256,
            16,
            0,
        )
        .unwrap();
        let launches = gpu.launches_snapshot();
        assert_eq!(launches.len(), 1);
        assert_eq!(
            (launches[0].grid, launches[0].block, launches[0].args),
            ([count, 1, 1], [1024, 1, 1], 10)
        );
        assert_eq!(
            *gpu.arguments.lock().unwrap(),
            vec![vec![
                (true, 1),
                (true, 2),
                (true, 3),
                (true, 4),
                (true, 5),
                (true, 6),
                (false, 8),
                (false, 154880),
                (false, 256),
                (false, 16)
            ]]
        );
        assert_eq!(gpu.sync_count(), 0);
    }
}
