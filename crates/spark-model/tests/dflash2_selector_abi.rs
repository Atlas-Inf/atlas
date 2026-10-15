// SPDX-License-Identifier: AGPL-3.0-only
//! Backend ABI regression: HIP retains the upstream one-CTA selector body.
use sha2::{Digest, Sha256};
use spark_model::layers::ops;
#[path = "dflash2_selector_abi/recording_gpu.rs"]
mod recording_gpu;
use recording_gpu::RecordingGpu;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

#[test]
fn hip_kernel_is_the_unmodified_upstream_implementation() {
    let source = include_bytes!("../../../kernels/strix-hip/common/dflash2_candidate_selector.cu");
    // Exact blob from Atlas 7ffd183f; changing the AMD algorithm needs its own qualification.
    assert_eq!(
        format!("{:x}", Sha256::digest(source)),
        "b3bf2613f70f2b3c72ba9d6e630576cd5b8f2f478f2f10b3030e6361d9ed4c34"
    );
}

#[test]
fn single_selector_dispatch_matches_the_backend_abi_without_host_readback() {
    assert_eq!(
        ops::DFLASH2_SELECTOR_SYMBOL,
        if cfg!(atlas_hip) {
            "dflash2_candidate_selector_batched"
        } else {
            "dflash2_candidate_selector"
        }
    );
    assert_eq!(ops::DFLASH2_SELECTOR_BANS_END_TOKENS, !cfg!(atlas_hip));
    let gpu = RecordingGpu::default();
    // A nonzero ban must remain a valid request on HIP. Its upstream drafter
    // does not mask EOS; target verification still enforces the minimum.
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
    if cfg!(atlas_hip) {
        assert_eq!(
            (launch.grid, launch.block, launch.args),
            ([1, 1, 1], [1024, 1, 1], 10)
        );
    } else {
        assert_eq!(
            (launch.grid, launch.block, launch.args),
            ([12, 8, 1], [512, 1, 1], 16)
        );
    }
    let expected = if cfg!(atlas_hip) {
        vec![
            (true, 1),
            (true, 2),
            (true, 3),
            (true, 4),
            (true, 5),
            (true, 6),
            (false, 8),
            (false, 154880),
            (false, 256),
            (false, 16),
        ]
    } else {
        vec![
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
        ]
    };
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
