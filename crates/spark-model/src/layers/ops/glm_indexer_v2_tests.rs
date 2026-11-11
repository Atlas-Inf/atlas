// SPDX-License-Identifier: AGPL-3.0-only
//! `glm_index_logits_bf16_mma_v2` launch contract.
use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

fn v2(rows: u32, stride: u32) -> Result<IndexLogitsLaunch> {
    let (tile, pools) = (GLM_INDEX_LOGITS_V2_ROWS, GLM_INDEX_LOGITS_V2_POOLS);
    index_logits_launch(256, rows, 4096, stride, 32, 128, 4, 16, tile, pools)
}

#[test]
fn v2_grid_width_trades_waves_against_query_reloads() -> Result<()> {
    // 1024 row tiles already fill the waves: each CTA walks half the history.
    assert_eq!(v2(4096, 15361)?.grid, [2, 1024, 1]);
    // Fewer rows split the history further, down to 16 chunks per CTA,
    // or fewer when that would leave less than one wave.
    assert_eq!(v2(512, 16500)?.grid, [9, 128, 1]);
    assert_eq!(v2(64, 16400)?.grid, [32, 16, 1]);
    assert_eq!(v2(8, 16385)?.grid, [72, 2, 1]);
    let launch = v2(9, 33)?;
    assert_eq!(launch.grid, [2, 3, 1]);
    assert_eq!((launch.block, launch.shared_mem), ([128, 1, 1], 32_768));
    Ok(())
}

/// The launch geometry must match the kernel's compile-time tile: with any
/// other block size the kernel does no work, leaving stale logits for top-K.
#[test]
fn v2_launch_constants_match_the_kernel_source() {
    let source = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../kernels/gb10/glm-5.3-flash/nvfp4/glm_indexer_wmma.cu"
    ));
    for declaration in [
        format!("kIndexV2Warps = {GLM_INDEX_LOGITS_V2_ROWS};"),
        format!("kIndexV2Pools = {V2_CHUNK_POOLS};"),
        // index_v2_smem_bytes, which index_logits_launch's shared_mem mirrors.
        "return 2 * Pools * 256 + Warps * 32 * 32 * 4;".to_string(),
    ] {
        assert!(
            source.contains(&declaration),
            "glm_indexer_wmma.cu no longer has `{declaration}`"
        );
    }
}

/// v2 stages keys with 16-byte `cp.async`: an unaligned cache base or block
/// stride must fail before anything is submitted.
#[test]
fn v2_rejects_unaligned_key_cache_before_launching() {
    let gpu = MockGpuBackend::new();
    let launch = |cache: u64, stride: u64, rows_per_cta: u32, pools_per_cta: u32| {
        let (query, other) = (DevicePtr(256), DevicePtr(0));
        glm_index_logits(
            &gpu,
            KernelHandle(1),
            query,
            other,
            DevicePtr(cache),
            other,
            other,
            9,
            4096,
            1027,
            32,
            128,
            4,
            16,
            stride,
            rows_per_cta,
            pools_per_cta,
            0,
        )
    };
    let (tile, pools) = (GLM_INDEX_LOGITS_V2_ROWS, GLM_INDEX_LOGITS_V2_POOLS);
    for (cache, stride) in [(0x108, 1024), (0x100, 1032), (0x104, 1028)] {
        assert!(
            launch(cache, stride, tile, pools).is_err(),
            "{cache:#x} {stride}"
        );
    }
    assert_eq!(gpu.launch_count(), 0);
    launch(0x100, 1024, tile, pools).unwrap();
    // The WMMA scorer reads keys element by element and keeps accepting them.
    launch(0x108, 1032, 8, 32).unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 2);
    assert_eq!(
        (launches[0].grid, launches[0].block),
        ([33, 3, 1], [128, 1, 1])
    );
}
