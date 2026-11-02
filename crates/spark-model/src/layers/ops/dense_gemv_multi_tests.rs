// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use spark_runtime::gpu::mock::{MockGpuBackend, MockLaunch};

/// The one launch `dense_gemv_batchm_dual` issues for eight rows of N=4096.
fn batchm_dual_launch(k: u32) -> MockLaunch {
    let gpu = MockGpuBackend::new();
    let weights = [DevicePtr(0x100), DevicePtr(0x200)].map(|weight| DenseWeight { weight });
    dense_gemv_batchm_dual(
        &gpu,
        KernelHandle(7),
        [DevicePtr(0x300), DevicePtr(0x400)],
        [&weights[0], &weights[1]],
        [DevicePtr(0x500), DevicePtr(0x600)],
        8,
        4096,
        k,
        0,
    )
    .unwrap();
    let launches = gpu.launches_snapshot();
    assert_eq!(launches.len(), 1);
    launches[0].clone()
}

#[test]
fn kda_gate_pair_at_k128_uses_the_short_k_dual_tier() {
    // MockGpuBackend::kernel resolves every symbol to 0xDEAD; the caller's
    // generic handle is 7.
    let launch = batchm_dual_launch(128);
    assert_eq!(launch.func, 0xDEAD);
    assert_eq!(launch.grid, [4096 / 16, 1, 2]);
    assert_eq!(launch.block, [256, 1, 1]);
}

#[test]
fn other_k_keeps_the_generic_dual() {
    let launch = batchm_dual_launch(4096);
    assert_eq!(launch.func, 7);
    assert_eq!(launch.grid, [4096 / 4, 1, 2]);
    assert_eq!(launch.block, [256, 1, 1]);
}

#[test]
fn prefill_triple_walks_plane_tiles_along_x_over_128_row_tiles() {
    let gpu = MockGpuBackend::new();
    let w = DenseWeight {
        weight: DevicePtr(0x1000),
    };
    for (m, n, grid) in [
        (8196, [32, 128], [3, 65, 1]),
        (3692, [32, 128], [3, 29, 1]),
        (2, [32, 128], [3, 1, 1]),
        (256, [200, 300], [2 + 2 * 3, 2, 1]),
    ] {
        dense_gemm_pipelined_triple_n(
            &gpu,
            KernelHandle(7),
            DevicePtr(0x2000),
            [&w; 3],
            [DevicePtr(0x3000); 3],
            m,
            n,
            4096,
            0,
        )
        .unwrap();
        let launch = gpu.launches_snapshot().pop().unwrap();
        assert_eq!((launch.grid, launch.block), (grid, [256, 1, 1]));
    }
}

/// The fused verify triples run the load-ahead body (two CTAs per SM). The KDA
/// b / f_a / g_a grid of 96 CTAs is the widest they take; a wider one is
/// refused before launch instead of running slower than three plain launches.
#[test]
fn verify_triples_refuse_a_grid_wider_than_the_load_ahead_limit() {
    let gpu = MockGpuBackend::new();
    let w = DenseWeight {
        weight: DevicePtr(0x1000),
    };
    let (x, y) = (DevicePtr(0x2000), DevicePtr(0x3000));
    let batchm =
        |n| dense_gemv_batchm_triple_n(&gpu, KernelHandle(7), x, [&w; 3], [y; 3], 8, n, 4096, 0);
    let batch5 = |[first, other]: [u32; 2]| {
        let k = KernelHandle(9);
        dense_gemv_batch5_triple_n(&gpu, k, x, &w, &w, &w, y, y, y, first, other, 4096, 0)
    };
    for n in [[32, 128], [128, 128], [4, 4]] {
        assert!(dense_gemv_triple_fits(n[0], n[1]));
        batchm(n).unwrap();
        batch5(n).unwrap();
        let grid = gpu.launches_snapshot().pop().unwrap().grid;
        assert_eq!(grid, [n[0].max(n[1]) / 4, 1, 3]);
        assert!(grid[0] * grid[2] <= DENSE_GEMV_AHEAD_MAX_CTAS);
    }
    let launched = gpu.launch_count();
    for n in [[32, 132], [132, 32], [8192, 8192]] {
        assert!(!dense_gemv_triple_fits(n[0], n[1]));
        for refused in [batchm(n), batch5(n)] {
            let error = refused.unwrap_err().to_string();
            assert!(error.contains("96 CTAs"), "{error}");
        }
    }
    assert_eq!(gpu.launch_count(), launched);
}

/// The packed side projections' split: row r of the `[m, 32 + 128 + 128]`
/// GEMM output lands at row r of each dense plane, column for column.
#[test]
fn packed_triple_split_lands_each_column_block_in_its_plane() {
    let gpu = MockGpuBackend::new();
    let (m, widths) = (5usize, [32u32, 128, 128]);
    let n: usize = widths.iter().map(|&w| w as usize).sum();
    let src_bytes: Vec<u8> = (0..m * n * 2).map(|i| (i * 7 + i / 251) as u8).collect();
    let src = gpu.alloc(src_bytes.len()).unwrap();
    gpu.copy_h2d(&src_bytes, src).unwrap();
    let outs = widths.map(|w| gpu.alloc(m * w as usize * 2).unwrap());
    split_columns(&gpu, src, outs, m as u32, widths, 0).unwrap();
    assert_eq!(gpu.d2d_2d_count(), 3);
    let mut col = 0;
    for (out, w) in outs.into_iter().zip(widths) {
        let w = w as usize * 2;
        let mut got = vec![0u8; m * w];
        gpu.copy_d2h(out, &mut got).unwrap();
        for r in 0..m {
            let row = &src_bytes[r * n * 2 + col..r * n * 2 + col + w];
            assert_eq!(
                &got[r * w..(r + 1) * w],
                row,
                "plane at column {col}, row {r}"
            );
        }
        col += w;
    }
}
