// SPDX-License-Identifier: AGPL-3.0-only

//! FP8 block twins on the recording mock: each rank's twin is the matching
//! slice of the checkpoint's values and scale grid, and only an E4M3 + F32
//! 128 x 128 grid counts as FP8 block.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

fn upload(gpu: &MockGpuBackend, bytes: &[u8]) -> DevicePtr {
    let p = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(bytes, p).unwrap();
    p
}

fn read(gpu: &MockGpuBackend, p: DevicePtr, len: usize) -> Vec<u8> {
    let mut b = vec![0u8; len];
    gpu.copy_d2h(p, &mut b).unwrap();
    b
}

const N: usize = 256;
const K: usize = 512;

/// `s.weight` E4M3 `[256, 512]` + F32 `scale_shape`, distinct bytes / scales.
fn store(gpu: &MockGpuBackend, scale_shape: Vec<usize>) -> (WeightStore, Vec<u8>, Vec<f32>) {
    let w: Vec<u8> = (0..N * K).map(|i| (i * 7 + i / K) as u8 & 0x7E).collect();
    let s: Vec<f32> = (0..(N / 128) * (K / 128)).map(|i| 0.5 + i as f32).collect();
    let sb: Vec<u8> = s.iter().flat_map(|f| f.to_le_bytes()).collect();
    let map = HashMap::from([
        (
            "s.weight".to_string(),
            WeightTensor {
                ptr: upload(gpu, &w),
                shape: vec![N, K],
                dtype: WeightDtype::FP8E4M3,
            },
        ),
        (
            "s.weight_scale".to_string(),
            WeightTensor {
                ptr: upload(gpu, &sb),
                shape: scale_shape,
                dtype: WeightDtype::FP32,
            },
        ),
    ]);
    (WeightStore::from_map(map), w, s)
}

fn f32s(b: &[u8]) -> Vec<f32> {
    b.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn both_grid_spellings_are_fp8_block() {
    for shape in [vec![2, 4], vec![2, 1, 4, 1]] {
        let gpu = MockGpuBackend::new();
        let (st, ..) = store(&gpu, shape);
        let raw = raw_fp8_block(&st, "s").unwrap().unwrap();
        assert_eq!((raw.n, raw.k), (N, K));
    }
}

#[test]
fn other_scale_grids_are_not_fp8_block() {
    // A 64-row block, a per-row vector, a scalar: not this format.
    for shape in [vec![4, 4], vec![256], vec![]] {
        let gpu = MockGpuBackend::new();
        let (st, ..) = store(&gpu, shape);
        assert!(raw_fp8_block(&st, "s").unwrap().is_none());
    }
    let gpu = MockGpuBackend::new();
    let (st, ..) = store(&gpu, vec![2, 4]);
    assert!(raw_fp8_block(&st, "absent").unwrap().is_none());
}

#[test]
fn each_ranks_twin_is_its_slice_of_the_checkpoint() {
    for rank in 0..2 {
        let gpu = MockGpuBackend::new();
        let (st, w, s) = store(&gpu, vec![2, 1, 4, 1]);
        let raw = raw_fp8_block(&st, "s").unwrap().unwrap();
        let stream = gpu.default_stream();

        let (t, n, k) =
            fp8_block_twin(&gpu, raw, TpShardKind::Replicated, rank, 2, stream).unwrap();
        assert_eq!((n, k), (N, K));
        assert_eq!(read(&gpu, t.data, N * K), w);
        assert_eq!(f32s(&read(&gpu, t.scales, s.len() * 4)), s);

        // Column-parallel: the rank's 128 rows and their grid row.
        let (t, n, k) =
            fp8_block_twin(&gpu, raw, TpShardKind::ColumnParallel, rank, 2, stream).unwrap();
        assert_eq!((n, k), (N / 2, K));
        assert_eq!(
            read(&gpu, t.data, N / 2 * K),
            w[rank * N / 2 * K..(rank + 1) * N / 2 * K]
        );
        assert_eq!(f32s(&read(&gpu, t.scales, 16)), s[rank * 4..rank * 4 + 4]);

        // Row-parallel: the rank's 256 columns of every row, 2 grid columns.
        let (t, n, k) =
            fp8_block_twin(&gpu, raw, TpShardKind::RowParallel, rank, 2, stream).unwrap();
        assert_eq!((n, k), (N, K / 2));
        let want: Vec<u8> = (0..N)
            .flat_map(|r| w[r * K + rank * K / 2..r * K + (rank + 1) * K / 2].to_vec())
            .collect();
        assert_eq!(read(&gpu, t.data, N * K / 2), want);
        let want_s: Vec<f32> = (0..2)
            .flat_map(|r| s[r * 4 + rank * 2..r * 4 + rank * 2 + 2].to_vec())
            .collect();
        assert_eq!(f32s(&read(&gpu, t.scales, 16)), want_s);
        // The checkpoint bytes stay for the loader's release.
        assert!(!st.was_reclaimed("s.weight"));
    }
}

#[test]
fn a_rank_boundary_off_the_grid_is_refused_with_nothing_left() {
    let gpu = MockGpuBackend::new();
    let (st, ..) = store(&gpu, vec![2, 4]);
    let raw = raw_fp8_block(&st, "s").unwrap().unwrap();
    let live = gpu.alloc_count();
    // 256 rows over 4 ranks is 64 rows a rank: half a block.
    let stream = gpu.default_stream();
    assert!(fp8_block_twin(&gpu, raw, TpShardKind::ColumnParallel, 0, 4, stream).is_err());
    assert!(fp8_block_twin(&gpu, raw, TpShardKind::RowParallel, 0, 8, stream).is_err());
    assert_eq!(gpu.alloc_count(), live);
}

#[test]
fn host_reference_applies_each_block_scale() {
    // One row of 256 columns: e4m3 1.0 (0x38) everywhere, scales 2 and 3.
    let data = vec![0x38u8; 128 * 256];
    let scales = [2.0f32, 3.0];
    let x = vec![1.0f32; 256];
    let y = fp8_block_gemm_host(&x, &data, &scales, 1, 128, 256);
    assert!(y.iter().all(|&v| v == 128.0 * 2.0 + 128.0 * 3.0));
}
