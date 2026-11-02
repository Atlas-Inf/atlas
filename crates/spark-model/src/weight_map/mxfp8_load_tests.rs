// SPDX-License-Identifier: AGPL-3.0-only

//! The MXFP8 twins on the recording mock (real bytes, no kernels): every
//! rank's twin equals the matching slice of the full checkpoint tensor, cut
//! independently here on the host.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightTensor;

/// A deterministic byte pattern distinct per (tensor, row, col).
fn pattern(tag: u8, n: usize, k: usize) -> Vec<u8> {
    (0..n * k)
        .map(|i| (i / k).wrapping_mul(31) as u8 ^ (i % k).wrapping_mul(7) as u8 ^ tag)
        .collect()
}

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

/// A full MXFP8 tensor on the mock and its host bytes `(values, scales)`.
fn tensor(gpu: &MockGpuBackend, tag: u8, n: usize, k: usize) -> (RawMxfp8, Vec<u8>, Vec<u8>) {
    let (w, s) = (pattern(tag, n, k), pattern(!tag, n, k / 32));
    let raw = RawMxfp8 {
        data: upload(gpu, &w),
        scales: upload(gpu, &s),
        n,
        k,
    };
    (raw, w, s)
}

/// Rows `[lo, hi)` of a row-major `[_, width]` host matrix.
fn rows(m: &[u8], width: usize, lo: usize, hi: usize) -> Vec<u8> {
    m[lo * width..hi * width].to_vec()
}

#[test]
fn qkvz_twin_is_each_segment_s_rank_rows_stacked() {
    // Qwen3.8-Flash-Next GDN, scaled down 1/16 on rows and 1/20 on k:
    // in_proj_qkv = [Q 2 | K 2 | V 6] heads-rows, in_proj_z = [Z 6], k = 128.
    let (q, kk, v) = (128usize, 128usize, 384usize);
    let k = 128;
    for tp in [1usize, 2] {
        for rank in 0..tp {
            let gpu = MockGpuBackend::new();
            let (qkv, qkv_w, qkv_s) = tensor(&gpu, 0x11, q + kk + v, k);
            let (z, z_w, z_s) = tensor(&gpu, 0x5a, v, k);
            let (twin, n, got_k) =
                mxfp8_rows_twin(&gpu, &[(qkv, &[q, kk, v]), (z, &[v])], rank, tp, 0).unwrap();
            assert_eq!((n, got_k), ((q + kk + 2 * v) / tp, k));
            // Host reference: each segment's rank slice, in [Q|K|V|Z] order
            // (segment = source, first row, rows).
            let mut want_w = Vec::new();
            let mut want_s = Vec::new();
            let segments = [
                (&qkv_w, &qkv_s, 0, q),
                (&qkv_w, &qkv_s, q, kk),
                (&qkv_w, &qkv_s, q + kk, v),
                (&z_w, &z_s, 0, v),
            ];
            for (mw, ms, base, seg) in segments {
                let (lo, hi) = (base + rank * seg / tp, base + (rank + 1) * seg / tp);
                want_w.extend(rows(mw, k, lo, hi));
                want_s.extend(rows(ms, k / 32, lo, hi));
            }
            assert_eq!(read(&gpu, twin.data, n * k), want_w, "tp={tp} rank={rank}");
            assert_eq!(
                read(&gpu, twin.scales, n * k / 32),
                want_s,
                "tp={tp} rank={rank}"
            );
            assert!(!twin.tiled);
        }
    }
}

#[test]
fn out_proj_twin_is_the_rank_s_k_columns_with_their_scale_blocks() {
    // out_proj [h, value_dim] = [2560, 6144] at 1/20 rows: rank r keeps
    // columns [3072 r, 3072 (r + 1)) and scale blocks [96 r, 96 (r + 1)).
    let (n, k) = (128usize, 6144usize);
    for tp in [1usize, 2] {
        for rank in 0..tp {
            let gpu = MockGpuBackend::new();
            let (raw, w, s) = tensor(&gpu, 0x3c, n, k);
            let (twin, got_n, lk) = mxfp8_cols_twin(&gpu, raw, rank, tp, 0).unwrap();
            assert_eq!((got_n, lk), (n, k / tp));
            let (lg, g) = (lk / 32, k / 32);
            let want_w: Vec<u8> = (0..n)
                .flat_map(|r| w[r * k + rank * lk..r * k + (rank + 1) * lk].to_vec())
                .collect();
            let want_s: Vec<u8> = (0..n)
                .flat_map(|r| s[r * g + rank * lg..r * g + (rank + 1) * lg].to_vec())
                .collect();
            assert_eq!(read(&gpu, twin.data, n * lk), want_w, "tp={tp} rank={rank}");
            assert_eq!(
                read(&gpu, twin.scales, n * lg),
                want_s,
                "tp={tp} rank={rank}"
            );
            // One pitched copy each for values and scales.
            assert_eq!(gpu.d2d_2d_count(), 2);
        }
    }
}

#[test]
fn a_k_slice_off_the_32_grid_or_a_ragged_stack_is_refused() {
    let gpu = MockGpuBackend::new();
    // 2 ranks of k = 96 -> 48 columns each: a scale block straddles them.
    let (raw, ..) = tensor(&gpu, 1, 16, 96);
    let e = mxfp8_cols_twin(&gpu, raw, 0, 2, 0).unwrap_err();
    assert!(e.to_string().contains("32-value scale block"), "{e}");
    assert!(mxfp8_cols_twin(&gpu, raw, 2, 2, 0).is_err());
    // Segments that do not sum to the part's rows, or parts on different k.
    let (a, ..) = tensor(&gpu, 2, 16, 64);
    let (b, ..) = tensor(&gpu, 3, 16, 128);
    assert!(mxfp8_rows_twin(&gpu, &[(a, &[8, 4])], 0, 1, 0).is_err());
    assert!(mxfp8_rows_twin(&gpu, &[(a, &[16]), (b, &[16])], 0, 1, 0).is_err());
    // A segment not divisible by the TP size.
    assert!(mxfp8_rows_twin(&gpu, &[(a, &[5, 11])], 0, 2, 0).is_err());
}

fn store(entries: &[(&str, Vec<usize>, WeightDtype, u64)]) -> WeightStore {
    WeightStore::from_map(
        entries
            .iter()
            .map(|(name, shape, dtype, ptr)| {
                (
                    name.to_string(),
                    WeightTensor {
                        ptr: DevicePtr(*ptr),
                        shape: shape.clone(),
                        dtype: *dtype,
                    },
                )
            })
            .collect(),
    )
}

#[test]
fn raw_tensor_is_recognised_by_dtype_and_shape_only() {
    use WeightDtype::*;
    let mx = store(&[
        ("p.weight", vec![640, 2560], FP8E4M3, 0x1000),
        ("p.weight_scale", vec![640, 80], UInt8, 0x2000),
        ("e.weight", vec![640, 2560], FP8E4M3, 0x3000),
        ("e.weight_scale", vec![640, 80], FP8E8M0, 0x4000),
    ]);
    let want = RawMxfp8 {
        data: DevicePtr(0x1000),
        scales: DevicePtr(0x2000),
        n: 640,
        k: 2560,
    };
    assert_eq!(raw_mxfp8(&mx, "p").unwrap(), Some(want));
    assert!(raw_mxfp8(&mx, "e").unwrap().is_some());
    assert_eq!(mxfp8_geometry(&mx, "p").unwrap(), Some((640, 2560)));
    // Not MXFP8: BF16; FP8 with a per-tensor F32 scale; FP8 with a per-row
    // BF16 scale (unsloth FP8); FP8 with a 128-block `weight_scale_inv` (no
    // `weight_scale`); packed NVFP4 (U8 weight, E4M3 scale).
    for other in [
        store(&[("p.weight", vec![640, 2560], BF16, 0x1000)]),
        store(&[
            ("p.weight", vec![640, 2560], FP8E4M3, 0x1000),
            ("p.weight_scale", vec![1], FP32, 0x2000),
        ]),
        store(&[
            ("p.weight", vec![640, 2560], FP8E4M3, 0x1000),
            ("p.weight_scale", vec![640, 1], BF16, 0x2000),
        ]),
        store(&[
            ("p.weight", vec![640, 2560], FP8E4M3, 0x1000),
            ("p.weight_scale_inv", vec![5, 20], FP32, 0x2000),
        ]),
        store(&[
            ("p.weight", vec![640, 1280], UInt8, 0x1000),
            ("p.weight_scale", vec![640, 160], FP8E4M3, 0x2000),
        ]),
    ] {
        assert_eq!(raw_mxfp8(&other, "p").unwrap(), None);
        assert_eq!(mxfp8_geometry(&other, "p").unwrap(), None);
    }
    // An E4M3 weight beside a U8 scale of any other shape (or K off the
    // 32 grid) is refused.
    for weight in [vec![640, 2560 + 16], vec![640, 2560, 1]] {
        let bad = store(&[
            ("p.weight", weight, FP8E4M3, 0x1000),
            ("p.weight_scale", vec![640, 80], UInt8, 0x2000),
        ]);
        assert!(mxfp8_geometry(&bad, "p").is_err());
    }
    for scale in [vec![640, 20], vec![80, 640], vec![640 * 80]] {
        let bad = store(&[
            ("p.weight", vec![640, 2560], FP8E4M3, 0x1000),
            ("p.weight_scale", scale, UInt8, 0x2000),
        ]);
        assert!(raw_mxfp8(&bad, "p").is_err());
    }
    // No `.weight` (packed NVFP4, EXL3 trellis): not MXFP8.
    assert_eq!(raw_mxfp8(&mx, "absent").unwrap(), None);
}

#[test]
fn released_raw_bytes_are_refused_not_read() {
    use WeightDtype::*;
    let gpu = MockGpuBackend::new();
    let (w, s) = (gpu.alloc(64 * 64).unwrap(), gpu.alloc(64 * 2).unwrap());
    let st = store(&[
        ("p.weight", vec![64, 64], FP8E4M3, w.0),
        ("p.weight_scale", vec![64, 2], UInt8, s.0),
    ]);
    assert!(raw_mxfp8(&st, "p").unwrap().is_some());
    release_raw_mxfp8(&st, &gpu, "p").unwrap();
    release_raw_mxfp8(&st, &gpu, "p").unwrap(); // idempotent
    let e = raw_mxfp8(&st, "p").unwrap_err();
    assert!(e.to_string().contains("released"), "{e}");
    // The geometry probe reads dtypes and shapes only, so it still answers.
    assert_eq!(mxfp8_geometry(&st, "p").unwrap(), Some((64, 64)));
}

#[test]
fn row_range_twin_is_the_contiguous_row_slice() {
    // GLM-5.3's lm_head is 154880 resident rows of which the first 154856
    // are the vocabulary: each rank's half starts at rank * 154856/2, which
    // is NOT the rank * n/2 a rows twin would cut. Scaled down here.
    let (n, k) = (40, 64);
    let gpu = MockGpuBackend::new();
    let (raw, w, s) = tensor(&gpu, 0x33, n, k);
    for (first, rows) in [(0, 19), (19, 19), (0, n), (39, 1)] {
        let twin = mxfp8_row_range_twin(&gpu, raw, first, rows, 0).unwrap();
        assert!(!twin.tiled);
        assert_eq!(
            read(&gpu, twin.data, rows * k),
            w[first * k..(first + rows) * k]
        );
        assert_eq!(
            read(&gpu, twin.scales, rows * k / 32),
            s[first * k / 32..(first + rows) * k / 32]
        );
    }
    for (first, rows) in [(0, 0), (21, 20), (usize::MAX, 2)] {
        assert!(mxfp8_row_range_twin(&gpu, raw, first, rows, 0).is_err());
    }
}
