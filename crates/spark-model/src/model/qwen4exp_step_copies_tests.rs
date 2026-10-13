// SPDX-License-Identifier: AGPL-3.0-only

//! [`copy_rows`]: the pitched form lands the loop's bytes, in one call.

use super::*;
use spark_runtime::gpu::mock::MockGpuBackend;

/// A source of `rows` rows `src_pitch` apart, every byte distinct-ish, and a
/// destination pre-filled with a sentinel so a byte the copy must not touch
/// (row padding) is checked too.
fn setup(
    gpu: &MockGpuBackend,
    rows: usize,
    src_pitch: usize,
    dst_pitch: usize,
) -> (DevicePtr, DevicePtr) {
    let src = gpu.alloc(rows * src_pitch).unwrap();
    // The last row may end past `rows * dst_pitch` when the pitch is short.
    let dst_bytes = rows * dst_pitch + src_pitch;
    let dst = gpu.alloc(dst_bytes).unwrap();
    let fill: Vec<u8> = (0..rows * src_pitch)
        .map(|i| (i * 7 % 251 + 1) as u8)
        .collect();
    gpu.copy_h2d(&fill, src).unwrap();
    gpu.copy_h2d(&vec![0xEE; dst_bytes], dst).unwrap();
    (src, dst)
}

fn run(
    pitched: bool,
    rows: usize,
    src_pitch: usize,
    dst_pitch: usize,
    width: usize,
) -> (Vec<u8>, usize, usize) {
    let gpu = MockGpuBackend::new();
    let (src, dst) = setup(&gpu, rows, src_pitch, dst_pitch);
    copy_rows(
        &gpu, src, src_pitch, dst, dst_pitch, width, rows, pitched, 0,
    )
    .unwrap();
    (
        gpu.read_alloc(dst).unwrap(),
        gpu.d2d_count(),
        gpu.d2d_2d_count(),
    )
}

#[test]
fn pitched_lands_the_loops_bytes_in_one_call() {
    // The production shapes: Q+gate (12288 B rows into a 13312 B interleaved
    // stride) and K/V (512 B), at 1..=32 rows, scaled down 64x.
    for &(width, src_pitch, dst_pitch) in &[(192, 192, 208), (8, 8, 208), (8, 16, 24)] {
        for rows in [1, 2, 4, 7, 32] {
            let (looped, l_d2d, l_2d) = run(false, rows, src_pitch, dst_pitch, width);
            let (batched, b_d2d, b_2d) = run(true, rows, src_pitch, dst_pitch, width);
            assert_eq!(looped, batched, "rows={rows} width={width}");
            assert_eq!((l_d2d, l_2d), (rows, 0));
            assert_eq!(
                (b_d2d, b_2d),
                (0, 1),
                "one pitched copy whatever the row count"
            );
        }
    }
}

#[test]
fn overlapping_pitch_takes_the_loop() {
    // pitch < width is not a 2-D shape: the loop runs, verbatim.
    let (looped, ..) = run(false, 4, 16, 8, 12);
    let (fallback, d2d, d2d_2d) = run(true, 4, 16, 8, 12);
    assert_eq!(looped, fallback);
    assert_eq!((d2d, d2d_2d), (4, 0));
}

#[test]
fn empty_shapes_issue_nothing() {
    let gpu = MockGpuBackend::new();
    let (src, dst) = setup(&gpu, 2, 8, 8);
    copy_rows(&gpu, src, 8, dst, 8, 8, 0, true, 0).unwrap();
    copy_rows(&gpu, src, 8, dst, 8, 0, 2, true, 0).unwrap();
    assert_eq!((gpu.d2d_count(), gpu.d2d_2d_count()), (0, 0));
}
