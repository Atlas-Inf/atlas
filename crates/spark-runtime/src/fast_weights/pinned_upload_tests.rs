// SPDX-License-Identifier: AGPL-3.0-only

use super::*;
use crate::gpu::mock::MockGpuBackend;
use std::io::Write;
use std::os::unix::io::AsRawFd;

/// A scratch file of `len` bytes, byte `i` being `(i * 7 + i / 4096) as u8`
/// so a piece read from the wrong offset shows.
fn scratch(tag: &str, len: usize) -> (std::path::PathBuf, std::fs::File, Vec<u8>) {
    let data: Vec<u8> = (0..len).map(|i| (i * 7 + i / 4096) as u8).collect();
    let path = std::env::temp_dir().join(format!("atlas-pinned-{tag}-{}", std::process::id()));
    std::fs::File::create(&path)
        .unwrap()
        .write_all(&data)
        .unwrap();
    let file = std::fs::File::open(&path).unwrap();
    (path, file, data)
}

#[test]
fn pieces_cover_each_tensor_exactly_once() {
    let a = DevicePtr(0x1000);
    let b = DevicePtr(0x9000);
    let got = pieces(&[(10, 25, a), (100, 0, b), (200, 10, b)], 10);
    let want = [
        (10, 10, a),
        (20, 10, a.offset(10)),
        (30, 5, a.offset(20)),
        (200, 10, b),
    ];
    let want: Vec<Piece> = want
        .iter()
        .map(|&(file_off, len, dst)| Piece { file_off, len, dst })
        .collect();
    assert_eq!(got, want);
}

/// Every tensor lands byte for byte, whether or not its reads are widened to
/// O_DIRECT's aligned windows, including one larger than a piece and one at
/// the very end of the file (its window runs past end-of-file).
#[test]
fn every_tensor_lands_byte_for_byte() {
    let len = PIECE_BYTES + 3 * 4096 + 123;
    let (path, file, data) = scratch("land", len);
    let gpu = MockGpuBackend::new();
    let up = PinnedUploader::new(&gpu).unwrap();
    let spans = [(5usize, 1000usize), (4096, 4096), (9000, PIECE_BYTES + 77)];
    let tail = (len - 50, 50);
    for direct in [false, true] {
        let spans: Vec<(u64, usize)> = spans
            .iter()
            .chain([&tail])
            .map(|&(off, n)| (off as u64, n))
            .collect();
        let ptrs = up
            .upload(file.as_raw_fd(), direct, &spans, |i| {
                Some(gpu.alloc(spans[i].1).unwrap())
            })
            .unwrap();
        for (&(off, n), dst) in spans.iter().zip(ptrs) {
            let mut back = vec![0u8; n];
            gpu.copy_d2h(dst.unwrap(), &mut back).unwrap();
            assert!(
                back == data[off as usize..off as usize + n],
                "tensor at {off} (+{n}), aligned windows: {direct}"
            );
        }
    }
    std::fs::remove_file(path).ok();
}

/// A tensor the file cannot supply fails the upload; it does not hang it or
/// pass it.
#[test]
fn a_tensor_past_end_of_file_is_an_error() {
    let (path, file, _) = scratch("short", 8192);
    let gpu = MockGpuBackend::new();
    let up = PinnedUploader::new(&gpu).unwrap();
    let err = up
        .upload(file.as_raw_fd(), false, &[(0, 100), (4096, 8192)], |i| {
            Some(gpu.alloc([100, 8192][i]).unwrap())
        })
        .expect_err("8 KiB from offset 4096 of an 8 KiB file");
    assert!(err.to_string().contains("short read"), "{err}");
    std::fs::remove_file(path).ok();
}

/// Nothing to upload is not an error and starts no worker.
#[test]
fn no_tensors_is_a_no_op() {
    let (path, file, _) = scratch("empty", 4096);
    let gpu = MockGpuBackend::new();
    let up = PinnedUploader::new(&gpu).unwrap();
    assert!(
        up.upload(file.as_raw_fd(), true, &[], |_| None)
            .unwrap()
            .is_empty()
    );
    std::fs::remove_file(path).ok();
}

/// A tensor with no device memory is skipped and reported as such; the
/// others still land.
#[test]
fn an_unallocated_tensor_is_left_to_the_caller() {
    let (path, file, data) = scratch("skip", 3 * 4096);
    let gpu = MockGpuBackend::new();
    let up = PinnedUploader::new(&gpu).unwrap();
    let spans = [(0, 4096), (4096, 4096), (8192, 4096)];
    let ptrs = up
        .upload(file.as_raw_fd(), true, &spans, |i| {
            (i != 1).then(|| gpu.alloc(4096).unwrap())
        })
        .unwrap();
    assert!(ptrs[1].is_none());
    let mut back = vec![0u8; 4096];
    gpu.copy_d2h(ptrs[2].unwrap(), &mut back).unwrap();
    assert!(back == data[8192..]);
    std::fs::remove_file(path).ok();
}
