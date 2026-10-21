// SPDX-License-Identifier: AGPL-3.0-only
//
//! TP sharding of native MXFP8 modules (nvidia/Qwen3.8-Flash-Next-FP4-Local).
//!
//! An MXFP8 module is two logical row-major tensors: E4M3 `data` `[n, k]` at
//! 1 B/element and E8M0 `scales` `[n, k / 32]` at 1 B each. Every shard slices
//! both with the SAME row or column selection:
//!
//! - **Column / segmented**: rank `r` keeps rows `[r * seg / tp, (r+1) *
//!   seg / tp)` of every segment — a contiguous (or per-segment contiguous)
//!   slice in both tensors, since scales share the row axis.
//! - **Row-parallel**: rank `r` keeps columns `[r * k/tp, (r+1) * k/tp)` of
//!   `data` and `[r * k/(32*tp), (r+1) * k/(32*tp))` of `scales`, so each rank
//!   owns whole 32-input scale groups — required, or a scale would describe
//!   inputs living on another rank. `(k / tp) % 32 == 0` is enforced.
//!
//! Byte-exact: slicing packed E4M3 + E8M0 storage needs no dequant, so a
//! local shard is bit-identical to the rows/columns of the shipped module.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{TpShardKind, shard_dense_bytes, slice_segments};

/// Inputs per E8M0 scale — mirrors `mx8_common::MX8_GROUP` without the
/// weight-loader dependency.
const MX8_GROUP: usize = 32;

/// Shard an MXFP8 module `[data, scales]` on its ROW axis.
///
/// `segments` lists the full (pre-shard) row counts of each contiguous block;
/// a single-element slice is the plain `ColumnParallel` case. Each segment is
/// sliced to this rank's range independently and re-packed, in both `data`
/// (row width `k`) and `scales` (row width `k / 32`).
///
/// Returns `([local_data, local_scales], local_total_rows)`. At `tp_size <= 1`
/// the source pointers pass through untouched (no copies; the caller must not
/// free or reclaim them).
pub fn shard_mxfp8_rows(
    src: [DevicePtr; 2],
    segments: &[usize],
    k: usize,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<([DevicePtr; 2], usize)> {
    ensure!(
        k.is_multiple_of(MX8_GROUP),
        "MXFP8 shard: k {k} not a multiple of {MX8_GROUP}"
    );
    let (data, local_rows) = slice_segments(src[0], segments, k, 1, tp_rank, tp_size, gpu)?;
    let (scales, scale_rows) =
        slice_segments(src[1], segments, k / MX8_GROUP, 1, tp_rank, tp_size, gpu)?;
    ensure!(
        scale_rows == local_rows,
        "MXFP8 shard: {local_rows} data rows but {scale_rows} scale rows"
    );
    Ok(([data, scales], local_rows))
}

/// Shard an MXFP8 module `[data, scales]` `RowParallel` on its INPUT axis:
/// `data` `[n, k]` → `[n, k / tp]`, `scales` `[n, k / 32]` → `[n, k / (32*tp)]`.
/// The per-row column slices keep each rank's 32-input scale groups whole —
/// `(k / tp) % 32 == 0` is required — so the local projection dequantizes
/// exactly the shipped columns. Partial sums reconcile in the existing
/// post-projection all-reduce.
///
/// Returns `([local_data, local_scales], local_k)`. At `tp_size <= 1` the
/// source pointers pass through untouched.
pub fn shard_mxfp8_row_parallel(
    src: [DevicePtr; 2],
    n: usize,
    k: usize,
    tp_rank: usize,
    tp_size: usize,
    gpu: &dyn GpuBackend,
) -> Result<([DevicePtr; 2], usize)> {
    if tp_size <= 1 {
        return Ok((src, k));
    }
    let local_k = k / tp_size.max(1);
    ensure!(
        k.is_multiple_of(tp_size) && local_k.is_multiple_of(MX8_GROUP),
        "MXFP8 row-parallel shard: k {k} splits into {local_k} per rank over \
         tp {tp_size}, which is not a multiple of the {MX8_GROUP}-input scale \
         group — a scale would describe inputs living on another rank"
    );
    let (data, _, dk) = shard_dense_bytes(
        src[0],
        n,
        k,
        1,
        TpShardKind::RowParallel,
        tp_rank,
        tp_size,
        gpu,
    )?;
    let (scales, _, sk) = shard_dense_bytes(
        src[1],
        n,
        k / MX8_GROUP,
        1,
        TpShardKind::RowParallel,
        tp_rank,
        tp_size,
        gpu,
    )?;
    ensure!(
        dk == local_k && sk == local_k / MX8_GROUP,
        "MXFP8 row-parallel shard: local k {dk} data / {sk} scales, expected {local_k} / {}",
        local_k / MX8_GROUP
    );
    Ok(([data, scales], local_k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use spark_runtime::gpu::mock::MockGpuBackend;

    /// E4M3 (finite only — tests avoid NaN/inf encodings) → f32.
    fn e4m3(b: u8) -> f32 {
        let s = if b & 0x80 != 0 { -1.0 } else { 1.0 };
        let e = (b >> 3) & 0x0f;
        let m = b & 0x07;
        let v = if e == 0 {
            (m as f32 / 8.0) * 2f32.powi(-6)
        } else {
            (1.0 + m as f32 / 8.0) * 2f32.powi(e as i32 - 7)
        };
        s * v
    }

    /// Host oracle: dequant an MXFP8 module `data [n,k]` + `scales [n,k/32]`.
    fn dequant(data: &[u8], scales: &[u8], n: usize, k: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n * k);
        for r in 0..n {
            for c in 0..k {
                let sc = scales[r * (k / 32) + c / 32];
                out.push(e4m3(data[r * k + c]) * 2f32.powi(sc as i32 - 127));
            }
        }
        out
    }

    /// Deterministic module bytes: values chosen so no scale group is
    /// degenerate and a wrong scale column flips results.
    fn synth_module(n: usize, k: usize) -> (Vec<u8>, Vec<u8>) {
        let data: Vec<u8> = (0..n * k).map(|i| ((i * 37 + 11) % 100) as u8).collect();
        let scales: Vec<u8> = (0..n * (k / 32))
            .map(|i| (120 + ((i * 13 + 5) % 9)) as u8)
            .collect();
        (data, scales)
    }

    fn upload(gpu: &MockGpuBackend, bytes: &[u8]) -> DevicePtr {
        let p = gpu.alloc(bytes.len()).unwrap();
        gpu.copy_h2d(bytes, p).unwrap();
        p
    }

    fn fetch(gpu: &MockGpuBackend, p: DevicePtr, bytes: usize) -> Vec<u8> {
        let mut v = vec![0u8; bytes];
        gpu.copy_d2h(p, &mut v).unwrap();
        v
    }

    #[test]
    fn column_shard_offsets_match_host_slice() {
        let gpu = MockGpuBackend::new();
        let (n, k) = (8usize, 128usize);
        let (data, scales) = synth_module(n, k);
        let src = [upload(&gpu, &data), upload(&gpu, &scales)];
        for rank in 0..2 {
            let (l, rows) = shard_mxfp8_rows(src, &[n], k, rank, 2, &gpu).unwrap();
            assert_eq!(rows, n / 2);
            let ld = fetch(&gpu, l[0], rows * k);
            let ls = fetch(&gpu, l[1], rows * (k / 32));
            let base = rank * (n / 2);
            assert_eq!(ld, data[base * k..base * k + rows * k]);
            assert_eq!(
                ls,
                scales[base * (k / 32)..base * (k / 32) + rows * (k / 32)]
            );
            // dequant(shard) == slice(dequant(full)), element for element.
            let want = dequant(&data, &scales, n, k);
            let got = dequant(&ld, &ls, rows, k);
            assert_eq!(got, want[base * k..base * k + rows * k]);
        }
    }

    #[test]
    fn segmented_shard_slices_each_block_independently() {
        let gpu = MockGpuBackend::new();
        let segs = [2usize, 2usize, 4usize];
        let (n, k) = (8usize, 64usize);
        let (data, scales) = synth_module(n, k);
        let src = [upload(&gpu, &data), upload(&gpu, &scales)];
        let (l, rows) = shard_mxfp8_rows(src, &segs, k, 1, 2, &gpu).unwrap();
        // rank 1: second half of every segment.
        let mut want_off = Vec::new();
        let mut off = 0usize;
        for &s in &segs {
            let local = s / 2;
            want_off.push((off + local) * k);
            want_off.push((off + local * 2) * k);
            off += s;
        }
        let ld = fetch(&gpu, l[0], rows * k);
        let mut want = Vec::new();
        for w in want_off.chunks_exact(2) {
            want.extend_from_slice(&data[w[0]..w[1]]);
        }
        assert_eq!(ld, want);
    }

    #[test]
    fn row_parallel_keeps_whole_scale_groups() {
        let gpu = MockGpuBackend::new();
        let (n, k) = (4usize, 128usize);
        let (data, scales) = synth_module(n, k);
        let src = [upload(&gpu, &data), upload(&gpu, &scales)];
        for rank in 0..2 {
            let (l, lk) = shard_mxfp8_row_parallel(src, n, k, rank, 2, &gpu).unwrap();
            assert_eq!(lk, k / 2);
            let ld = fetch(&gpu, l[0], n * lk);
            let ls = fetch(&gpu, l[1], n * (lk / 32));
            // Build the expected local module on the host: per-row column
            // slice of data AND of scales, then compare dequants.
            let mut wd = vec![0u8; n * lk];
            let mut ws = vec![0u8; n * (lk / 32)];
            for r in 0..n {
                wd[r * lk..r * lk + lk]
                    .copy_from_slice(&data[r * k + rank * lk..r * k + (rank + 1) * lk]);
                ws[r * (lk / 32)..r * (lk / 32) + lk / 32].copy_from_slice(
                    &scales[r * (k / 32) + rank * (lk / 32)..r * (k / 32) + (rank + 1) * (lk / 32)],
                );
            }
            assert_eq!(ld, wd);
            assert_eq!(ls, ws);
            assert_eq!(dequant(&ld, &ls, n, lk), dequant(&wd, &ws, n, lk));
        }
    }

    #[test]
    fn row_parallel_refuses_misaligned_scale_groups() {
        let gpu = MockGpuBackend::new();
        let (n, k) = (4usize, 96usize);
        let (data, scales) = synth_module(n, k);
        let src = [upload(&gpu, &data), upload(&gpu, &scales)];
        // 96/2 = 48, not a multiple of 32 -> scale groups would straddle ranks.
        assert!(shard_mxfp8_row_parallel(src, n, k, 0, 2, &gpu).is_err());
        // 64/3 is not integral at all.
        let (d2, s2) = synth_module(n, 64);
        let src2 = [upload(&gpu, &d2), upload(&gpu, &s2)];
        assert!(shard_mxfp8_row_parallel(src2, n, 64, 0, 3, &gpu).is_err());
    }

    #[test]
    fn off_by_one_scale_column_fails_dequant_equality() {
        // Negative control: the same column slice with the scale column
        // shifted by one must NOT compare equal to a true slice — otherwise
        // the dequant-equality tests above would pass on a misaligned shard.
        let (n, k) = (4usize, 128usize);
        let (data, scales) = synth_module(n, k);
        let lk = k / 2;
        let r = 1usize;
        let mut good_s = vec![0u8; lk / 32];
        good_s.copy_from_slice(&scales[r * (k / 32) + (lk / 32)..r * (k / 32) + 2 * (lk / 32)]);
        let mut bad_s = good_s.clone();
        bad_s.copy_from_slice(&scales[r * (k / 32) + (lk / 32) + 1..r * (k / 32) + (lk / 32) + 3]);
        let d = &data[r * k + lk..r * k + 2 * lk];
        let mut good = Vec::new();
        let mut bad = Vec::new();
        for c in 0..lk {
            good.push(e4m3(d[c]) * 2f32.powi(good_s[c / 32] as i32 - 127));
            bad.push(e4m3(d[c]) * 2f32.powi(bad_s[c / 32] as i32 - 127));
        }
        assert_ne!(good, bad);
    }

    /// Row-parallel reassembly: dequant(shard0) || dequant(shard1) per row
    /// equals dequant(full) bit-for-bit, on hostile E4M3/E8M0 bytes (NaN
    /// codes, exponent extremes) compared via f32::to_bits.
    #[test]
    fn row_parallel_reassembly_is_bit_exact_on_hostile_bytes() {
        let gpu = MockGpuBackend::new();
        let (n, k) = (4usize, 128usize);
        // E4M3 0x7F / 0xFF are NaN codes; E8M0 0xFF is NaN, 0x00 subnormal.
        let mut data: Vec<u8> = (0..n * k).map(|i| (i * 91) as u8).collect();
        let mut scales: Vec<u8> = (0..n * (k / 32)).map(|i| (i * 61) as u8).collect();
        for (i, b) in [0x00u8, 0x7F, 0xFF, 0x80, 0xFE].iter().enumerate() {
            data[i] = *b;
            scales[i] = [0x00u8, 0xFE, 0xFF, 0x7F, 0x80][i];
        }
        let src = [upload(&gpu, &data), upload(&gpu, &scales)];
        let full = dequant(&data, &scales, n, k);
        for r in 0..n {
            let mut joined = Vec::new();
            for rank in 0..2 {
                let (l, lk) = shard_mxfp8_row_parallel(src, n, k, rank, 2, &gpu).unwrap();
                let ld = fetch(&gpu, l[0], n * lk);
                let ls = fetch(&gpu, l[1], n * (lk / 32));
                // Row r of the local module: dequant it standalone.
                let d = dequant(
                    &ld[r * lk..(r + 1) * lk],
                    &ls[r * (lk / 32)..(r + 1) * (lk / 32)],
                    1,
                    lk,
                );
                joined.extend_from_slice(&d);
            }
            let got: Vec<u32> = joined.iter().map(|v| v.to_bits()).collect();
            let want: Vec<u32> = full[r * k..(r + 1) * k].iter().map(|v| v.to_bits()).collect();
            assert_eq!(got, want, "row {r}: reassembled dequant differs");
        }
        // (k / tp) % 32 != 0 must be rejected: a rank would split a scale group.
        assert!(
            shard_mxfp8_row_parallel(src, n, 96, 0, 2, &gpu).is_err(),
            "k=96 over tp=2 leaves 48 not a multiple of 32"
        );
    }
}
