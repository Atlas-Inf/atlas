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
