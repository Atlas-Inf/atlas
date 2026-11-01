// SPDX-License-Identifier: AGPL-3.0-only

//! A checkpoint's own FP8 block tensors as native twins, sliced for a TP rank.
//!
//! ModelOpt `FP8_PB_WO` (nvidia GLM-5.3-Flash-FP4-Local's MLA q_a / q_b /
//! kv_a / o and shared experts) ships a projection as `<prefix>.weight` E4M3
//! `[n, k]` plus `<prefix>.weight_scale` F32 `[n/128, 1, k/128, 1]` (or the
//! plain 2-D `[n/128, k/128]`): one f32 multiplier per 128 x 128 block, the
//! row-major grid the `fp8b_gemv_tc*` tiers and the block-FP8 GEMMs
//! (`w8a16_gemm_pipelined`, cuBLASLt block FP8) read. So a twin is the
//! checkpoint's bytes, copied and sliced, never requantized or rounded.
//!
//! TP slices keep whole 128 x 128 blocks: a rank's row (column-parallel) or
//! column (row-parallel) range must start and end on the 128 grid, which
//! [`fp8_block_twin`] checks; its scales are then exactly the matching rows /
//! columns of the grid.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::block_scale_grid;
use crate::tp_shard::TpShardKind;

/// Rows and columns per scale block.
pub const FP8_BLOCK: usize = 128;

/// A native FP8 block weight: E4M3 `[n, k]` and an f32 `[n/128, k/128]` grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fp8BlockWeight {
    pub data: DevicePtr,
    pub scales: DevicePtr,
    /// The values are in `mxfp8_tile16`'s chunk tiling (the grid stays
    /// row-major), served only by the `_tiled` tiers.
    pub tiled: bool,
}

impl Fp8BlockWeight {
    /// Untiled buffers for an `[n, k]` weight (`n`, `k` multiples of 128),
    /// the values freed again if the grid does not fit.
    pub fn alloc(gpu: &dyn GpuBackend, n: usize, k: usize) -> Result<Self> {
        ensure!(
            n > 0 && k > 0 && n.is_multiple_of(FP8_BLOCK) && k.is_multiple_of(FP8_BLOCK),
            "FP8 block twin [{n}, {k}] is not on the 128 grid"
        );
        let data = gpu.alloc(n * k)?;
        let scales = match gpu.alloc(grid_bytes(n, k)) {
            Ok(p) => p,
            Err(e) => {
                gpu.free(data)?;
                return Err(e);
            }
        };
        Ok(Self {
            data,
            scales,
            tiled: false,
        })
    }

    /// Free both buffers.
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.data)?;
        gpu.free(self.scales)
    }
}

/// Bytes of the f32 grid of an `[n, k]` FP8 block weight.
pub fn grid_bytes(n: usize, k: usize) -> usize {
    (n / FP8_BLOCK) * (k / FP8_BLOCK) * 4
}

/// Resident bytes of an `[n, k]` FP8 block weight (values + grid).
pub fn fp8_block_bytes(n: usize, k: usize) -> usize {
    n * k + grid_bytes(n, k)
}

/// One checkpoint FP8 block tensor as the store holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawFp8Block {
    /// E4M3 `[n, k]`, row-major.
    pub data: DevicePtr,
    /// F32 `[n/128, k/128]`, row-major.
    pub scales: DevicePtr,
    pub n: usize,
    pub k: usize,
}

/// The raw FP8 block tensor `<prefix>.weight` + `<prefix>.weight_scale`: an
/// E4M3 `[n, k]` weight (n, k multiples of 128) with an F32 scale grid of
/// exactly `[n/128, k/128]` (2-D or ModelOpt's 4-D spelling). `None` when
/// `prefix` is anything else in this checkpoint (BF16, MXFP8's one-byte
/// scales, a per-tensor scalar, other block sizes, no `.weight`). Refuses
/// bytes already released rather than hand out a dangling pointer.
pub fn raw_fp8_block(store: &WeightStore, prefix: &str) -> Result<Option<RawFp8Block>> {
    let (wname, sname) = (format!("{prefix}.weight"), format!("{prefix}.weight_scale"));
    let (Ok(w), Ok(s)) = (store.get(&wname), store.get(&sname)) else {
        return Ok(None);
    };
    if w.dtype != WeightDtype::FP8E4M3 || s.dtype != WeightDtype::FP32 {
        return Ok(None);
    }
    let (&[n, k], Some(grid)) = (&w.shape[..], block_scale_grid(&s.shape)) else {
        return Ok(None);
    };
    if !(n.is_multiple_of(FP8_BLOCK)
        && k.is_multiple_of(FP8_BLOCK)
        && grid == (n / FP8_BLOCK, k / FP8_BLOCK))
    {
        return Ok(None);
    }
    ensure!(
        !store.was_reclaimed(&wname) && !store.was_reclaimed(&sname),
        "{prefix}: the raw FP8 block bytes were released before every consumer read them"
    );
    Ok(Some(RawFp8Block {
        data: w.ptr,
        scales: s.ptr,
        n,
        k,
    }))
}

/// This rank's twin of `raw` sharded by `kind`: the whole tensor
/// (replicated), rows `[rank * n/tp, ..)` (column-parallel) or columns
/// `[rank * k/tp, ..)` (row-parallel), with the matching rows / columns of the
/// scale grid. Refuses a rank boundary off the 128 grid (its blocks would
/// straddle two ranks). Returns the twin and its `(n, k)`. Synchronizes
/// `stream`; frees what it allocated on error.
pub fn fp8_block_twin(
    gpu: &dyn GpuBackend,
    raw: RawFp8Block,
    kind: TpShardKind,
    tp_rank: usize,
    tp_size: usize,
    stream: u64,
) -> Result<(Fp8BlockWeight, usize, usize)> {
    let (n, k) = (raw.n, raw.k);
    ensure!(
        tp_size > 0 && tp_rank < tp_size,
        "fp8_block_twin: tp {tp_rank}/{tp_size}"
    );
    let (gn, gk) = (n / FP8_BLOCK, k / FP8_BLOCK);
    let (ln, lk) = match kind {
        TpShardKind::Replicated => (n, k),
        TpShardKind::ColumnParallel => (n / tp_size, k),
        TpShardKind::RowParallel => (n, k / tp_size),
    };
    let even = match kind {
        TpShardKind::Replicated => true,
        TpShardKind::ColumnParallel => n.is_multiple_of(tp_size),
        TpShardKind::RowParallel => k.is_multiple_of(tp_size),
    };
    if !even || !ln.is_multiple_of(FP8_BLOCK) || !lk.is_multiple_of(FP8_BLOCK) {
        bail!(
            "fp8_block_twin: a rank's [{ln}, {lk}] of [{n}, {k}] ({kind:?}, tp {tp_size}) \
             splits a 128 x 128 scale block; its scales would not be the checkpoint's"
        );
    }
    let twin = Fp8BlockWeight::alloc(gpu, ln, lk)?;
    let copy = || -> Result<()> {
        match kind {
            TpShardKind::Replicated => {
                gpu.copy_d2d_async(raw.data, twin.data, n * k, stream)?;
                gpu.copy_d2d_async(raw.scales, twin.scales, grid_bytes(n, k), stream)?;
            }
            TpShardKind::ColumnParallel => {
                gpu.copy_d2d_async(raw.data.offset(tp_rank * ln * k), twin.data, ln * k, stream)?;
                let rows = ln / FP8_BLOCK;
                gpu.copy_d2d_async(
                    raw.scales.offset(tp_rank * rows * gk * 4),
                    twin.scales,
                    rows * gk * 4,
                    stream,
                )?;
            }
            TpShardKind::RowParallel => {
                gpu.copy_d2d_2d_async(
                    raw.data.offset(tp_rank * lk),
                    k,
                    twin.data,
                    lk,
                    lk,
                    n,
                    stream,
                )?;
                let lgk = lk / FP8_BLOCK;
                gpu.copy_d2d_2d_async(
                    raw.scales.offset(tp_rank * lgk * 4),
                    gk * 4,
                    twin.scales,
                    lgk * 4,
                    lgk * 4,
                    gn,
                    stream,
                )?;
            }
        }
        gpu.synchronize(stream)
    };
    match copy() {
        Ok(()) => Ok((twin, ln, lk)),
        Err(e) => {
            twin.free(gpu)?;
            Err(e)
        }
    }
}

/// Host reference: `y[m, n] = sum_k x[m, k] * e4m3(w[n, k]) * s[n/128, k/128]`
/// in f64, from the checkpoint's bytes (the GPU tests' exact-math oracle).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn fp8_block_gemm_host(
    x: &[f32],
    data: &[u8],
    scales: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f64> {
    let gk = k / FP8_BLOCK;
    let mut y = vec![0f64; m * n];
    for r in 0..n {
        let row = &data[r * k..(r + 1) * k];
        for i in 0..m {
            let xi = &x[i * k..(i + 1) * k];
            let mut acc = 0f64;
            for (c, (&b, &a)) in row.iter().zip(xi).enumerate() {
                let s = scales[(r / FP8_BLOCK) * gk + c / FP8_BLOCK] as f64;
                acc += a as f64 * atlas_core::numeric::fp8_e4m3_to_f32(b) as f64 * s;
            }
            y[i * n + r] = acc;
        }
    }
    y
}

#[cfg(test)]
#[path = "fp8_block_load_tests.rs"]
mod tests;
