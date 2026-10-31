// SPDX-License-Identifier: AGPL-3.0-only

//! A checkpoint's own MXFP8 tensors as decode twins, sliced for a TP rank.
//!
//! An MXFP8 checkpoint (ModelOpt `quant_algo: MXFP8`, `group_size: 32`) ships
//! a projection as `<prefix>.weight` E4M3 `[n, k]` plus `<prefix>.weight_scale`
//! UE8M0 `[n, k/32]` (one power-of-two scale per 32 consecutive K values,
//! stored as U8). That is byte for byte the row-major layout
//! `mxfp8_gemv_tc{8,16,32}` read, so the twin is the checkpoint's bytes,
//! copied and sliced, never requantized: decode then computes with exactly
//! the weights the checkpoint defines (E4M3 times a power of two is exact in
//! the BF16 MMA operand).
//!
//! THE INTEGRATION SEAM is [`raw_mxfp8`] / [`release_raw_mxfp8`]: a loader
//! that keeps the raw tensors in the [`WeightStore`] under their checkpoint
//! names until the layer build has taken its twins needs nothing else. The
//! twins and the exact BF16 dequant (`dense_auto`) read the same raw bytes,
//! in either order, and neither releases them: the loader does, once, after
//! both have read them. [`raw_mxfp8`] refuses bytes already released.
//!
//! TP slices are byte-identical to the matching slice of the full tensor:
//! rows carry their own scales, so row (head) slices are plain row copies;
//! column (K) slices keep whole 32-value scale blocks only when the rank's
//! K boundary is a multiple of 32, which [`mxfp8_cols_twin`] checks.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::Mxfp8Weight;
use crate::layers::ops::MXFP8_BLOCK;
use crate::tp_shard::segment_copy_plan;

/// One checkpoint MXFP8 tensor as the store holds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawMxfp8 {
    /// E4M3 `[n, k]`, row-major.
    pub data: DevicePtr,
    /// UE8M0 `[n, k / 32]`, row-major.
    pub scales: DevicePtr,
    pub n: usize,
    pub k: usize,
}

/// `(n, k)` when `<prefix>.weight` + `<prefix>.weight_scale` is a checkpoint
/// MXFP8 projection, from dtypes and shapes alone; `None` when `prefix` is not
/// MXFP8 in this checkpoint (BF16, FP8 with a per-tensor F32, a per-row BF16
/// or a 128-block `weight_scale_inv` scale, or no `.weight` at all: packed
/// NVFP4 `.weight_packed`, EXL3 trellis). MXFP8 is recognised by an E4M3
/// weight with a ONE-BYTE `weight_scale` (U8 or F8_E8M0): no other FP8
/// scheme in the tree stores its scale that way. So an E4M3 weight with a
/// one-byte scale of any other shape is an error, not a fallback: it would
/// otherwise fall through to a layout it is not.
pub fn mxfp8_geometry(store: &WeightStore, prefix: &str) -> Result<Option<(usize, usize)>> {
    let (wname, sname) = (format!("{prefix}.weight"), format!("{prefix}.weight_scale"));
    let (Ok(w), Ok(s)) = (store.get(&wname), store.get(&sname)) else {
        return Ok(None);
    };
    if w.dtype != WeightDtype::FP8E4M3
        || !matches!(s.dtype, WeightDtype::UInt8 | WeightDtype::FP8E8M0)
    {
        return Ok(None);
    }
    let [n, k] = w.shape[..] else {
        bail!("{wname}: MXFP8 weight must be 2-D, got {:?}", w.shape);
    };
    ensure!(
        n > 0 && k > 0 && k.is_multiple_of(MXFP8_BLOCK) && s.shape == [n, k / MXFP8_BLOCK],
        "{prefix}: E4M3 weight {:?} with a {:?} scale {:?} is not MXFP8 \
         ([n, k] + [n, k/32], k % 32 == 0)",
        w.shape,
        s.dtype,
        s.shape,
    );
    Ok(Some((n, k)))
}

/// The raw MXFP8 tensor `<prefix>.weight` + `<prefix>.weight_scale` (see
/// [`mxfp8_geometry`] for what counts), with its device pointers. Refuses
/// bytes [`release_raw_mxfp8`] already freed rather than hand out a dangling
/// pointer.
pub fn raw_mxfp8(store: &WeightStore, prefix: &str) -> Result<Option<RawMxfp8>> {
    let Some((n, k)) = mxfp8_geometry(store, prefix)? else {
        return Ok(None);
    };
    let (wname, sname) = (format!("{prefix}.weight"), format!("{prefix}.weight_scale"));
    ensure!(
        !store.was_reclaimed(&wname) && !store.was_reclaimed(&sname),
        "{prefix}: the raw MXFP8 bytes were released before every consumer read them"
    );
    Ok(Some(RawMxfp8 {
        data: store.get(&wname)?.ptr,
        scales: store.get(&sname)?.ptr,
        n,
        k,
    }))
}

/// Free the raw tensors [`raw_mxfp8`] returned once every consumer (twin,
/// BF16 dequant) has read them. Idempotent.
pub fn release_raw_mxfp8(store: &WeightStore, gpu: &dyn GpuBackend, prefix: &str) -> Result<()> {
    store.reclaim(gpu, &format!("{prefix}.weight"))?;
    store.reclaim(gpu, &format!("{prefix}.weight_scale"))
}

/// Rows of `parts` stacked in order into one twin `[rows, k]`, each part cut
/// to this rank by its row `segments` (summing to its `n`): every segment
/// keeps its `[rank * seg/tp, (rank+1) * seg/tp)` rows, as
/// `tp_shard::segment_copy_plan` slices the BF16 copies (a GDN `[Q|K|V]` +
/// `[Z]` pair stacks to the layer's local `[Q|K|V|Z]`). Returns the twin and
/// its `(rows, k)`. Synchronizes `stream`.
pub fn mxfp8_rows_twin(
    gpu: &dyn GpuBackend,
    parts: &[(RawMxfp8, &[usize])],
    tp_rank: usize,
    tp_size: usize,
    stream: u64,
) -> Result<(Mxfp8Weight, usize, usize)> {
    let k = parts.first().map(|(raw, _)| raw.k).unwrap_or(0);
    ensure!(k > 0, "mxfp8_rows_twin: no parts");
    let mut plans = Vec::with_capacity(parts.len());
    let mut rows = 0usize;
    for (raw, segments) in parts {
        ensure!(
            raw.k == k && segments.iter().sum::<usize>() == raw.n,
            "mxfp8_rows_twin: part [{}, {}] with segments {segments:?} does not stack on k={k}",
            raw.n,
            raw.k,
        );
        let (data, local) = segment_copy_plan(segments, k, tp_rank, tp_size)?;
        let (scales, _) = segment_copy_plan(segments, k / MXFP8_BLOCK, tp_rank, tp_size)?;
        plans.push((rows, data, scales));
        rows += local;
    }
    let twin = Mxfp8Weight::alloc(gpu, rows, k)?;
    let copy = || -> Result<()> {
        for ((raw, _), (first, data, scales)) in parts.iter().zip(&plans) {
            let (dst_w, dst_s) = (first * k, first * k / MXFP8_BLOCK);
            for op in data {
                let dst = twin.data.offset(dst_w + op.dst_off);
                gpu.copy_d2d_async(raw.data.offset(op.src_off), dst, op.len, stream)?;
            }
            for op in scales {
                let dst = twin.scales.offset(dst_s + op.dst_off);
                gpu.copy_d2d_async(raw.scales.offset(op.src_off), dst, op.len, stream)?;
            }
        }
        gpu.synchronize(stream)
    };
    finish(gpu, twin, copy()).map(|t| (t, rows, k))
}

/// Columns `[rank * k/tp, (rank+1) * k/tp)` of `raw` (a row-parallel
/// projection's input slice, e.g. `out_proj` / `o_proj`) as one twin
/// `[n, k/tp]`, one pitched copy each for values and scales. Refuses a rank
/// boundary off the 32 grid: its scale block would straddle two ranks.
/// Returns the twin and its `(n, k/tp)`. Synchronizes `stream`.
pub fn mxfp8_cols_twin(
    gpu: &dyn GpuBackend,
    raw: RawMxfp8,
    tp_rank: usize,
    tp_size: usize,
    stream: u64,
) -> Result<(Mxfp8Weight, usize, usize)> {
    let (n, k) = (raw.n, raw.k);
    ensure!(
        tp_rank < tp_size && k.is_multiple_of(tp_size),
        "mxfp8_cols_twin: k={k} over tp {tp_rank}/{tp_size}"
    );
    let lk = k / tp_size;
    ensure!(
        lk.is_multiple_of(MXFP8_BLOCK),
        "mxfp8_cols_twin: a rank's {lk} columns of k={k} split a 32-value scale block; \
         its MXFP8 scales would not be the checkpoint's"
    );
    let (g, lg) = (k / MXFP8_BLOCK, lk / MXFP8_BLOCK);
    let twin = Mxfp8Weight::alloc(gpu, n, lk)?;
    let copy = || -> Result<()> {
        let (src_w, src_s) = (
            raw.data.offset(tp_rank * lk),
            raw.scales.offset(tp_rank * lg),
        );
        gpu.copy_d2d_2d_async(src_w, k, twin.data, lk, lk, n, stream)?;
        gpu.copy_d2d_2d_async(src_s, g, twin.scales, lg, lg, n, stream)?;
        gpu.synchronize(stream)
    };
    finish(gpu, twin, copy()).map(|t| (t, n, lk))
}

/// Rows `[first, first + rows)` of `raw` as one twin `[rows, k]`: a
/// contiguous slice of values and of scales (each row carries its own), e.g.
/// one rank's vocabulary half of an MXFP8 `lm_head`. Synchronizes `stream`.
pub fn mxfp8_row_range_twin(
    gpu: &dyn GpuBackend,
    raw: RawMxfp8,
    first: usize,
    rows: usize,
    stream: u64,
) -> Result<Mxfp8Weight> {
    ensure!(
        rows > 0 && first.checked_add(rows).is_some_and(|end| end <= raw.n),
        "mxfp8_row_range_twin: rows [{first}, {first}+{rows}) of n={}",
        raw.n
    );
    let (k, g) = (raw.k, raw.k / MXFP8_BLOCK);
    let twin = Mxfp8Weight::alloc(gpu, rows, k)?;
    let copy = || -> Result<()> {
        gpu.copy_d2d_async(raw.data.offset(first * k), twin.data, rows * k, stream)?;
        gpu.copy_d2d_async(raw.scales.offset(first * g), twin.scales, rows * g, stream)?;
        gpu.synchronize(stream)
    };
    finish(gpu, twin, copy())
}

/// The twin, or its buffers freed and the copy's error.
fn finish(gpu: &dyn GpuBackend, twin: Mxfp8Weight, copied: Result<()>) -> Result<Mxfp8Weight> {
    if let Err(e) = copied {
        gpu.free(twin.data)?;
        gpu.free(twin.scales)?;
        return Err(e);
    }
    Ok(twin)
}

#[cfg(test)]
#[path = "mxfp8_load_tests.rs"]
mod tests;
