// SPDX-License-Identifier: AGPL-3.0-only
//! The pinned MXFP8 / block-FP8 kernel of an `n x k` W8A8 GEMM: one in-order
//! cuBLASLt kernel for every row count (`mx_algo`'s untimed sibling of
//! `select`). Split from `mx_algo.rs` for the file-size cap.

use std::sync::Mutex;

use super::*;

/// Row counts a [`pinned`] kernel must take: a packed GLM prefill's passes
/// (`spark-model` `glm_packed_scope`), one prompt of 2 rows up to a full
/// arena.
const PINNED_ROWS: [u32; 5] = [1, 30, 64, 4096, 8192];
/// The block FP8 heights: cuBLASLt's VEC128 activation scales need a row
/// count on the 16 grid, which the caller pads to.
const PINNED_ROWS_BLK128: [u32; 5] = [16, 32, 64, 4096, 8192];
/// `(n, k, operand kind)` of a pinned kernel.
type PinKey = (u32, u32, u8);
static PINNED: Mutex<Vec<(PinKey, Algo)>> = Mutex::new(Vec::new());

/// ONE in-order kernel for `n x k` at every row count, never retuned: the
/// one [`fp8_gemm_mx_pinned_prepare`] timed at load, else (no load-time pick)
/// the heuristic's first in-order candidate (asked at 4096 rows) that takes
/// every [`PINNED_ROWS`] height, chosen without timing. Under one in-order
/// kernel every output is one FP32 accumulation over K in order, so a row's
/// bytes do not depend on the pass height; unlike [`select`], which tunes by
/// timing per row bucket (swept shapes) and retunes a key whose kernel
/// refuses a height, the choice cannot move with the pass mix or history.
pub(in crate::cublaslt) fn pinned(ctx: &Ctx, n: u32, k: u32, ops: Operands) -> Result<Algo> {
    pin(ctx, n, k, ops, None)
}

/// [`pinned`], choosing (when nothing is pinned yet) the fastest of the
/// eligible in-order candidates timed at the reference height on `timed`'s
/// operands (`[weight, act, out]` of a `ref_m`-row problem, and the stream).
fn pin(
    ctx: &Ctx,
    n: u32,
    k: u32,
    ops: Operands,
    timed: Option<([u64; 3], u32, u64)>,
) -> Result<Algo> {
    let key = (n, k, ops.kind());
    if let Some(a) = PINNED.lock().unwrap().iter().find(|e| e.0 == key) {
        return Ok(a.1);
    }
    let ref_m = timed.map_or(4096, |t| t.1);
    let reference = Problem::with(ref_m, n, k, ops)?;
    let rows: &[u32] = match ops {
        Operands::Blk128 { .. } => &PINNED_ROWS_BLK128,
        _ => &PINNED_ROWS,
    };
    let heights = rows
        .iter()
        .map(|&m| Problem::with(m, n, k, ops))
        .collect::<Result<Vec<_>>>()?;
    let fit: Vec<Algo> = candidates(ctx, &reference, true)?
        .into_iter()
        .filter(|a| {
            in_order(a) && takes(ctx, &reference, a) && heights.iter().all(|p| takes(ctx, p, a))
        })
        .collect();
    anyhow::ensure!(
        !fit.is_empty(),
        "cuBLASLt FP8 (kind {}) {n}x{k}: no in-order kernel takes every pinned height",
        ops.kind()
    );
    let (best, how) = match timed {
        Some((ptrs, _, stream)) if fit.len() > 1 => {
            let mut times = Vec::with_capacity(fit.len());
            for a in &fit {
                times.push(time(ctx, &reference, a, ptrs, stream)?);
            }
            let best = (0..fit.len())
                .min_by(|&i, &j| times[i].total_cmp(&times[j]))
                .expect("non-empty");
            (
                best,
                format!(
                    "fastest at {ref_m} rows, {:.3} ms ({} eligible; the first ran {:.3} ms)",
                    times[best],
                    fit.len(),
                    times[0]
                ),
            )
        }
        Some(_) => (0, "the only eligible kernel".to_string()),
        None => (
            0,
            "untimed: the heuristic's first eligible kernel".to_string(),
        ),
    };
    let algo = fit[best];
    tracing::info!(
        "cuBLASLt FP8 (kind {}) {n}x{k}: pinned to {} at every row count ({how})",
        ops.kind(),
        description(&algo)
    );
    let mut pinned = PINNED.lock().unwrap();
    if let Some(e) = pinned.iter().find(|e| e.0 == key) {
        return Ok(e.1);
    }
    pinned.push((key, algo));
    Ok(algo)
}

/// Pin the MXFP8 W8A8 kernel of `n x k` now (load time): the fastest
/// in-order candidate that takes every pinned height, timed at `ref_m` rows
/// on scratch operands, kept for the run (`pinned`). A no-op once pinned.
/// Synchronizes `stream`.
pub fn fp8_gemm_mx_pinned_prepare(n: u32, k: u32, ref_m: u32, stream: u64) -> Result<()> {
    let ctx = ctx()?;
    if PINNED.lock().unwrap().iter().any(|e| {
        e.0 == (
            n,
            k,
            Operands::Mxfp8 {
                weight_sf: 0,
                act_sf: 0,
            }
            .kind(),
        )
    }) {
        return Ok(());
    }
    anyhow::ensure!(
        ref_m > 0 && k.is_multiple_of(32),
        "cuBLASLt MXFP8 pin: {ref_m}x{n}x{k}"
    );
    let rows = ref_m.div_ceil(128) as usize * 128;
    let kpad = (k as usize / 32).div_ceil(4) * 4;
    let n_pad = (n as usize).div_ceil(128) * 128;
    let sizes = [
        (ref_m as usize * k as usize, 0u8),     // act
        (rows * kpad, 127u8),                   // act scales (2^0)
        (n as usize * k as usize, 0u8),         // weight
        (n_pad * kpad, 127u8),                  // weight scales
        (ref_m as usize * n as usize * 2, 0u8), // out
    ];
    let mut bufs = [0u64; 5];
    let result = (|| -> Result<()> {
        for (b, &(bytes, fill)) in bufs.iter_mut().zip(&sizes) {
            // SAFETY: allocation and fill of a fresh device buffer.
            unsafe {
                chk(cuMemAlloc_v2(b, bytes), "pin alloc")?;
                chk(cuMemsetD8Async(*b, fill, bytes, stream), "pin fill")?;
            }
        }
        let [act, act_sf, weight, weight_sf, out] = bufs;
        let ops = Operands::Mxfp8 { weight_sf, act_sf };
        pin(ctx, n, k, ops, Some(([weight, act, out], ref_m, stream)))?;
        // SAFETY: plain stream sync before the buffers are freed.
        unsafe { chk(cuStreamSynchronize(stream), "pin sync") }
    })();
    for b in bufs.into_iter().filter(|&b| b != 0) {
        // SAFETY: allocated above, no longer in use (synchronized or failed).
        unsafe {
            cuMemFree_v2(b);
        }
    }
    result
}
