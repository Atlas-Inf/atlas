// SPDX-License-Identifier: AGPL-3.0-only

//! qwen4_exp batched verify step: per-row copies issued as one pitched copy.
//!
//! A C8 x K=4 verify step (32 rows) issued ~1,450 small device-to-device
//! copies, most of them one per ROW, inside the captured decode graphs and in
//! the eager tail (nsys, TP=EP=2, rank 0). This module holds the switches
//! that turn such per-row loops into a single pitched copy and the one helper
//! they share. Each switch is default off and exact by construction: row `r`
//! of a pitched copy is the bytes the loop's copy `r` moved (see
//! [`copy_rows`]).
//!
//! | switch | loop replaced | per step at C8 x K=4 |
//! |---|---|---|
//! | `ATLAS_QWEN4EXP_QKV_ROWS_2D=1` | the exact Q/K/V scatter into the interleaved `qkv_buf` (`multi_seq/qkv_exact4.rs`): 3 copies a row | 96 copies -> 3 a QSA layer (~1,000 -> ~36) |
//! | `ATLAS_QWEN4EXP_QSA_COMMIT_ROWS=1` | the staged QSA ingest's host-side commit (`decode_pieces.rs` -> `qsa_staged.rs`): a raw-key copy (and a pool launch when a block closes) a row, eagerly after the run | 32 copies -> one a sequence and layer (~370 -> ~96, eager) |
//!
//! A pitched copy is one `cudaMemcpy2DAsync`: one memcpy node when the step
//! is captured, one call when it runs eagerly.

use std::sync::OnceLock;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// `ATLAS_QWEN4EXP_QKV_ROWS_2D=1`, read once.
pub(crate) fn qkv_rows_2d() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_QKV_ROWS_2D").as_deref() == Ok("1"))
}

/// `ATLAS_QWEN4EXP_QSA_COMMIT_ROWS=1`, read once.
pub(crate) fn qsa_commit_rows() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_QWEN4EXP_QSA_COMMIT_ROWS").as_deref() == Ok("1"))
}

/// Copy `height` rows of `width` bytes, row `r` from `src + r * src_pitch` to
/// `dst + r * dst_pitch`: ONE pitched copy when `pitched` (and the shape is a
/// valid 2-D copy), else the per-row loop it replaces, verbatim.
///
/// The pitched form writes row `r` from exactly the address the loop's copy
/// `r` reads to exactly the address it writes, so the destination bytes are
/// identical. It needs `pitch >= width` on both sides so rows never overlap
/// (the `cudaMemcpy2D` precondition, which also makes row order irrelevant);
/// anything else takes the loop.
#[allow(clippy::too_many_arguments)]
pub(crate) fn copy_rows(
    gpu: &dyn GpuBackend,
    src: DevicePtr,
    src_pitch: usize,
    dst: DevicePtr,
    dst_pitch: usize,
    width: usize,
    height: usize,
    pitched: bool,
    stream: u64,
) -> Result<()> {
    if height == 0 || width == 0 {
        return Ok(());
    }
    if pitched && src_pitch >= width && dst_pitch >= width {
        return gpu.copy_d2d_2d_async(src, src_pitch, dst, dst_pitch, width, height, stream);
    }
    for r in 0..height {
        gpu.copy_d2d_async(
            src.offset(r * src_pitch),
            dst.offset(r * dst_pitch),
            width,
            stream,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "qwen4exp_step_copies_tests.rs"]
mod tests;
