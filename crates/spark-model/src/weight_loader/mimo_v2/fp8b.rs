// SPDX-License-Identifier: AGPL-3.0-only

//! MiMo FP8 block QKV projections (`ATLAS_MIMO_FP8_QKV=1`).
//!
//! The checkpoint's `self_attn.qkv_proj` is E4M3 with an F32 scale per
//! 128x128 block (ModelOpt `FP8_PB_WO`), stored as 4 TP chunks whose row
//! counts are not all 128-aligned (full-attention chunks are 3392 rows).
//! The BF16 loader dequantizes chunk by chunk; this path keeps the
//! checkpoint's own bytes and scale grid on the GPU as an
//! [`Fp8BlockWeight`] for the `fp8b_gemv_tc*` decode/verify tiers and the
//! W8A16 prefill GEMM (`w8a16_gemm_pipelined`) — the same native path GLM's
//! FP8 block projections take, with the weights never rounded.
//!
//! Full-attention chunks are padded to the 128-row grid (3392 -> 3456 rows
//! per chunk, 64 headroom rows at each chunk end): the GEMM then sees one
//! plain `[n/128, k/128]` scale grid, and `mimo_qkv_rope_cache` already
//! strides chunks by `chunk_rows`, so the pad rows are never read.
//! Sliding-window chunks (3712 rows) are already aligned and copy through
//! untouched. Pad weight bytes are left uninitialized: the GEMV tiers are
//! row-invariant, so a pad row's output cannot contaminate a content row,
//! and the rope kernel never reads pad rows.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::{QKV_TP_CHUNKS, qkv_chunk_rows};
use crate::weight_map::{FP8_BLOCK, Fp8BlockWeight};

/// Opt-in switch: `ATLAS_MIMO_FP8_QKV=1` keeps QKV in native FP8 blocks.
pub(crate) const FP8_QKV_ENV: &str = "ATLAS_MIMO_FP8_QKV";

/// The switch as set (read once); unset means the BF16 path.
pub(crate) fn fp8_qkv_enabled() -> Result<bool> {
    static V: std::sync::OnceLock<std::result::Result<bool, String>> = std::sync::OnceLock::new();
    V.get_or_init(|| match std::env::var(FP8_QKV_ENV).ok().as_deref() {
        None | Some("") | Some("0") | Some("false") | Some("off") => Ok(false),
        Some("1") | Some("true") | Some("on") => Ok(true),
        Some(other) => Err(format!("{FP8_QKV_ENV} must be 1 or 0, got {other:?}")),
    })
    .clone()
    .map_err(anyhow::Error::msg)
}

/// Chunk rows padded to the FP8 block grid (3392 -> 3456, 3712 -> 3712).
pub(crate) fn padded_chunk_rows(chunk_rows: usize) -> usize {
    chunk_rows.div_ceil(FP8_BLOCK) * FP8_BLOCK
}

/// Load the TP-chunked fused QKV as a native FP8 block weight
/// (`[padded_rows, hidden]` E4M3 + F32 `[padded_rows/128, hidden/128]`
/// grid), returning the weight, the padded per-chunk row count the rope
/// kernel strides by and this rank's local chunk count.
///
/// `nq`/`nkv` are the PRE-TP (checkpoint) head counts. With `tp_size > 1`
/// only this rank's `QKV_TP_CHUNKS / tp_size` consecutive chunks are kept —
/// whole chunks with their own scale grids, so the 128-row block alignment
/// holds per chunk without any repacking.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_qkv_fp8b(
    store: &WeightStore,
    prefix: &str,
    nq: usize,
    nkv: usize,
    h: usize,
    gpu: &dyn GpuBackend,
    tp_rank: usize,
    tp_size: usize,
) -> Result<(Fp8BlockWeight, usize, usize)> {
    ensure!(
        tp_rank < tp_size && QKV_TP_CHUNKS.is_multiple_of(tp_size),
        "{prefix}: checkpoint {QKV_TP_CHUNKS} chunks not divisible by tp_size {tp_size}"
    );
    let local_chunks = QKV_TP_CHUNKS / tp_size;
    let chunk_rows = qkv_chunk_rows(nq, nkv);
    let rows = QKV_TP_CHUNKS * chunk_rows;
    let padded_chunk = padded_chunk_rows(chunk_rows);
    let padded_rows = local_chunks * padded_chunk;
    let (w_ptr, w_dtype, w_shape) = {
        let w = store.get(&format!("{prefix}.weight"))?;
        (w.ptr, w.dtype, w.shape.clone())
    };
    ensure!(
        w_shape == [rows, h],
        "{prefix}.weight: shape {w_shape:?}, expected [{rows}, {h}]"
    );
    if w_dtype != WeightDtype::FP8E4M3 {
        bail!("{prefix}.weight: {FP8_QKV_ENV} needs FP8 E4M3, got {w_dtype:?}");
    }
    let (s_ptr, s_dtype, s_shape) = {
        let s = store.get(&format!("{prefix}.weight_scale_inv"))?;
        (s.ptr, s.dtype, s.shape.clone())
    };
    let blocks_per_chunk = chunk_rows.div_ceil(FP8_BLOCK);
    let sk = h.div_ceil(FP8_BLOCK);
    ensure!(
        s_shape == [QKV_TP_CHUNKS * blocks_per_chunk, sk],
        "{prefix}.weight_scale_inv: shape {s_shape:?}, expected [{}, {sk}]",
        QKV_TP_CHUNKS * blocks_per_chunk
    );
    // The fp8b tiers read an F32 grid; the checkpoint ships F32.
    ensure!(
        s_dtype == WeightDtype::FP32,
        "{prefix}.weight_scale_inv: {FP8_QKV_ENV} needs F32 scales, got {s_dtype:?}"
    );
    // Padding adds only the partial block's remainder rows, so the padded
    // grid has exactly the checkpoint's block count for the kept chunks.
    ensure!(
        padded_rows / FP8_BLOCK == local_chunks * blocks_per_chunk,
        "{prefix}: padded grid has {} scale rows, checkpoint has {}",
        padded_rows / FP8_BLOCK,
        local_chunks * blocks_per_chunk
    );

    let data = gpu.alloc(padded_rows * h)?;
    let local_scale_rows = local_chunks * blocks_per_chunk * s_shape[1];
    let scales_bytes = local_scale_rows * 4;
    let scales = match gpu.alloc(scales_bytes) {
        Ok(p) => p,
        Err(e) => {
            gpu.free(data)?;
            return Err(e);
        }
    };
    let stream = gpu.default_stream();
    if padded_chunk == chunk_rows {
        // One contiguous row range when the local chunks cover the tensor;
        // otherwise per-chunk copies with the pad rows between them.
        if local_chunks == QKV_TP_CHUNKS {
            gpu.copy_d2d_async(w_ptr, data, rows * h, stream)?;
        } else {
            for j in 0..local_chunks {
                let c = tp_rank * local_chunks + j;
                gpu.copy_d2d_async(
                    w_ptr.offset(c * chunk_rows * h),
                    data.offset(j * chunk_rows * h),
                    chunk_rows * h,
                    stream,
                )?;
            }
        }
    } else {
        // Each chunk is contiguous in both layouts; the pad rows sit
        // between chunks in the destination only.
        for j in 0..local_chunks {
            let c = tp_rank * local_chunks + j;
            gpu.copy_d2d_async(
                w_ptr.offset(c * chunk_rows * h),
                data.offset(j * padded_chunk * h),
                chunk_rows * h,
                stream,
            )?;
        }
    }
    // The scale grid is row-major over (row block, k block); a chunk is a
    // contiguous band of `blocks_per_chunk` rows, so the local chunks' bands
    // are one contiguous range.
    let s_offset = (tp_rank * local_chunks * blocks_per_chunk * s_shape[1] * 4) as u64;
    gpu.copy_d2d_async(DevicePtr(s_ptr.0 + s_offset), scales, scales_bytes, stream)?;
    gpu.synchronize(stream)?;
    // Both source tensors are store-owned; reclaim records them so release
    // does not free the same pointers a second time.
    store.reclaim(gpu, &format!("{prefix}.weight"))?;
    store.reclaim(gpu, &format!("{prefix}.weight_scale_inv"))?;
    Ok((
        Fp8BlockWeight {
            data,
            scales,
            tiled: false,
        },
        padded_chunk,
        local_chunks,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_attention_chunks_pad_to_the_block_grid() {
        // 64 Q heads / 4 KV heads: 3392 rows per chunk.
        let chunk = qkv_chunk_rows(64, 4);
        assert_eq!(chunk, 3392);
        assert_eq!(padded_chunk_rows(chunk), 3456);
        assert_eq!(padded_chunk_rows(chunk) / FP8_BLOCK, 27);
    }

    #[test]
    fn sliding_window_chunks_are_already_aligned() {
        // 64 Q heads / 8 KV heads: 3712 rows per chunk.
        let chunk = qkv_chunk_rows(64, 8);
        assert_eq!(chunk, 3712);
        assert_eq!(padded_chunk_rows(chunk), chunk);
        assert_eq!(padded_chunk_rows(chunk) / FP8_BLOCK, 29);
    }
}
