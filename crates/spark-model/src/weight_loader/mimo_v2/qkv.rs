// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-layout loader for MiMo-V2's fused QKV projection.
//!
//! The checkpoint ships `qkv_proj` pre-sharded into four TP chunks
//! (`Q_c | K_c | V_c` per chunk). Atlas TP=2 keeps two whole chunks per rank
//! (chunk `c` holds Q heads `[c*16, (c+1)*16)` and the matching K/V head
//! range), so the row stride stays the shipped chunk stride and the rope
//! kernel's per-chunk math is unchanged.

use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::qwen3_attention::mimo::{MIMO_QK_DIM, MIMO_V_DIM};
use crate::weight_map::DenseWeight;

/// The published checkpoint stores fused QKV pre-sharded for TP=4.
pub(crate) const QKV_TP_CHUNKS: usize = 4;

/// Rows of one TP chunk of the fused QKV: `Q_c | K_c | V_c`.
pub(crate) fn qkv_chunk_rows(nq: usize, nkv: usize) -> usize {
    let qk = MIMO_QK_DIM as usize;
    let vd = MIMO_V_DIM as usize;
    (nq / QKV_TP_CHUNKS) * qk + (nkv / QKV_TP_CHUNKS) * (qk + vd)
}

/// Load the checkpoint's TP-chunked fused QKV as BF16
/// `[local_chunks * chunk_rows, hidden]`, keeping only this rank's
/// `QKV_TP_CHUNKS / tp_size` consecutive chunks (chunk `c` holds Q heads
/// `[c*16, (c+1)*16)` and the matching K/V head range, so TP=2 rank `r` takes
/// chunks `{2r, 2r+1}` — whole chunks, never split).
///
/// `nq`/`nkv` are the PRE-TP (checkpoint) head counts: the chunk row stride
/// is fixed by the shipped 4-chunk layout. Returns the weight, the per-chunk
/// row stride and the local chunk count for `MimoAttn::tp_chunks`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn load_qkv_chunked(
    store: &WeightStore,
    prefix: &str,
    nq: usize,
    nkv: usize,
    h: usize,
    gpu: &dyn GpuBackend,
    tp_rank: usize,
    tp_size: usize,
) -> Result<(DenseWeight, usize, usize)> {
    ensure!(
        tp_rank < tp_size && QKV_TP_CHUNKS.is_multiple_of(tp_size),
        "{prefix}: checkpoint {QKV_TP_CHUNKS} chunks not divisible by tp_size {tp_size}"
    );
    ensure!(
        nq.is_multiple_of(QKV_TP_CHUNKS) && nkv.is_multiple_of(QKV_TP_CHUNKS),
        "{prefix}: heads ({nq} q / {nkv} kv) not divisible by the {QKV_TP_CHUNKS} checkpoint chunks"
    );
    let local_chunks = QKV_TP_CHUNKS / tp_size;
    let chunk_rows = qkv_chunk_rows(nq, nkv);
    let rows = QKV_TP_CHUNKS * chunk_rows;
    let local_rows = local_chunks * chunk_rows;
    let (w_ptr, w_dtype, w_shape) = {
        let w = store.get(&format!("{prefix}.weight"))?;
        (w.ptr, w.dtype, w.shape.clone())
    };
    ensure!(
        w_shape == [rows, h],
        "{prefix}.weight: shape {w_shape:?}, expected [{rows}, {h}]"
    );
    match w_dtype {
        WeightDtype::BF16 => {
            if local_chunks == QKV_TP_CHUNKS {
                return Ok((DenseWeight { weight: w_ptr }, chunk_rows, local_chunks));
            }
            // TP shard: the local chunks are one contiguous row range of the
            // checkpoint tensor.
            let local_bytes = local_rows * h * 2;
            let dst = gpu.alloc(local_bytes)?;
            let src_offset = (tp_rank * local_chunks * chunk_rows * h) as u64;
            gpu.copy_d2d(DevicePtr(w_ptr.0 + src_offset), dst, local_bytes)?;
            return Ok((DenseWeight { weight: dst }, chunk_rows, local_chunks));
        }
        WeightDtype::FP8E4M3 => {}
        other => bail!("{prefix}.weight: unsupported dtype {other:?}"),
    }
    let (s_ptr, s_dtype, s_shape) = {
        let s = store.get(&format!("{prefix}.weight_scale_inv"))?;
        (s.ptr, s.dtype, s.shape.clone())
    };
    let blocks_per_chunk = chunk_rows.div_ceil(128);
    let sk = h.div_ceil(128);
    ensure!(
        s_shape == [QKV_TP_CHUNKS * blocks_per_chunk, sk],
        "{prefix}.weight_scale_inv: shape {s_shape:?}, expected [{}, {sk}] \
         ({QKV_TP_CHUNKS} chunks × {blocks_per_chunk} row blocks)",
        QKV_TP_CHUNKS * blocks_per_chunk
    );
    let scale_bytes = match s_dtype {
        WeightDtype::FP32 => 4,
        WeightDtype::BF16 => 2,
        other => bail!("{prefix}.weight_scale_inv: unsupported dtype {other:?}"),
    };
    let out = gpu.alloc(local_rows * h * 2)?;
    let kernel: KernelHandle = gpu.kernel(
        "dequant_fp8_blockscaled_bf16",
        "dequant_fp8_blockscaled_bf16",
    )?;
    let stream = gpu.default_stream();
    for j in 0..local_chunks {
        let c = tp_rank * local_chunks + j;
        KernelLaunch::new(gpu, kernel)
            .grid([div_ceil(h as u32, 64), div_ceil(chunk_rows as u32, 4), 1])
            .block([64, 4, 1])
            .arg_ptr(w_ptr.offset(c * chunk_rows * h))
            .arg_ptr(s_ptr.offset(c * blocks_per_chunk * sk * scale_bytes))
            .arg_ptr(out.offset(j * chunk_rows * h * 2))
            .arg_u32(chunk_rows as u32)
            .arg_u32(h as u32)
            .arg_u32(128)
            .arg_u32(128)
            .arg_u32(sk as u32)
            .arg_u32((scale_bytes == 4) as u32)
            .launch(stream)?;
    }
    gpu.synchronize(stream)?;
    // Both source tensors are store-owned; reclaim records them so release does
    // not free the same pointers a second time.
    store.reclaim(gpu, &format!("{prefix}.weight"))?;
    store.reclaim(gpu, &format!("{prefix}.weight_scale_inv"))?;
    Ok((DenseWeight { weight: out }, chunk_rows, local_chunks))
}
