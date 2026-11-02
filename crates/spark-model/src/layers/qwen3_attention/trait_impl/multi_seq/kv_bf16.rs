// SPDX-License-Identifier: AGPL-3.0-only

//! Exact BF16 K/V projection for Qwen4 MTP prompt absorption. Groups of up
//! to eight rows use the scalar GEMV's K order and reduction tree. Q is
//! unused by KV-only absorption; its slots remain untouched here. Gated Q
//! uses a wider Q region, but K norm, RoPE and cache writes never consume Q,
//! exactly as in the existing NVFP4 KV-only path.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{ctx::MultiSeqCtx, qkv::bf16_batchm_enabled};
use crate::layers::{ops, qwen3_attention::Qwen3AttentionLayer};
use crate::weight_map::DenseWeight;

impl Qwen3AttentionLayer {
    pub(super) fn kv_bf16_batchm_supported(&self, model_type: &str) -> bool {
        model_type == "qwen4_exp"
            && self.lora.is_none()
            && self.q_weight.is_none()
            && self.k_weight.is_none()
            && self.v_weight.is_none()
            && self.dense_gemv_batchm_k.0 != 0
            && !self.attn.k_proj.weight.is_null()
            && !self.attn.v_proj.weight.is_null()
    }

    pub(super) fn ms_kv_batchm_bf16(&self, c: &MultiSeqCtx<'_>) -> Result<bool> {
        if !self.kv_bf16_batchm_supported(&c.fwd.config.model_type)
            || !bf16_batchm_enabled()
            || c.fwd.levers.batch_bisect(ops::BISECT_ATTN_PROJ)
        {
            return Ok(false);
        }
        project_rows(
            c.fwd.gpu,
            self.dense_gemv_batchm_k,
            c.normed,
            [&self.attn.k_proj, &self.attn.v_proj],
            c.qkv_buf,
            c.n,
            c.h,
            (c.nkv * c.hd) as usize,
            c.q_proj_bytes / c.bf16,
            c.per_seq_qkv / c.bf16,
            c.stream,
        )?;
        Ok(true)
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project_rows(
    gpu: &dyn GpuBackend,
    kernel: KernelHandle,
    input: DevicePtr,
    weights: [&DenseWeight; 2],
    qkv: DevicePtr,
    rows: usize,
    hidden: usize,
    kv_dim: usize,
    q_dim: usize,
    stride: usize,
    stream: u64,
) -> Result<()> {
    let max_rows = ops::DENSE_GEMV_BATCHM_MAX_M as usize;
    for start in (0..rows).step_by(max_rows) {
        let count = (rows - start).min(max_rows);
        for (projection, weight) in weights.iter().enumerate() {
            ops::dense_gemv_batchm(
                gpu,
                kernel,
                input.offset(start * hidden * 2),
                weight,
                qkv.offset((start * stride + q_dim + projection * kv_dim) * 2),
                count as u32,
                kv_dim as u32,
                hidden as u32,
                stride as u32,
                stream,
            )?;
        }
    }
    Ok(())
}
