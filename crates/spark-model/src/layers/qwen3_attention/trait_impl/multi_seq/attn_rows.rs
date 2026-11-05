// SPDX-License-Identifier: AGPL-3.0-only

//! Verify-row attention: the K rows of ONE sequence (DFlash verify) against
//! its paged BF16 cache in one prefill-style launch (`ricursive_verify_attn`),
//! reading each KV tile once per q head for all rows instead of once per row
//! and q head as the batched decode kernel does. Graph-safe: the KV extent is
//! read on device from the last row's `seq_len`.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// `Some(attn_out)` when this layer serves verify rows this way, else `None`.
    pub(super) fn ms_verify_rows_attention(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<Option<DevicePtr>> {
        let Some(kernel) = self.verify_rows_attn_k() else {
            return Ok(None);
        };
        if !c.single_seq_rows || self.kv_dtype != KvCacheDtype::Bf16 {
            return Ok(None);
        }
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            seq_lens,
            ..
        } = *c;
        ensure!(
            seq_lens.windows(2).all(|w| w[1] == w[0] + 1),
            "verify rows must be consecutive positions of one sequence"
        );
        // Q rows sit `per_seq_qkv` apart in qkv_buf; the kernel reads packed Q.
        let q = fwd.buffers.ssm_qkvz();
        let q_bytes = q_dim as usize * bf16;
        fwd.gpu.copy_d2d_2d_async(qkv_buf, per_seq_qkv, q, q_bytes, q_bytes, n, stream)?;
        let attn_out = fwd.buffers.attn_output();
        let last_len = meta.seq_len.offset((n - 1) * 4);
        let layer = self.attn_layer_idx;
        ops::verify_rows_attention_paged(
            fwd.gpu,
            kernel,
            q,
            kv_cache.k_pool_ptr(layer),
            kv_cache.v_pool_ptr(layer),
            attn_out,
            meta.block_table,
            last_len,
            [n as u32, nq, nkv, hd],
            bs,
            self.sliding_window.unwrap_or(0),
            self.effective_attn_scale(hd),
            stream,
        )?;
        Ok(Some(attn_out))
    }
}
