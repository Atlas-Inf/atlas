// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence / verify Q, K, V on native MXFP8 weights (FP4-Local).
//!
//! The same shape as `qkv_fp8.rs`: each projection runs once over all `n`
//! rows into a contiguous scratch (one weight pass per eight rows, every row
//! bit-identical to single-token decode), the gated `[Q|gate]` rows are
//! split, then rows are copied into the per-sequence `qkv_buf` layout.

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// `Ok(false)`, nothing launched, unless the layer's projections are MXFP8.
    pub(super) fn ms_qkv_batch_mx8(&self, c: &MultiSeqCtx<'_>) -> Result<bool> {
        if self.q_weight.as_ref().and_then(|w| w.as_mxfp8()).is_none() {
            return Ok(false);
        }
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_dim,
            q_proj_bytes,
            per_seq_qkv,
            normed,
            qkv_buf,
            ..
        } = *c;
        let q_scratch = fwd.buffers.ssm_qkvz();
        let kv_bytes = (nkv * hd) as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        for (w, out) in [
            (self.q_weight.as_ref(), q_scratch),
            (self.k_weight.as_ref(), k_scratch),
            (self.v_weight.as_ref(), v_scratch),
        ] {
            self.try_mx8(fwd, w, normed, out, n, stream)
                .unwrap_or_else(|| anyhow::bail!("MXFP8 q_proj with a non-MXFP8 k/v_proj"))?;
        }
        if self.gated && !self.q_lora_active() {
            ops::deinterleave_qg(
                fwd.gpu,
                self.deinterleave_qg_k,
                q_scratch,
                n as u32,
                nq,
                hd,
                q_proj_dim,
                stream,
            )?;
        }
        for row in 0..n {
            let q_out = qkv_buf.offset(row * per_seq_qkv);
            let k_out = q_out.offset(q_proj_bytes);
            let v_out = k_out.offset(kv_bytes);
            let copy = |src: spark_runtime::gpu::DevicePtr, dst, bytes| {
                fwd.gpu.copy_d2d_async(src, dst, bytes, stream)
            };
            copy(q_scratch.offset(row * q_proj_bytes), q_out, q_proj_bytes)?;
            copy(k_scratch.offset(row * kv_bytes), k_out, kv_bytes)?;
            copy(v_scratch.offset(row * kv_bytes), v_out, kv_bytes)?;
        }
        Ok(true)
    }
}
