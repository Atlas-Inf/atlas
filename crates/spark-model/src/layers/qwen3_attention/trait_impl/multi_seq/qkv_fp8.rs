// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_qkv_batchm_fp8(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
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
        let q = self.q_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
        let k = self.k_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
        let v = self.v_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
        let q_scratch = fwd.buffers.ssm_qkvz();
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        let batch_kernel = if n <= 4 {
            self.w8a16_gemv_batch4_k
        } else {
            self.w8a16_gemv_batch16_k
        };

        let project = |weight: &crate::weight_map::Fp8Weight, output, rows| {
            ops::w8a16_gemv_batch4(
                fwd.gpu,
                batch_kernel,
                normed,
                weight.weight,
                weight.row_scale,
                output,
                n as u32,
                rows,
                h as u32,
                stream,
            )
        };
        project(q, q_scratch, q_proj_dim)?;
        project(k, k_scratch, kv_dim)?;
        project(v, v_scratch, kv_dim)?;

        let q_lora_active = self
            .lora
            .as_ref()
            .and_then(|weights| weights.q.as_ref())
            .is_some();
        if self.gated && !q_lora_active {
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
            fwd.gpu.copy_d2d_async(
                q_scratch.offset(row * q_proj_bytes),
                q_out,
                q_proj_bytes,
                stream,
            )?;
            fwd.gpu
                .copy_d2d_async(k_scratch.offset(row * kv_bytes), k_out, kv_bytes, stream)?;
            fwd.gpu
                .copy_d2d_async(v_scratch.offset(row * kv_bytes), v_out, kv_bytes, stream)?;
        }
        Ok(())
    }
}
