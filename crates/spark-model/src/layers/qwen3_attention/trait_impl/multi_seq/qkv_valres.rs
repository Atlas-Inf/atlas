// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-row Q/K/V for value-residual layers (Ricursive Ripple): dense BF16
//! q/k/v where V reads the blended `lerp(x0, normed, blend)` rows.
//!
//! One BF16 GEMM per projection reads each weight once for all `n` rows (the
//! DFlash verify width), instead of the per-row GEMV loop gated BF16 layers
//! otherwise fall into. The contiguous outputs land in the per-row `qkv_buf`
//! layout through one pitched copy per projection, and the interleaved
//! `[Q_h | G_h]` rows are split in place.

use anyhow::{Result, ensure};

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    pub(super) fn ms_qkv_valres_bf16(&self, c: &MultiSeqCtx<'_>) -> Result<()> {
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
        ensure!(
            self.qkv_is_dense_bf16() && self.lora.is_none(),
            "value-residual Q/K/V needs dense BF16 projections without LoRA"
        );
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let v_in = self.value_residual_input(fwd.gpu, normed, n, h, stream)?;

        let q_scratch = fwd.buffers.ssm_qkvz();
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        let (m, k) = (n as u32, h as u32);
        let gpu = fwd.gpu;
        // q|gate (the widest projection) straight into its qkv_buf rows.
        let q_direct = self.wide_bf16_gemv(n);
        if let Some(gemv) = q_direct {
            let ldc = (per_seq_qkv / bf16) as u32;
            let q_w = self.attn.q_proj.weight;
            // Gated: the store splits [Q_h | G_h] (no deinterleave pass).
            let qg_hd = if self.gated { hd } else { 0 };
            let shape = [m, q_proj_dim, k];
            ops::bf16_gemv_m32(gpu, gemv, normed, q_w, qkv_buf, shape, ldc, qg_hd, stream)?;
        } else {
            ops::bf16_gemm(normed, self.attn.q_proj.weight.0, q_scratch, m, q_proj_dim, k, stream)?;
        }
        ops::bf16_gemm(normed, self.attn.k_proj.weight.0, k_scratch, m, kv_dim, k, stream)?;
        ops::bf16_gemm(v_in, self.attn.v_proj.weight.0, v_scratch, m, kv_dim, k, stream)?;

        if q_direct.is_none() {
            gpu.copy_d2d_2d_async(q_scratch, q_proj_bytes, qkv_buf, per_seq_qkv, q_proj_bytes, n, stream)?;
        }
        let k_dst = qkv_buf.offset(q_proj_bytes);
        gpu.copy_d2d_2d_async(k_scratch, kv_bytes, k_dst, per_seq_qkv, kv_bytes, n, stream)?;
        let v_dst = k_dst.offset(kv_bytes);
        gpu.copy_d2d_2d_async(v_scratch, kv_bytes, v_dst, per_seq_qkv, kv_bytes, n, stream)?;
        // The direct q GEMV already stored [Q_all | G_all].
        if self.gated && q_direct.is_none() {
            ops::deinterleave_qg(
                gpu,
                self.deinterleave_qg_k,
                qkv_buf,
                m,
                nq,
                hd,
                (per_seq_qkv / bf16) as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
