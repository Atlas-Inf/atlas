// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-row Q/K/V for value-residual layers (Ricursive Ripple): dense BF16
//! q/k/v (or NVFP4 q|gate, opt-in) where V reads the blended
//! `lerp(x0, normed, blend)` rows.
//!
//! One BF16 GEMM per projection reads each weight once for all `n` rows (the
//! DFlash verify width), instead of the per-row GEMV loop gated BF16 layers
//! otherwise fall into; `set_value_residual` guarantees the dense BF16 weights. The contiguous outputs land in the per-row `qkv_buf`
//! layout through one pitched copy per projection, and the interleaved
//! `[Q_h | G_h]` rows are split in place.

use anyhow::Result;

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
        let kv_dim = nkv * hd;
        let kv_bytes = kv_dim as usize * bf16;
        let v_in = self.value_residual_input(fwd.gpu, normed, n, h, stream)?;

        let q_scratch = fwd.buffers.ssm_qkvz();
        let k_scratch = fwd.buffers.attn_output();
        let v_scratch = k_scratch.offset(n * kv_bytes);
        let (m, k) = (n as u32, h as u32);
        let gpu = fwd.gpu;
        // NVFP4 q|gate (`ATLAS_RIPPLE_Q_NVFP4=1`, no BF16 copy): the verify
        // GEMM on the transposed twin, into the scratch rows.
        let q_nvfp4 = self.q_decode_nvfp4.as_ref();
        // Else BF16 q|gate (the widest projection) straight into its qkv_buf rows.
        let q_direct = q_nvfp4.is_none().then(|| self.wide_bf16_gemv(n)).flatten();
        if let Some(qd) = q_nvfp4 {
            let qt = self.q_nvfp4_t.as_ref();
            self.wide_verify_gemm(c, normed, qd, qt, q_scratch, m, q_proj_dim, k)?;
        } else if let Some(gemv) = q_direct {
            let ldc = (per_seq_qkv / bf16) as u32;
            let q_w = self.attn.q_proj.weight;
            // Gated: the store splits [Q_h | G_h] (no deinterleave pass).
            let qg_hd = if self.gated { hd } else { 0 };
            let shape = [m, q_proj_dim, k];
            ops::bf16_gemv_m32(gpu, gemv, normed, q_w, qkv_buf, shape, ldc, qg_hd, stream)?;
        } else {
            ops::bf16_gemm(
                normed,
                self.attn.q_proj.weight.0,
                q_scratch,
                m,
                q_proj_dim,
                k,
                stream,
            )?;
        }
        ops::bf16_gemm(
            normed,
            self.attn.k_proj.weight.0,
            k_scratch,
            m,
            kv_dim,
            k,
            stream,
        )?;
        ops::bf16_gemm(
            v_in,
            self.attn.v_proj.weight.0,
            v_scratch,
            m,
            kv_dim,
            k,
            stream,
        )?;

        if q_direct.is_none() {
            gpu.copy_d2d_2d_async(
                q_scratch,
                q_proj_bytes,
                qkv_buf,
                per_seq_qkv,
                q_proj_bytes,
                n,
                stream,
            )?;
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
