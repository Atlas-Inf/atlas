// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-row residual + FFN for PeriNorm blocks (post-attention and post-FFN
//! output norms inside the residual: Ricursive Ripple). Mirrors
//! `decode_inner`'s per-token sequence over all `n` rows; the rest of the
//! `ms_phase_ffn` ladder never applied either output norm.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::ctx::MultiSeqCtx;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

impl Qwen3AttentionLayer {
    /// Dense PeriNorm blocks without Gemma-4's layer scalar or dual FFN.
    /// Gemma-4 keeps its existing multi-row behaviour.
    pub(super) fn has_post_sublayer_norms(&self) -> bool {
        (self.post_attn_out_norm.is_some() || self.post_ffn_out_norm.is_some())
            && self.layer_scalar.is_none()
            && self.moe_ffn.is_none()
    }

    pub(super) fn ms_phase_ffn_perinorm(&self, c: &MultiSeqCtx<'_>, o_out: DevicePtr) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            h,
            eps,
            hidden,
            residual,
            ..
        } = *c;
        let (Some(post_attn), Some(post_ffn)) = (&self.post_attn_out_norm, &self.post_ffn_out_norm)
        else {
            anyhow::bail!("PeriNorm multi-row FFN needs both post-sublayer norms");
        };
        ensure!(
            self.ffn.is_dense()
                && self.moe_ffn.is_none()
                && self.layer_scalar.is_none()
                && self.mla.is_none()
                && self.hc.is_none(),
            "PeriNorm multi-row FFN covers dense blocks only (no MoE/layer scalar/MLA/HC)"
        );
        let (gpu, rows, width) = (fwd.gpu, n as u32, h as u32);
        ops::rms_norm(gpu, self.rms_norm_w_k, o_out, post_attn, o_out, rows, width, eps, stream)?;
        let normed2 = fwd.buffers.norm_output();
        ops::residual_add_rms_norm(
            gpu,
            self.residual_add_rms_norm_k,
            hidden,
            o_out,
            &self.post_attn_norm,
            normed2,
            residual,
            rows,
            width,
            eps,
            stream,
        )?;
        self.ffn.forward_prefill(normed2, n, fwd, stream)?;
        let ffn_out = fwd.buffers.moe_output();
        ops::rms_norm(gpu, self.rms_norm_w_k, ffn_out, post_ffn, ffn_out, rows, width, eps, stream)?;
        ops::residual_add(gpu, self.residual_add_k, hidden, ffn_out, rows * width, stream)
    }
}
