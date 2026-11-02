// SPDX-License-Identifier: AGPL-3.0-only

//! Serial mHC FFN continuation; this seam does not own or detach GPU storage.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{Qwen3AttentionLayer, ctx};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// These pointers name the shared BufferArena highway/post/comb spans. The
/// normalized FFN input is likewise shared at MultiSeqCtx::normed. A future
/// paired driver MUST preserve each owner's normalized rows and all three mHC
/// spans before another attention phase overwrites them, and restore or supply
/// checked owner-specific views before continuation. Merely holding this value
/// does not make two interleaved owners safe. It is intentionally not Clone.
pub(super) struct HcFfnPhase {
    pub(super) hc_streams: DevicePtr,
    pub(super) post: DevicePtr,
    pub(super) comb: DevicePtr,
    pub(super) diag_this: bool,
}

impl Qwen3AttentionLayer {
    pub(super) fn ms_hc_ffn_post(
        &self,
        c: &ctx::MultiSeqCtx<'_>,
        phase: HcFfnPhase,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        let HcFfnPhase {
            hc_streams,
            post,
            comb,
            diag_this,
        } = phase;
        let independent = crate::model::glm_independent::ffn_rows_selected(ctx, n)?;
        let glm_batched_ffn = ctx.config.model_type == "glm5_next" && matches!(n, 3..=5);
        let compact_c2 = if !independent && n == 2 {
            self.ffn.try_forward_c2_compact(c.normed, ctx, stream)?
        } else {
            None
        };
        if independent || compact_c2.is_some() || glm_batched_ffn {
            let (moe_out, deferred_shared_gate) = if independent {
                (
                    self.ffn.forward_independent(c.normed, n, ctx, stream)?,
                    None,
                )
            } else if let Some(output) = compact_c2 {
                (output, None)
            } else if n == 3 {
                self.ffn.forward_k3(c.normed, ctx, stream)?;
                (ctx.buffers.moe_output(), None)
            } else if n == 4 {
                (self.ffn.forward_c4(c.normed, ctx, stream)?, None)
            } else {
                self.ffn.forward_k5_for_hc(
                    c.normed,
                    self.hc_post_moe_blend_k.0 != 0,
                    ctx,
                    stream,
                )?
            };
            if let Some(gate_weight) = deferred_shared_gate {
                ops::hc_post_moe_blend(
                    ctx.gpu,
                    self.hc_post_moe_blend_k,
                    moe_out,
                    ctx.buffers.attn_output(),
                    c.normed,
                    gate_weight,
                    hc_streams,
                    post,
                    comb,
                    hc_streams,
                    n as u32,
                    h as u32,
                    hc_mult,
                    stream,
                )?;
            } else {
                ops::hc_post(
                    ctx.gpu,
                    self.hc_post_k,
                    moe_out,
                    hc_streams,
                    post,
                    comb,
                    hc_streams,
                    n as u32,
                    h as u32,
                    hc_mult,
                    stream,
                )?;
            }
        } else {
            for i in 0..n {
                let normed2_i = c.normed.offset(i * c.h * c.bf16);
                let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
                // hc_streams is the FP32 mHC highway (4 bytes/elem), not BF16.
                let hc_streams_i = hc_streams.offset(i * hc.hc_mult * c.h * 4);
                let post_i = post.offset(i * hc.hc_mult * 4);
                let comb_i = comb.offset(i * hc.hc_mult * hc.hc_mult * 4);
                ops::hc_post(
                    ctx.gpu,
                    self.hc_post_k,
                    moe_out,
                    hc_streams_i,
                    post_i,
                    comb_i,
                    hc_streams_i,
                    1,
                    h as u32,
                    hc_mult,
                    stream,
                )?;
            }
        }
        self.ms_hc_finish(c, hc_streams, diag_this, ctx, stream)
    }

    pub(super) fn ms_hc_supplied_post(
        &self,
        c: &ctx::MultiSeqCtx<'_>,
        phase: HcFfnPhase,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let HcFfnPhase {
            hc_streams,
            post,
            comb,
            diag_this,
        } = phase;
        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            output,
            hc_streams,
            post,
            comb,
            hc_streams,
            c.n as u32,
            c.h as u32,
            self.hc.as_ref().unwrap().hc_mult as u32,
            stream,
        )?;
        self.ms_hc_finish(c, hc_streams, diag_this, ctx, stream)
    }

    fn ms_hc_finish(
        &self,
        c: &ctx::MultiSeqCtx<'_>,
        hc_streams: DevicePtr,
        diag_this: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        let is_last_layer = self.block_idx + 1 == ctx.config.num_hidden_layers;
        if diag_this {
            super::super::diag_norm(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::super::diag_norm(
                ctx.gpu,
                hc_streams,
                n * (hc_mult as usize) * h,
                stream,
                &format!(
                    "V4-msdecode L{} hc_post-ffn ALL_STREAMS",
                    self.attn_layer_idx
                ),
            );
        }

        if is_last_layer && let Some(ref head) = hc.head {
            ops::hc_head(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head.hc_fn,
                head.hc_scale,
                head.hc_base,
                c.hidden,
                n as u32,
                h as u32,
                hc_mult,
                eps,
                hc.hc_eps,
                stream,
            )?;
            if diag_this {
                super::super::diag_norm(
                    ctx.gpu,
                    c.hidden,
                    n * h,
                    stream,
                    &format!("V4-msdecode L{} hc_head", self.attn_layer_idx),
                );
            }
        } else if is_last_layer {
            tracing::warn!(
                "V4-msdecode L{}: hc_head SKIPPED (no head weights)",
                self.attn_layer_idx
            );
        }

        Ok(())
    }
}
