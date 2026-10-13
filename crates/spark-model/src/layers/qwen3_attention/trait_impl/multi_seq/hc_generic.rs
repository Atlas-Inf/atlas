// SPDX-License-Identifier: AGPL-3.0-only
//! Upstream non-GLM HC continuation, in its original FFN/post/head order.
use super::{Qwen3AttentionLayer, ctx, hc_ffn::HcFfnPhase, ops};
use crate::layer::ForwardContext;
use anyhow::Result;

impl Qwen3AttentionLayer {
    pub(super) fn ms_hc_generic_finish(
        &self,
        c: &ctx::MultiSeqCtx<'_>,
        ctx: &ForwardContext,
        stream: u64,
        phase: HcFfnPhase,
        is_last_layer: bool,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = c.n;
        let hc = self.hc.as_ref().unwrap();
        let hc_mult = hc.hc_mult as u32;
        let HcFfnPhase {
            hc_streams,
            post,
            comb,
            diag_this,
        } = phase;
        // ATLAS_QWEN4EXP_BATCH_FAST: the live rows through `forward`'s own
        // kernels with the EP all-reduce batched, then ONE elementwise post.
        // The host ids are the live rows (`decode_a2` pads past them).
        // Under BATCH_SMALL its EP blend may take the post too (`posted`).
        let exact_rows = if ctx.levers.qwen4exp_batch_fast {
            let active = ctx.host_token_ids.map_or(n, |t| t.len().min(n));
            let fold =
                ops::qwen4exp_decode_fuse::MoeHcPost::for_rows(ctx.levers, hc, hc_streams, post);
            self.ffn
                .forward_rows_padded(c.normed, n, active, fold, ctx, stream)?
        } else {
            None
        };
        if let Some((moe_out, false)) = exact_rows {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                moe_out,
                hc_streams,
                post,
                comb,
                hc_streams,
                n as u32,
                h as u32,
                stream,
            )?;
        }
        // Per-token sequential FFN (MLA models always take this path).
        for i in (0..n).filter(|_| exact_rows.is_none()) {
            let normed2_i = c.normed.offset(i * c.h * c.bf16);
            let moe_out = self.ffn.forward(normed2_i, ctx, stream)?;
            // hc_streams is the FP32 mHC highway (4 bytes/elem), not BF16.
            let hc_streams_i = hc_streams.offset(i * hc.hc_mult * c.h * 4);
            let post_i = post.offset(i * hc.hc_mult * 4);
            let comb_i = comb.offset(i * hc.hc_mult * hc.hc_mult * 4);
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                moe_out,
                hc_streams_i,
                post_i,
                comb_i,
                hc_streams_i,
                1,
                h as u32,
                stream,
            )?;
        }
        if diag_this {
            super::super::diag_norm_f32(
                ctx.gpu,
                hc_streams,
                h,
                stream,
                &format!("V4-msdecode L{} hc_post-ffn", self.attn_layer_idx),
            );
            super::super::diag_norm_f32(
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
            ops::hc_head_site_rows(
                ctx.gpu,
                self.hc_head_k,
                hc_streams,
                head,
                hc,
                c.hidden,
                ctx.buffers.hc_lowrank_scratch(),
                n as u32,
                h as u32,
                eps,
                ctx.levers.qwen4exp_hc_rows(),
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
