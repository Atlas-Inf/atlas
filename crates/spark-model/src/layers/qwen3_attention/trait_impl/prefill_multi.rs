// SPDX-License-Identifier: AGPL-3.0-only

//! The full-attention layer of a multi-sequence prefill pass under an mHC
//! highway (`ATLAS_QWEN4EXP_PREFILL_MULTI`): `prefill_inner_hc` over several
//! sequences' rows. The mHC collapses, the TP reduce and the MoE run once
//! over every row; the attention block (q/k/v, norms, RoPE, the KV write, the
//! QSA ingest, attention, the gate, `o_proj`) per sequence through the paged
//! path the row-invariant prefill takes in every chunk, its output gathered
//! into `moe_output`. Each sequence's prompt sits under the QSA inert bound
//! (dense attention), so no QSA selection runs.

use anyhow::Result;

use super::super::Qwen3AttentionLayer;
use crate::layer::MultiPass;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(in crate::layers::qwen3_attention) fn prefill_multi_hc(
        &self,
        pass: &mut MultiPass<'_, '_>,
    ) -> Result<()> {
        let (hidden, total, ctx, stream) = (pass.hidden, pass.total, pass.ctx, pass.stream);
        let (segs, kv_cache) = (&mut pass.segs, &mut *pass.kv_cache);
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("multi-sequence attention prefill without mHC"))?;
        anyhow::ensure!(
            !ops::HcVariant::of(hc).applies_block_input_norm()
                && !self.ffn.is_none()
                && self.mla.is_none()
                && !self.high_speed_swap_engaged(kv_cache),
            "multi-sequence attention prefill serves the qwen4_exp low-rank highway layer only"
        );
        let (h, eps, n) = (
            ctx.config.hidden_size,
            ctx.config.rms_norm_eps as f32,
            total as u32,
        );
        let (is_first_layer, is_last_layer) = self.hc_prefill_layer_bounds(hc, ctx);
        let (streams, post, comb) = (
            ctx.buffers.hc_streams(),
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
        );
        if is_first_layer {
            ops::qwen4exp_prefill_seam::clear_pending();
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                n,
                h as u32,
                hc.hc_mult as u32,
                stream,
            )?;
        }
        // ── Attention sublayer ── (the previous layer's deferred post fused in)
        if !ops::qwen4exp_prefill_seam::pre_with_pending(
            ctx.gpu,
            self.hc_post_k,
            hc,
            &hc.attn,
            streams,
            hidden,
            post,
            ctx.buffers.hc_lowrank_scratch(),
            n,
            h as u32,
            eps,
            stream,
        )? {
            self.hc_pre_prefill_site(&hc.attn, hc, hidden, n, ctx, stream)?;
        }
        // Each sequence's block reads its rows of the collapse in `hidden`
        // (the single pass reads the same bytes copied to `norm_output`, which
        // the block's `o_proj` overwrites) and leaves its output in
        // `norm_output`, gathered here.
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        det.tap("in", hidden, (0, total), h * 2);
        let attn_out = ctx.buffers.moe_output();
        for seg in segs.iter_mut() {
            let x = hidden.offset(seg.row0 * h * 2);
            crate::det_trace::set_chunk_start(seg.row0);
            if let Some(ref qsa) = self.qsa {
                anyhow::ensure!(
                    seg.start + seg.rows <= qsa.inert_bound(),
                    "multi-sequence prefill: a sequence past the QSA inert bound"
                );
                let st = crate::layers::qwen3_attention::helpers::qsa_seq_state(
                    qsa,
                    &mut *seg.state,
                    ctx.gpu,
                )?;
                qsa.prefill_ingest(st, x, seg.rows, seg.start, ctx.gpu, stream)?;
            }
            let o = self.prefill_attention_paged(
                &mut *seg.state,
                x,
                seg.rows,
                seg.start,
                kv_cache,
                seg.block_table,
                seg.disk_block_ids,
                seg.disk_last_offloaded_per_layer,
                None,
                seg.kv_write_start,
                seg.ctx,
                stream,
            )?;
            ctx.gpu.copy_d2d_async(
                o,
                attn_out.offset(seg.row0 * h * 2),
                seg.rows * h * 2,
                stream,
            )?;
        }
        crate::det_trace::set_chunk_start(0);
        det.tap("attn", attn_out, (0, total), h * 2);
        if ctx.config.tp_world_size > 1
            && let Some(comm) = ctx.comm
        {
            comm.all_reduce_async(attn_out.0, total * h * 2, stream)?;
        }
        if let Some(ref post_norm) = self.post_attn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                attn_out,
                post_norm,
                attn_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        // ── FFN sublayer ──
        let normed2 = ctx.buffers.norm_output();
        if !self.hc_post_pre_prefill_seam(hc, attn_out, hidden, n, false, ctx, stream)? {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                attn_out,
                streams,
                post,
                comb,
                streams,
                n,
                h as u32,
                stream,
            )?;
            self.hc_pre_prefill_site(&hc.ffn, hc, hidden, n, ctx, stream)?;
        }
        ctx.gpu
            .copy_d2d_async(hidden, normed2, total * h * 2, stream)?;
        self.ffn.forward_prefill(normed2, total, ctx, stream)?;
        let dense_out = ctx.buffers.moe_output();
        if let Some(ref post_norm) = self.post_ffn_out_norm {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                dense_out,
                post_norm,
                dense_out,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        let deferred = !is_last_layer
            && ops::qwen4exp_prefill_seam::defer_post(ctx.gpu, hc, dense_out, n, h as u32);
        if !deferred {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                dense_out,
                streams,
                post,
                comb,
                streams,
                n,
                h as u32,
                stream,
            )?;
        }
        if is_last_layer {
            let head = hc.head.as_ref().ok_or_else(|| {
                anyhow::anyhow!("multi-sequence prefill: last layer without hc_head")
            })?;
            ops::hc_head_site(
                ctx.gpu,
                self.hc_head_k,
                streams,
                head,
                hc,
                hidden,
                ctx.buffers.hc_lowrank_scratch(),
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        Ok(())
    }
}
