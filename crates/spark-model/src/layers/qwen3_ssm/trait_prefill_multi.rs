// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN layer of a multi-sequence prefill pass under an mHC highway
//! (`ATLAS_QWEN4EXP_PREFILL_MULTI`): `prefill_inner_hc` over several
//! sequences' rows. The mHC seams, the projections, gates, norms and the MoE
//! run once over every row (row-invariant under
//! `ATLAS_QWEN4EXP_PREFILL_ROWINV`); PLE, the conv and the recurrence per
//! sequence, on its own state. No sequence-parallel split, no mid-chunk
//! capture, no diagnostics taps.

use super::*;
use crate::layer::MultiPass;

impl Qwen3SsmLayer {
    /// Steps 2-10: QKVZ projection, conv1d, gates, the delta-rule recurrence,
    /// the gated norm, and `out_proj`. Returns the buffer holding
    /// `out_proj`'s output.
    ///
    /// `ssm_layer_idx` is passed in rather than re-fetched: it comes from a
    /// global call counter that must be bumped exactly once per layer per
    /// prefill, and both entry paths bump it before calling here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_block(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ssm_layer_idx: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_block_segs(
            normed,
            num_tokens,
            &mut [(0, num_tokens, state)],
            ssm_layer_idx,
            ctx,
            stream,
        )
    }

    pub(super) fn prefill_multi_hc(&self, pass: &mut MultiPass<'_, '_>) -> Result<()> {
        let (hidden, total, ctx, stream) = (pass.hidden, pass.total, pass.ctx, pass.stream);
        let segs = &mut pass.segs;
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("multi-sequence GDN prefill without mHC"))?;
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC: ATLAS_FP32_ROUTING is not served on the highway path"
        );
        let (h, eps, n) = (
            ctx.config.hidden_size,
            ctx.config.rms_norm_eps as f32,
            total as u32,
        );
        let ssm_layer_idx =
            super::debug::SSM_LAYER_CALL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let (streams, post, comb) = (
            ctx.buffers.hc_streams(),
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
        );
        let scratch = ctx.buffers.hc_lowrank_scratch();
        if hc.is_first_model_layer {
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
        if let Some(ple) = self.ple.as_ref() {
            ops::qwen4exp_prefill_seam::flush_pending(
                ctx.gpu,
                self.hc_post_k,
                hc,
                streams,
                post,
                h as u32,
                stream,
            )?;
            let row = hc.hc_mult * h * 4;
            for seg in segs.iter_mut() {
                let ssm = seg
                    .state
                    .as_any_mut()
                    .downcast_mut::<crate::layer::SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
                if ssm.ple.is_none() {
                    ssm.ple = Some(ple.new_seq_state(ctx.gpu)?);
                }
                let st = ssm.ple.as_mut().expect("just created");
                ple.forward(
                    st,
                    streams.offset(seg.row0 * row),
                    seg.rows,
                    seg.start == 0,
                    seg.ctx,
                    stream,
                )?;
            }
        }
        // ── GDN sublayer ── (the previous layer's deferred post fused in)
        if !ops::qwen4exp_prefill_seam::pre_with_pending(
            ctx.gpu,
            self.hc_post_k,
            hc,
            &hc.attn,
            streams,
            hidden,
            post,
            scratch,
            n,
            h as u32,
            eps,
            stream,
        )? {
            ops::hc_pre_site(
                ctx.gpu,
                self.hc_pre_k,
                streams,
                &hc.attn,
                hc,
                hidden,
                post,
                comb,
                scratch,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        crate::det_trace::on_stream(ctx.gpu, stream).tap("in", hidden, (0, total), h * 2);
        let mut rows: Vec<(usize, usize, &mut dyn LayerState)> = segs
            .iter_mut()
            .map(|s| (s.row0, s.rows, &mut *s.state as &mut dyn LayerState))
            .collect();
        let out_proj_buf =
            self.prefill_block_segs(hidden, total, &mut rows, ssm_layer_idx, ctx, stream)?;
        // ── MoE sublayer ──
        let seam = ops::qwen4exp_prefill_hc::hc_post_pre_seam(
            ctx.gpu,
            hc,
            &hc.ffn,
            out_proj_buf,
            streams,
            hidden,
            post,
            scratch,
            n,
            h as u32,
            eps,
            stream,
        )?;
        if !seam {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                out_proj_buf,
                streams,
                post,
                comb,
                streams,
                n,
                h as u32,
                stream,
            )?;
            ops::hc_pre_site(
                ctx.gpu,
                self.hc_pre_k,
                streams,
                &hc.ffn,
                hc,
                hidden,
                post,
                comb,
                scratch,
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        self.ffn.forward_prefill(hidden, total, ctx, stream)?;
        let moe_out = ctx.buffers.moe_output();
        if !ops::qwen4exp_prefill_seam::defer_post(ctx.gpu, hc, moe_out, n, h as u32) {
            ops::hc_post_site(
                ctx.gpu,
                self.hc_post_k,
                hc,
                moe_out,
                streams,
                post,
                comb,
                streams,
                n,
                h as u32,
                stream,
            )?;
        }
        Ok(())
    }
}

pub(super) fn ssm_state_of(state: &mut dyn LayerState) -> Result<&mut SsmLayerState> {
    state
        .as_any_mut()
        .downcast_mut::<SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))
}
