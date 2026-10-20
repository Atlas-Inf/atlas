// SPDX-License-Identifier: AGPL-3.0-only

//! FFN dispatch and post-mHC continuation of the same owner's attention phase.

use super::*;

impl Glm5KdaLayer {
    pub(super) fn forward_ffn(
        &self,
        phase: FfnPhase,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (normed, tokens, decode, capture_verify_intermediates) = (
            phase.normed,
            phase.tokens,
            phase.decode,
            phase.capture_verify_intermediates,
        );
        let mut deferred_shared_gate = None;
        let sp = crate::layers::glm_sp::current()
            .filter(|sp| !decode && !capture_verify_intermediates && tokens == sp.total());
        let ffn_out = if let Some(sp) = sp {
            self.ffn.forward_prefill_sp(normed, sp, ctx, stream)?
        } else if capture_verify_intermediates && tokens == 2 {
            self.ffn.forward_k2(normed, ctx, stream)?;
            ctx.buffers.moe_output()
        } else if capture_verify_intermediates && tokens == 3 {
            self.ffn.forward_k3(normed, ctx, stream)?;
            ctx.buffers.moe_output()
        } else if capture_verify_intermediates && tokens == 4 && kda_batched_ffn_enabled() {
            // Dense layers use one batch4 GEMV; GLM MoE layers preserve the
            // parallel K2 routed path while evaluating the shared expert once.
            self.ffn.forward_k4(normed, ctx, stream)?
        } else if capture_verify_intermediates && tokens == 5 && kda_batched_ffn_enabled() {
            let (out, gate) =
                self.ffn
                    .forward_k5_for_hc(normed, self.hc_post_moe_blend_k.0 != 0, ctx, stream)?;
            deferred_shared_gate = gate;
            out
        } else if capture_verify_intermediates
            && crate::model::glm_independent::ffn_rows_selected(ctx, tokens)?
        {
            // GLM DFlash verify block: one exact grouped pass over its rows.
            self.ffn.forward_independent(normed, tokens, ctx, stream)?
        } else if capture_verify_intermediates {
            self.ffn.forward_batched(normed, tokens, ctx, stream)?;
            ctx.buffers.moe_output()
        } else if decode {
            self.ffn.forward(normed, ctx, stream)?
        } else {
            self.ffn.forward_prefill(normed, tokens, ctx, stream)?;
            ctx.buffers.moe_output()
        };
        self.forward_ffn_post(phase, ffn_out, deferred_shared_gate, ctx, stream)
    }

    pub(super) fn forward_ffn_post(
        &self,
        phase: FfnPhase,
        ffn_out: DevicePtr,
        deferred_shared_gate: Option<DevicePtr>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let FfnPhase {
            hidden,
            normed,
            tokens,
            decode,
            capture_verify_intermediates,
            mut profile_timer,
        } = phase;
        // Sequence-parallel prefill: `ffn_out` and the highway hold this
        // rank's rows; the contracted rows land at their chunk position.
        let sp = crate::layers::glm_sp::current()
            .filter(|sp| !decode && !capture_verify_intermediates && tokens == sp.total());
        let m = sp.map_or(tokens as u32, |sp| sp.rows as u32);
        let h = self.hidden_size as u32;
        profile::step(ctx, stream, &mut profile_timer, "ffn")?;
        if let Some(gate_weight) = deferred_shared_gate {
            let run_fused = || {
                ops::hc_post_moe_blend(
                    ctx.gpu,
                    self.hc_post_moe_blend_k,
                    ffn_out,
                    ctx.buffers.attn_output(),
                    normed,
                    gate_weight,
                    ctx.buffers.hc_streams(),
                    ctx.buffers.hc_post(),
                    ctx.buffers.hc_comb(),
                    ctx.buffers.hc_streams(),
                    m,
                    h,
                    self.hc.hc_mult as u32,
                    stream,
                )
            };
            if verify_fused_moe_hc_check_once() {
                let routed_bytes = tokens * self.hidden_size * size_of::<u16>();
                let highway_bytes = tokens * self.hc.hc_mult * self.hidden_size * size_of::<f32>();
                let residual_save = ctx.buffers.expert_gate_out();
                let routed_save = ctx.buffers.expert_up_out();
                let fused_save = ctx.buffers.expert_down_out();
                ctx.gpu.copy_d2d_async(
                    ctx.buffers.hc_streams(),
                    residual_save,
                    highway_bytes,
                    stream,
                )?;
                ctx.gpu
                    .copy_d2d_async(ffn_out, routed_save, routed_bytes, stream)?;
                run_fused()?;
                ctx.gpu.copy_d2d_async(
                    ctx.buffers.hc_streams(),
                    fused_save,
                    highway_bytes,
                    stream,
                )?;
                ctx.gpu.copy_d2d_async(
                    residual_save,
                    ctx.buffers.hc_streams(),
                    highway_bytes,
                    stream,
                )?;
                ctx.gpu
                    .copy_d2d_async(routed_save, ffn_out, routed_bytes, stream)?;
                self.ffn.finish_k5_deferred_shared_blend(
                    ffn_out,
                    ctx.buffers.attn_output(),
                    normed,
                    gate_weight,
                    ctx,
                    stream,
                )?;
                self.hc_post(ffn_out, m, ctx, stream)?;
                let mut got = vec![0u8; highway_bytes];
                let mut want = vec![0u8; highway_bytes];
                ctx.gpu.copy_d2h(fused_save, &mut got)?;
                ctx.gpu.copy_d2h(ctx.buffers.hc_streams(), &mut want)?;
                let mismatches = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                anyhow::ensure!(
                    mismatches == 0,
                    "GLM K=5 fused MoE blend/mHC differs from oracle in {mismatches}/{highway_bytes} bytes"
                );
                tracing::info!(
                    "GLM K=5 fused MoE blend/mHC: exact oracle match ({highway_bytes} bytes)"
                );
            } else {
                run_fused()?;
            }
        } else {
            self.hc_post(ffn_out, m, ctx, stream)?;
        }
        profile::step(ctx, stream, &mut profile_timer, "hc_ffn_post")?;

        if self.layer_idx + 1 == ctx.config.num_hidden_layers {
            ops::hc_contract(
                ctx.gpu,
                self.hc_contract_k,
                ctx.buffers.hc_streams(),
                sp.map_or(hidden, |sp| sp.local(hidden, self.hidden_size)),
                m,
                h,
                self.hc.hc_mult as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
