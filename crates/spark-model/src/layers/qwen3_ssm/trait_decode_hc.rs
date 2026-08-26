// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN single-token decode under an mHC highway.
//!
//! The prefill twin of this lives in `trait_prefill_hc.rs`; the reasoning
//! about why the highway cannot be bolted onto `decode_inner` is there, and
//! applies unchanged: `decode_inner` maintains a single `residual` through
//! `rms_norm_residual` / `residual_add_rms_norm` / `residual_add`, and under
//! mHC each of those would add the block output a second time.
//!
//! Decode is the simpler of the two — `ssm_forward` is already a
//! residual-free function of `normed`, so no extraction was needed on this
//! side.

use super::*;

impl Qwen3SsmLayer {
    pub(super) fn decode_inner_hc(
        &self,
        hidden: DevicePtr,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hc = self
            .hc
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("decode_inner_hc without mHC weights"))?;
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc_mult = hc.hc_mult as u32;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        // ATLAS_FP32_ROUTING has the SSM's fused `residual_add_rms_norm_gatef32`
        // populate `moe_router_in_f32` for the gate GEMM to read at full
        // precision. This path does not run that kernel — `hc_norm` inside
        // `hc_pre` is the norm — so the buffer would hold the PREVIOUS layer's
        // activations and the router would route on them. Latent today (the
        // flag is off by default) and silent if it ever is not.
        anyhow::ensure!(
            !self.ffn.fp32_routing_active(),
            "qwen3_ssm mHC: ATLAS_FP32_ROUTING needs the fused gate-f32 norm, \
             which the highway path replaces. The router would read a stale \
             moe_router_in_f32. Unset it."
        );

        let streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();

        if hc.is_first_model_layer {
            ops::hc_expand(
                ctx.gpu,
                self.hc_expand_k,
                hidden,
                streams,
                1,
                h as u32,
                hc_mult,
                stream,
            )?;
        }

        // Decode never starts a sequence — prefill did — so `fresh` is false
        // and the conv state plus the 2-token history carry from the last step.
        if let Some(ple) = self.ple.as_ref() {
            ple.forward(streams, 1, false, ctx, stream)?;
        }

        // ── GDN sublayer. `hidden` is scratch; the highway carries state. ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.attn,
            hc,
            hidden,
            post,
            comb,
            1,
            h as u32,
            eps,
            stream,
        )?;
        // No `input_norm`: `hc_norm` inside `hc_pre` is this layer's norm.
        // The checkpoint carries no per-layer norms and the loader's
        // ones-placeholder would NOT make a second RMS pass an identity.
        let ssm_out = self.ssm_forward(hidden, ssm_state, ctx, stream, false)?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            ssm_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            stream,
        )?;

        // ── MoE sublayer ──
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            &hc.ffn,
            hc,
            hidden,
            post,
            comb,
            1,
            h as u32,
            eps,
            stream,
        )?;
        let moe_out = self.ffn.forward(hidden, ctx, stream)?;
        ops::hc_post_site(
            ctx.gpu,
            self.hc_post_k,
            hc,
            moe_out,
            streams,
            post,
            comb,
            streams,
            1,
            h as u32,
            stream,
        )?;

        Ok(())
    }
}
