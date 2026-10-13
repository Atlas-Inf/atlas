// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 mHC dispatch; batched prefill pre-mix is shared with the MLA sites.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Glm5KdaLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::qwen3_attention::HcSiteWeights;

fn parse_fast_prefill(value: Option<&str>) -> bool {
    matches!(value, Some("1" | "true" | "yes"))
}

fn fast_prefill(tokens: u32) -> bool {
    (tokens >= 128 && parse_fast_prefill(std::env::var("ATLAS_HC_CUBLAS_PREFILL").ok().as_deref()))
        || (tokens == 5
            && !crate::layers::canonical_verify::enabled()
            && parse_fast_prefill(std::env::var("ATLAS_GLM_K5_HC_CUBLAS").ok().as_deref()))
}

impl Glm5KdaLayer {
    pub(super) fn hc_pre(
        &self,
        site: &HcSiteWeights,
        hidden: DevicePtr,
        tokens: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if tokens <= 32
            && crate::layers::qwen3_attention::hc_post_pre_prefill_fused(
                site,
                None,
                hidden,
                tokens,
                self.hc.hc_mult as u32,
                self.hc.sinkhorn_iters as u32,
                self.hc.hc_eps,
                ctx,
                stream,
            )?
        {
            return Ok(());
        }
        if fast_prefill(tokens) {
            return crate::layers::qwen3_attention::hc_pre_prefill_mix(
                site,
                hidden,
                tokens,
                self.hc.hc_mult as u32,
                self.hc.sinkhorn_iters as u32,
                self.hc.hc_eps,
                self.hc_pre_from_raw_mix_k,
                ctx,
                stream,
            );
        }
        if ops::try_hc_pre_split(
            ctx.gpu,
            self.hc_pre_k,
            self.hc_pre_mix_k,
            self.hc_pre_from_raw_mix_k,
            ctx.buffers.gate_logits_f32(),
            ctx.buffers.sizes().gate_logits_f32,
            ctx.buffers.hc_streams(),
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            hidden,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            tokens,
            self.hidden_size as u32,
            self.hc.hc_mult as u32,
            self.hc.sinkhorn_iters as u32,
            ctx.config.rms_norm_eps as f32,
            self.hc.hc_eps,
            stream,
        )? {
            return Ok(());
        }
        ops::hc_pre(
            ctx.gpu,
            self.hc_pre_k,
            ctx.buffers.hc_streams(),
            site.hc_fn,
            site.hc_scale,
            site.hc_base,
            hidden,
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            tokens,
            self.hidden_size as u32,
            self.hc.hc_mult as u32,
            self.hc.sinkhorn_iters as u32,
            ctx.config.rms_norm_eps as f32,
            self.hc.hc_eps,
            stream,
        )
    }

    pub(super) fn hc_post(
        &self,
        block_out: DevicePtr,
        tokens: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::hc_post(
            ctx.gpu,
            self.hc_post_k,
            block_out,
            ctx.buffers.hc_streams(),
            ctx.buffers.hc_post(),
            ctx.buffers.hc_comb(),
            ctx.buffers.hc_streams(),
            tokens,
            self.hidden_size as u32,
            self.hc.hc_mult as u32,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::parse_fast_prefill;

    #[test]
    fn fast_prefill_is_explicitly_opt_in() {
        assert!(parse_fast_prefill(Some("1")));
        assert!(parse_fast_prefill(Some("true")));
        assert!(!parse_fast_prefill(None));
        assert!(!parse_fast_prefill(Some("0")));
    }
}
