// SPDX-License-Identifier: AGPL-3.0-only

//! `hc_pre_site` for the multi-sequence path, with the split `hc_pre` fast
//! path for short Sinkhorn batches.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// `hc_pre_site` for the multi-sequence path, taking the exact split
    /// `hc_pre` for short Sinkhorn batches (see `ops::try_hc_pre_split`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn ms_hc_pre_site(
        &self,
        site: &crate::layers::qwen3_attention::HcSiteWeights,
        hc: &crate::layers::qwen3_attention::HcWeights,
        streams: DevicePtr,
        y_out: DevicePtr,
        post: DevicePtr,
        comb: DevicePtr,
        n: usize,
        eps: f32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        if site.lowrank.is_none()
            && ops::try_hc_pre_split(
                ctx.gpu,
                self.hc_pre_k,
                self.hc_pre_mix_k,
                self.hc_pre_from_raw_mix_k,
                ctx.buffers.gate_logits_f32(),
                ctx.buffers.sizes().gate_logits_f32,
                streams,
                site.hc_fn,
                site.hc_scale,
                site.hc_base,
                y_out,
                post,
                comb,
                n as u32,
                h,
                hc.hc_mult as u32,
                hc.sinkhorn_iters as u32,
                eps,
                hc.hc_eps,
                stream,
            )?
        {
            return Ok(());
        }
        ops::hc_pre_site(
            ctx.gpu,
            self.hc_pre_k,
            streams,
            site,
            hc,
            y_out,
            post,
            comb,
            ctx.buffers.hc_lowrank_scratch(),
            n as u32,
            h,
            eps,
            stream,
        )
    }
}
