// SPDX-License-Identifier: AGPL-3.0-only

//! GLM arms of the HC prefill body (`prefill_inner_hc`): layer bounds, the
//! pre-site dispatch, the fused post/pre seam, the paged-MLA gate and the
//! final-layer contraction.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::{HcNorm, HcSeam, HcSiteWeights, HcWeights, Qwen3AttentionLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// `(is_first_layer, is_last_layer)` of the HC highway.
    pub(super) fn hc_prefill_layer_bounds(
        &self,
        hc: &HcWeights,
        ctx: &ForwardContext,
    ) -> (bool, bool) {
        // GLM carries its physical block index; upstream mixed models carry model indices.
        if ctx.config.model_type == "glm5_next" {
            (
                self.block_idx == 0,
                self.block_idx + 1 == ctx.config.num_hidden_layers,
            )
        } else {
            (hc.is_first_model_layer, hc.is_last_model_layer)
        }
    }

    /// One HC pre site of the prefill (GLM's own prefill mix, else the
    /// generic site). Returns whether `norm` (GLM only) was written too.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_pre_prefill_site(
        &self,
        site: &HcSiteWeights,
        hc: &HcWeights,
        hidden: DevicePtr,
        n: u32,
        norm: Option<HcNorm>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let hc_streams = ctx.buffers.hc_streams();
        let post = ctx.buffers.hc_post();
        let comb = ctx.buffers.hc_comb();
        if ctx.config.model_type == "glm5_next" {
            return self.hc_pre_prefill(site, hc, hidden, n, norm, ctx, stream);
        } else {
            ops::hc_pre_site(
                ctx.gpu,
                self.hc_pre_k,
                hc_streams,
                site,
                hc,
                hidden,
                post,
                comb,
                ctx.buffers.hc_lowrank_scratch(),
                n,
                h as u32,
                eps,
                stream,
            )?;
        }
        Ok(false)
    }

    /// `ATLAS_GLM_HC_PREFILL_FUSE2`: the GLM vanilla block input norm
    /// (`weight` into `out`) for a pre site's finalizer to write.
    pub(super) fn hc_prefill_norm(
        &self,
        weight: &crate::weight_map::DenseWeight,
        out: DevicePtr,
        ctx: &ForwardContext,
    ) -> Option<HcNorm> {
        (ctx.config.model_type == "glm5_next"
            && self.norm_vanilla
            && super::super::hc_prefill_fuse2_enabled())
        .then_some(HcNorm {
            weight: weight.weight,
            out,
        })
    }

    /// `qwen3_attention::hc_seam_overlapped` for this GLM layer's site
    /// `site` with block input norm `norm_weight`
    /// (`ATLAS_GLM_SP_GATHER_OVERLAP`; never under diagnostics, and a post
    /// only with no norm on the attention output).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_seam_overlapped_glm(
        &self,
        hc: &HcWeights,
        site: &HcSiteWeights,
        norm_weight: &crate::weight_map::DenseWeight,
        (post, gather): (bool, bool),
        hidden: DevicePtr,
        normed: DevicePtr,
        sp: Option<crate::layers::glm_sp::SpRows>,
        diag_this: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if ctx.config.model_type != "glm5_next"
            || diag_this
            || !ops::HcVariant::of(hc).applies_block_input_norm()
            || (post && self.post_attn_out_norm.is_some())
        {
            return Ok(false);
        }
        super::super::hc_seam_overlapped(
            hc,
            site,
            (post, gather),
            hidden,
            normed,
            sp,
            ctx,
            stream,
            |out| self.hc_prefill_norm(norm_weight, out, ctx),
            |input, out, rows| {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    input,
                    norm_weight,
                    out,
                    rows,
                    ctx.config.hidden_size as u32,
                    ctx.config.rms_norm_eps as f32,
                    stream,
                )
            },
        )
    }

    /// What ran of the attention site's post fused with the FFN site's
    /// pre-mix (and `norm`; GLM only, never under diagnostics).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn hc_post_pre_prefill_seam(
        &self,
        hc: &HcWeights,
        attn_out: DevicePtr,
        hidden: DevicePtr,
        n: u32,
        diag_this: bool,
        norm: Option<HcNorm>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<HcSeam> {
        let hc_mult = hc.hc_mult as u32;
        // GLM: this site's post fused with the FFN site's pre-mix.
        let glm = if ctx.config.model_type == "glm5_next" && !diag_this {
            super::super::hc_post_pre_prefill_fused(
                &hc.ffn,
                Some(attn_out),
                hidden,
                n,
                hc_mult,
                hc.sinkhorn_iters as u32,
                hc.hc_eps,
                norm,
                ctx,
                stream,
            )?
        } else {
            HcSeam::Skipped
        };
        if glm.ran() {
            return Ok(glm);
        }
        // qwen4_exp (ATLAS_QWEN4EXP_PREFILL_HC): the same seam on the low-rank
        // collapse, bit-identical to the separate post and pre.
        let seam = !diag_this
            && ops::qwen4exp_prefill_hc::hc_post_pre_seam(
                ctx.gpu,
                hc,
                &hc.ffn,
                attn_out,
                ctx.buffers.hc_streams(),
                hidden,
                ctx.buffers.hc_post(),
                ctx.buffers.hc_lowrank_scratch(),
                n,
                ctx.config.hidden_size as u32,
                ctx.config.rms_norm_eps as f32,
                stream,
            )?;
        Ok(if seam { HcSeam::Ran } else { HcSeam::Skipped })
    }

    /// Whether this prefill chunk takes the paged (GLM sparse) attention arm.
    pub(super) fn glm_paged_prefill(&self, ctx: &ForwardContext) -> bool {
        // GLM semantic-index MLA runs every chunk, the first included, through
        // the paged sparse path: the dense cache-skip arm attends to every
        // earlier row (not the top-k selection) and is quadratic in the chunk.
        ctx.attn_metadata.is_some_and(|m| !m.block_table.is_null())
            && self
                .mla
                .as_ref()
                .is_some_and(|mla| mla.glm_indexer.is_some())
    }

    /// GLM's last-layer highway contraction into `hidden`.
    pub(super) fn hc_contract_prefill(
        &self,
        hc_streams: DevicePtr,
        hidden: DevicePtr,
        n: u32,
        hc_mult: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        ops::hc_contract(
            ctx.gpu,
            self.hc_contract_k,
            hc_streams,
            hidden,
            n,
            h as u32,
            hc_mult,
            stream,
        )
    }
}
