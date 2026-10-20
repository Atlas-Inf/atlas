// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_SP_GATHER_OVERLAP`: a KDA layer's sequence-parallel seams with
//! their exchanges overlapping their compute (`layers::glm_sp_overlap`).

use super::*;
use crate::layers::glm_sp::SpRows;
use crate::layers::qwen3_attention::{
    HcNorm, HcSiteWeights, hc_prefill_fuse2_enabled, hc_seam_overlapped,
};

impl Glm5KdaLayer {
    /// `qwen3_attention::hc_seam_overlapped` for this layer's site `site`
    /// with block input norm `norm_weight` (fused into the finalizer under
    /// `ATLAS_GLM_HC_PREFILL_FUSE2`, else this layer's `rms_norm`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn sp_seam_overlapped(
        &self,
        site: &HcSiteWeights,
        norm_weight: &DenseWeight,
        post_gather: (bool, bool),
        hidden: DevicePtr,
        normed: DevicePtr,
        sp: Option<SpRows>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let fuse = hc_prefill_fuse2_enabled();
        hc_seam_overlapped(
            &self.hc,
            site,
            post_gather,
            hidden,
            normed,
            sp,
            ctx,
            stream,
            |out| {
                fuse.then_some(HcNorm {
                    weight: norm_weight.weight,
                    out,
                })
            },
            |input, out, rows| {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_k,
                    input,
                    norm_weight,
                    out,
                    rows,
                    self.hidden_size as u32,
                    ctx.config.rms_norm_eps as f32,
                    stream,
                )
            },
        )
    }
}
