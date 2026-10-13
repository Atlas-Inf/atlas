// SPDX-License-Identifier: AGPL-3.0-only

//! Q/K/V projections of the KDA attention rows (fused K=5 verify GEMV,
//! verify projections, or the hot cast route shared across Q/K/V).

use super::*;

impl Glm5KdaLayer {
    /// Project `normed` into the Q, K and V planes of `projected`
    /// (`plane_bytes` apart).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_attention_qkv(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        plane_bytes: usize,
        m: u32,
        p: usize,
        h: u32,
        decode: bool,
        capture_verify_intermediates: bool,
        profile_timer: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let fused_qkv = capture_verify_intermediates
            && m == 5
            && self.w4a16_gemv_batch5_qkv_k.0 != 0
            && verify_fused_qkv_enabled()
            && !crate::layers::canonical_verify::enabled();
        // Whether Q took the Lt FP8 route, leaving `normed` cast in scratch.
        let mut q_fp8 = false;
        if fused_qkv {
            ops::w4a16_gemv_batch5_qkv(
                ctx.gpu,
                self.w4a16_gemv_batch5_qkv_k,
                ops::gemv_touch(ops::Twin::W4Batch5Qkv),
                normed,
                &self.weights.q_proj.nvfp4,
                &self.weights.k_proj.nvfp4,
                &self.weights.v_proj.nvfp4,
                projected,
                m,
                p as u32,
                h,
                stream,
            )?;
        } else if capture_verify_intermediates {
            self.project_hot_verify(
                normed,
                &self.weights.q_proj,
                projected,
                m,
                p as u32,
                h,
                ctx,
                stream,
            )?;
        } else {
            q_fp8 = self.project_hot_cast(
                normed,
                &self.weights.q_proj,
                projected,
                m,
                p as u32,
                h,
                decode,
                true,
                ctx,
                stream,
            )?;
        }
        profile::step(ctx, stream, profile_timer, "q_proj")?;
        if !fused_qkv {
            if capture_verify_intermediates {
                self.project_hot_verify(
                    normed,
                    &self.weights.k_proj,
                    projected.offset(plane_bytes),
                    m,
                    p as u32,
                    h,
                    ctx,
                    stream,
                )?;
                self.project_hot_verify(
                    normed,
                    &self.weights.v_proj,
                    projected.offset(2 * plane_bytes),
                    m,
                    p as u32,
                    h,
                    ctx,
                    stream,
                )?;
            } else {
                // K and V reuse Q's E4M3 cast of `normed` on the Lt FP8 route.
                let k_fp8 = self.project_hot_cast(
                    normed,
                    &self.weights.k_proj,
                    projected.offset(plane_bytes),
                    m,
                    p as u32,
                    h,
                    decode,
                    !q_fp8,
                    ctx,
                    stream,
                )?;
                self.project_hot_cast(
                    normed,
                    &self.weights.v_proj,
                    projected.offset(2 * plane_bytes),
                    m,
                    p as u32,
                    h,
                    decode,
                    !k_fp8,
                    ctx,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
