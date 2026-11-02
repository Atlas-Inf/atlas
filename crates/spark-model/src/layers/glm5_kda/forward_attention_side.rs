// SPDX-License-Identifier: AGPL-3.0-only

//! The KDA side projections off the attention input: beta / f_a / g_a, then
//! the f_b / g_b gates. Split out of `forward_attention.rs` (500-LoC cap).

use super::*;

impl Glm5KdaLayer {
    /// beta, f_a and g_a of `normed` into the planes after q/k/v in
    /// `projected`, then g1 = f_b(f_a) and g2 = g_b(g_a) in
    /// `ssm_deinterleaved`; returns `(beta, g1, g2)`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_side_projections(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        plane_bytes: usize,
        m: u32,
        decode: bool,
        verify_proj: bool,
        capture_verify_intermediates: bool,
        profile_timer: &mut Option<std::time::Instant>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr, DevicePtr)> {
        let (tokens, h, p, bf16) = (
            m as usize,
            self.hidden_size as u32,
            self.heads * self.dim,
            2,
        );
        let canonical = crate::layers::canonical_verify::enabled();
        let beta = projected.offset(3 * plane_bytes);
        let fa = beta.offset(tokens * self.heads * bf16);
        let ga = fa.offset(tokens * self.dim * bf16);
        let fused_dense_pairs = capture_verify_intermediates
            && !canonical
            && m == 5
            && self.dense_gemv_batch5_dual_k.0 != 0
            && verify_fused_dense_pairs_enabled();
        let fused_dense_triple = capture_verify_intermediates
            && !canonical
            && m == 5
            && self.dense_gemv_batch5_triple_n_k.0 != 0
            && verify_fused_dense_triple_enabled();
        // Verify rows 2..=8 otherwise: the same batchm body per plane, one grid.
        let batchm_fused = capture_verify_intermediates
            && !canonical
            && (2..=ops::DENSE_GEMV_BATCHM_MAX_M).contains(&m)
            && self.dense_gemv_batchm_triple_n_k.0 != 0
            && self.dense_gemv_batchm_dual_k.0 != 0;
        if canonical && verify_proj {
            // One canonical launch for the three side projections.
            let (heads, dim) = (self.heads as u32, self.dim as u32);
            crate::layers::canonical_verify::dense_planes(
                ctx.gpu,
                &[
                    (normed, &self.weights.b_proj, beta, heads, h),
                    (normed, &self.weights.f_a_proj, fa, dim, h),
                    (normed, &self.weights.g_a_proj, ga, dim, h),
                ],
                m,
                stream,
            )?;
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else if batchm_fused && !fused_dense_triple {
            ops::dense_gemv_batchm_triple_n(
                ctx.gpu,
                self.dense_gemv_batchm_triple_n_k,
                normed,
                [
                    &self.weights.b_proj,
                    &self.weights.f_a_proj,
                    &self.weights.g_a_proj,
                ],
                [beta, fa, ga],
                m,
                [self.heads as u32, self.dim as u32],
                h,
                stream,
            )?;
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else if fused_dense_triple {
            ops::dense_gemv_batch5_triple_n(
                ctx.gpu,
                self.dense_gemv_batch5_triple_n_k,
                normed,
                &self.weights.b_proj,
                &self.weights.f_a_proj,
                &self.weights.g_a_proj,
                beta,
                fa,
                ga,
                self.heads as u32,
                self.dim as u32,
                h,
                stream,
            )?;
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else if !verify_proj
            && m > 1
            && self.dense_gemm_pipelined_k.0 != 0
            && self.dense_gemm_pipelined_triple_n_k.0 != 0
        {
            // The three pipelined launches `project_dense` would make, in one
            // grid, or their bytes from one GEMM over the packed weights.
            if decode
                || ctx.graph_capture
                || !self.side_proj_packed(normed, [beta, fa, ga], m, ctx, stream)?
            {
                ops::dense_gemm_pipelined_triple_n(
                    ctx.gpu,
                    self.dense_gemm_pipelined_triple_n_k,
                    normed,
                    [
                        &self.weights.b_proj,
                        &self.weights.f_a_proj,
                        &self.weights.g_a_proj,
                    ],
                    [beta, fa, ga],
                    m,
                    [self.heads as u32, self.dim as u32],
                    h,
                    stream,
                )?;
            }
            profile::step(ctx, stream, profile_timer, "beta_f_a_g_a")?;
        } else {
            if verify_proj {
                self.project_dense_verify(
                    normed,
                    &self.weights.b_proj,
                    beta,
                    m,
                    self.heads as u32,
                    h,
                    ctx,
                    stream,
                )?;
            } else {
                self.project_dense(
                    normed,
                    &self.weights.b_proj,
                    beta,
                    m,
                    self.heads as u32,
                    h,
                    ctx,
                    stream,
                )?;
            }
            profile::step(ctx, stream, profile_timer, "kv_beta_proj")?;
            if fused_dense_pairs {
                ops::dense_gemv_batch5_dual(
                    ctx.gpu,
                    self.dense_gemv_batch5_dual_k,
                    normed,
                    normed,
                    &self.weights.f_a_proj,
                    &self.weights.g_a_proj,
                    fa,
                    ga,
                    self.dim as u32,
                    h,
                    stream,
                )?;
            } else {
                if verify_proj {
                    self.project_dense_verify(
                        normed,
                        &self.weights.f_a_proj,
                        fa,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                } else {
                    self.project_dense(
                        normed,
                        &self.weights.f_a_proj,
                        fa,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                }
                profile::step(ctx, stream, profile_timer, "f_a_proj")?;
                if verify_proj {
                    self.project_dense_verify(
                        normed,
                        &self.weights.g_a_proj,
                        ga,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                } else {
                    self.project_dense(
                        normed,
                        &self.weights.g_a_proj,
                        ga,
                        m,
                        self.dim as u32,
                        h,
                        ctx,
                        stream,
                    )?;
                }
            }
            if fused_dense_pairs {
                profile::step(ctx, stream, profile_timer, "f_a_g_a")?;
            }
        }

        let g1 = ctx.buffers.ssm_deinterleaved();
        let g2 = g1.offset(plane_bytes);
        if canonical && verify_proj {
            let dim = self.dim as u32;
            crate::layers::canonical_verify::dense_planes(
                ctx.gpu,
                &[
                    (fa, &self.weights.f_b_proj, g1, p as u32, dim),
                    (ga, &self.weights.g_b_proj, g2, p as u32, dim),
                ],
                m,
                stream,
            )?;
        } else if batchm_fused && !fused_dense_pairs {
            ops::dense_gemv_batchm_dual(
                ctx.gpu,
                self.dense_gemv_batchm_dual_k,
                [fa, ga],
                [&self.weights.f_b_proj, &self.weights.g_b_proj],
                [g1, g2],
                m,
                p as u32,
                self.dim as u32,
                stream,
            )?;
        } else if fused_dense_pairs {
            ops::dense_gemv_batch5_dual(
                ctx.gpu,
                self.dense_gemv_batch5_dual_k,
                fa,
                ga,
                &self.weights.f_b_proj,
                &self.weights.g_b_proj,
                g1,
                g2,
                p as u32,
                self.dim as u32,
                stream,
            )?;
        } else if verify_proj {
            self.project_dense_verify(
                fa,
                &self.weights.f_b_proj,
                g1,
                m,
                p as u32,
                self.dim as u32,
                ctx,
                stream,
            )?;
            self.project_dense_verify(
                ga,
                &self.weights.g_b_proj,
                g2,
                m,
                p as u32,
                self.dim as u32,
                ctx,
                stream,
            )?;
        } else {
            self.project_dense(
                fa,
                &self.weights.f_b_proj,
                g1,
                m,
                p as u32,
                self.dim as u32,
                ctx,
                stream,
            )?;
            self.project_dense(
                ga,
                &self.weights.g_b_proj,
                g2,
                m,
                p as u32,
                self.dim as u32,
                ctx,
                stream,
            )?;
        }
        profile::step(ctx, stream, profile_timer, "g_a_f_b_g_b")?;
        Ok((beta, g1, g2))
    }

    /// `ATLAS_GLM_KDA_SIDE_PROJ_LT`: b / f_a / g_a as one CUTLASS GEMM over
    /// the weights the loader stored back to back, staged in
    /// `ssm_deinterleaved` (g1 / g2 are written only after). `false`: off,
    /// not packed, or CUTLASS declined; nothing launched.
    fn side_proj_packed(
        &self,
        normed: DevicePtr,
        outputs: [DevicePtr; 3],
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let w = &self.weights;
        let row = self.hidden_size * 2;
        if !crate::layers::kda_side_proj_lt_enabled() {
            return Ok(false);
        }
        static REPORTED: std::sync::Once = std::sync::Once::new();
        let report = |what: &str| {
            REPORTED.call_once(|| tracing::info!("ATLAS_GLM_KDA_SIDE_PROJ_LT: {what} ({m} rows)"))
        };
        if w.f_a_proj.weight != w.b_proj.weight.offset(self.heads * row)
            || w.g_a_proj.weight != w.f_a_proj.weight.offset(self.dim * row)
        {
            report("b / f_a / g_a are not packed; the pipelined triple runs");
            return Ok(false);
        }
        let scratch = ctx.buffers.ssm_deinterleaved();
        ensure!(
            m as usize * (self.heads + 2 * self.dim) * 2 <= ctx.buffers.sizes().ssm_deinterleaved,
            "GLM KDA packed side projections: {m} rows overflow the staging scratch"
        );
        let ran = ops::dense_gemm_packed_triple_n(
            ctx.gpu,
            normed,
            w.b_proj.weight,
            scratch,
            outputs,
            m,
            [self.heads as u32, self.dim as u32],
            self.hidden_size as u32,
            stream,
        )?;
        report(if ran {
            "packed CUTLASS GEMM"
        } else {
            "CUTLASS declined the packed GEMM; the pipelined triple runs"
        });
        Ok(ran)
    }
}
