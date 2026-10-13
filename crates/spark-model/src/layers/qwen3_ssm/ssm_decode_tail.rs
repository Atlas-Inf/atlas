// SPDX-License-Identifier: AGPL-3.0-only

//! The single-token GDN mixer's tail: the exact fused decode step
//! (`ATLAS_QWEN4EXP_DECODE_FUSE`, GDN group) and the out projection both arms
//! end in. Split from `ssm_forward.rs` for the 500-LoC cap.

use super::*;

impl Qwen3SsmLayer {
    /// Steps 3-7 of [`Self::ssm_forward`] (BA gates, conv + L2 norm,
    /// recurrence, sigmoid-gated RMS norm) as one launch of
    /// `qwen4exp_gdn_decode_fused`, writing the gated-norm output to
    /// `ssm_qkvz` and the same recurrence state, conv window, gates and betas
    /// as the four kernels. Only on the arm those four kernels form (FP32 conv
    /// output, FP32 recurrence state, FP32 GDN output into the sigmoid gated
    /// norm). Returns whether it ran; on `false` nothing was launched.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn gdn_decode_fused(
        &self,
        normed: DevicePtr,
        state: &SsmLayerState,
        qkvz: DevicePtr,
        gates: DevicePtr,
        beta: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !self.four_kernel_f32_arm(ctx) {
            return Ok(false);
        }
        let c = ctx.config;
        ops::qwen4exp_decode_fuse::gdn_decode(
            ctx.gpu,
            &ops::qwen4exp_decode_fuse::GdnDecode {
                h_state: state.h_state,
                conv_state: state.conv_state,
                qkvz,
                conv_w: self.ssm.conv1d.weight,
                ba_in: normed,
                ba_w: self.ssm.in_proj_ba.weight,
                a_log: self.ssm.a_log.weight,
                dt_bias: self.ssm.dt_bias.weight,
                gate_out: gates,
                beta_out: beta,
                norm_w: self.ssm.norm.weight,
                out: ctx.buffers.ssm_qkvz(),
            },
            c.linear_num_key_heads as u32,
            c.linear_num_value_heads as u32,
            c.linear_key_head_dim as u32,
            c.linear_value_head_dim as u32,
            c.linear_conv_kernel_dim as u32,
            c.hidden_size as u32,
            1e-6,
            c.rms_norm_eps as f32,
            stream,
        )
    }

    /// Whether a decode token's steps 3-7 are the four kernels the fused step
    /// reproduces: FP32 conv output, FP32 recurrence state and output, the
    /// sigmoid gated norm, no fused GDN+norm kernel.
    pub(super) fn four_kernel_f32_arm(&self, ctx: &ForwardContext) -> bool {
        let fused_norm = self.gdn_f32_norm_k.0 != 0 && super::gdn_fused_norm_enabled();
        self.conv1d_l2norm_f32_k.0 != 0
            && self.gdn_f32_k.0 != 0
            && self.gated_rms_norm_f32_k.0 != 0
            && !fused_norm
            && !super::ssm_h_fp16_enabled()
            && ctx.config.output_gate_type == "sigmoid"
    }

    /// Step 8 of [`Self::ssm_forward`]: the out projection of the gated-norm
    /// output and the TP all-reduce.
    pub(super) fn ssm_out_proj(
        &self,
        normed_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
        trace: bool,
        debug: bool,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let value_dim = ctx.config.linear_num_value_heads * ctx.config.linear_value_head_dim;
        // ── 8. Output projection: [value_dim → hidden_size] ──
        let out = ctx.buffers.moe_output();
        if let Some(ref fp8) = self.out_proj_fp8w {
            // `w8a16_gemv` consumes `[N/BS,K/BS] BF16` block scales —
            // the canonical Qwen FP8 release format.
            fp8.scale_format.expect(
                crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
                "ssm_forward::out_proj_fp8w → w8a16_gemv",
            );
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed_out,
                fp8.weight,
                fp8.row_scale,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        } else if let Some(ref dense_out) = self.out_proj_dense {
            if ctx.levers.gdn_fp8_decode
                && self.dense_gemv_fp8w_k.0 != 0
                && let Some(ref fp8w) = self.out_proj_fp8w_rowwise
                && fp8w.scale_format == crate::weight_map::WeightQuantFormat::Fp8PerRow
            {
                // Per-row-FP8 decode copy (ATLAS_GDN_FP8_DECODE): half the
                // BF16 weight bytes for the serial-decode GEMV.
                ops::dense_gemv_fp8w(
                    ctx.gpu,
                    self.dense_gemv_fp8w_k,
                    normed_out,
                    &crate::weight_map::Fp8DenseWeight {
                        weight: fp8w.weight,
                        row_scale: fp8w.row_scale,
                    },
                    out,
                    h,
                    value_dim as u32,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed_out,
                    dense_out,
                    out,
                    h,
                    value_dim as u32,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                normed_out,
                &self.ssm.out_proj,
                out,
                h,
                value_dim as u32,
                stream,
            )?;
        }
        if trace {
            ctx.gpu.synchronize(stream).inspect_err(|_e| {
                tracing::error!("CRASH at out_proj");
            })?;
        }
        if debug {
            ctx.gpu.synchronize(stream)?;
            Self::debug_bf16(ctx.gpu, "out-proj", out, 4);
        }

        // GDN HeadParallel: `out` is this rank's PARTIAL row-parallel out_proj
        // over its local value heads. Reduce across TP ranks to the complete
        // SSM output before the caller's residual add. Single-token path
        // (dense_gemv / w8a16_gemv / w4a16_gemv above → one position), so
        // num_tokens = 1. No-op at tp=1. Covers single-token decode
        // (trait_decode) and per-sequence multi-seq decode (trait_decode_multi_seq).
        self.ssm_tp_all_reduce(out, 1, ctx, stream)?;

        Ok(out)
    }
}
