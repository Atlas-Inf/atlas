// SPDX-License-Identifier: AGPL-3.0-only

//! Batched Mamba-2 verify: prefill GEMMs + n=1 conv FIR + fused persistent
//! scan. Conv stays sequential so FIR intermediates stay lossless. H dumps
//! go into the contiguous `h_state_intermediates` slab (slot*ni + t).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

fn use_fused_mamba_verify(
    kernel_available: bool,
    lightning_exact: bool,
    fused_not_disabled: bool,
) -> bool {
    kernel_available && !lightning_exact && fused_not_disabled
}

impl NemotronMamba2Layer {
    pub(super) fn decode_batched_verify(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let n = num_tokens as u32;
        let bf16 = 2usize;
        let gs = self.n_groups * self.state_size;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            n,
            h as u32,
            eps,
            stream,
        )?;

        let proj = ctx.buffers.ssm_qkvz();
        // Exact verify: decode-identical batchm twins; otherwise the (faster,
        // non-exact) prefill GEMMs the fused path tolerates.
        if ctx.levers.mamba_verify_exact {
            self.verify_exact_in_proj(normed, proj, n, ctx, stream)?;
        } else {
            let pd_fp8_ok = self.fp8_gemm_t_k.0 != 0;
            self.prefill_in_proj(normed, proj, n, h, false, false, pd_fp8_ok, ctx, stream)?;
        }

        let xbc_tmp = ctx.buffers.ssm_deinterleaved();
        let packed = ctx.buffers.qkv_output();
        let y_out = ctx.buffers.attn_output();
        let h_bytes = ctx.config.ssm_h_state_bytes();
        let conv_bytes = ctx.config.ssm_conv_state_bytes();
        let row_bytes = self.d_xbc * bf16;

        for t in 0..num_tokens {
            let xbc_in = proj.offset(t * self.in_proj_size * bf16 + self.d_inner * bf16);
            self.conv1d_update_biased(
                ctx.gpu,
                ssm_state.conv_state,
                xbc_in,
                xbc_tmp,
                self.d_xbc as u32,
                self.d_conv as u32,
                1,
                stream,
            )?;
            ctx.gpu
                .copy_d2d_async(xbc_tmp, packed.offset(t * row_bytes), row_bytes, stream)?;
            if t + 1 < num_tokens && t < ssm_state.conv_state_intermediates.len() {
                ctx.gpu.copy_d2d_async(
                    ssm_state.conv_state,
                    ssm_state.conv_state_intermediates[t],
                    conv_bytes,
                    stream,
                )?;
            }
        }

        let x_ptr = packed;
        let b_ptr = packed.offset(self.d_inner * bf16);
        let c_ptr = packed.offset((self.d_inner + gs) * bf16);
        let dt_ptr = proj.offset((self.d_inner + self.d_xbc) * bf16);
        let n_inter = ssm_state
            .h_state_intermediates
            .len()
            .min(num_tokens.saturating_sub(1));
        let (h_inter, inter_stride) = if n_inter > 0 {
            (ssm_state.h_state_intermediates[0], (h_bytes / 4) as u32)
        } else {
            (DevicePtr::NULL, 0u32)
        };

        let use_exact_persistent = self.mamba2_ssm_verify_exact_k.0 != 0
            && (ctx.levers.mamba_verify_exact
                || (ctx.levers.lightning_mamba_exact_recurrence
                    && std::env::var("ATLAS_LIGHTNING_MAMBA_EXACT_PERSISTENT").as_deref()
                        == Ok("1")));
        let use_fused = use_fused_mamba_verify(
            self.mamba2_ssm_verify_k.0 != 0,
            ctx.levers.lightning_mamba_exact_recurrence,
            ctx.levers.mamba_verify_fused,
        );
        if std::env::var("ATLAS_DFLASH_LAYER_TIMING").ok().as_deref() == Some("1") {
            tracing::info!(
                "mamba2_ssm_verify handle={} fused={}",
                self.mamba2_ssm_verify_k.0,
                use_fused
            );
        }
        if use_exact_persistent || use_fused {
            ops::mamba2_ssm_verify(
                ctx.gpu,
                if use_exact_persistent {
                    self.mamba2_ssm_verify_exact_k
                } else {
                    self.mamba2_ssm_verify_k
                },
                ssm_state.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                dt_ptr,
                self.ssm.a_log.weight,
                self.ssm.d_param.weight,
                self.ssm.dt_bias.weight,
                y_out,
                h_inter,
                n_inter as u32,
                inter_stride,
                1,
                n,
                self.num_heads as u32,
                self.head_dim as u32,
                self.state_size as u32,
                self.n_groups as u32,
                1e-9,
                1e9,
                self.d_xbc as u32,
                self.d_xbc as u32,
                self.in_proj_size as u32,
                self.d_inner as u32,
                use_exact_persistent,
                stream,
            )?;
        } else {
            for t in 0..num_tokens {
                let xt = packed.offset(t * row_bytes);
                self.ssm_decode(
                    ctx.gpu,
                    ssm_state.h_state,
                    xt,
                    xt.offset(self.d_inner * bf16),
                    xt.offset((self.d_inner + gs) * bf16),
                    proj.offset(t * self.in_proj_size * bf16 + (self.d_inner + self.d_xbc) * bf16),
                    y_out.offset(t * self.d_inner * bf16),
                    1,
                    stream,
                )?;
                if t + 1 < num_tokens && t < ssm_state.h_state_intermediates.len() {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        ssm_state.h_state_intermediates[t],
                        h_bytes,
                        stream,
                    )?;
                }
            }
        }

        let gated_out = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_out,
            proj,
            &self.ssm.ssm_norm,
            gated_out,
            n,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        let out = ctx.buffers.qkv_output();
        if ctx.levers.mamba_verify_exact {
            self.verify_exact_out_proj(gated_out, out, n, ctx, stream)?;
        } else {
            let pd_fp8_ok = self.fp8_gemm_t_k.0 != 0;
            self.prefill_out_proj(gated_out, out, n, h, false, false, pd_fp8_ok, ctx, stream)?;
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            out,
            (num_tokens * h) as u32,
            stream,
        )?;
        Ok(())
    }

    /// Every kernel the exact batched verify needs resolved: the persistent
    /// exact scan, plus a bit-identical batch-M twin for whatever dtype the
    /// in/out projections are stored in (FP8 on omni, BF16 or NVFP4 on
    /// variants — an unresolved twin or an NVFP4-only layer falls back to
    /// the per-row decode loop, which is already exact).
    pub(super) fn mamba_verify_exact_ready(&self) -> bool {
        if self.mamba2_ssm_verify_exact_k.0 == 0 {
            return false;
        }
        let in_ok = if self.in_proj_bf16.is_some() {
            self.dense_gemv_bf16_batchm_k.0 != 0
        } else if self.in_proj_fp8.is_some() {
            self.w8a16_gemv_batch16_k.0 != 0
        } else {
            self.w4a16_gemv_batch16_k.0 != 0
        };
        let out_ok = if self.out_proj_bf16.is_some() {
            self.dense_gemv_bf16_batchm_k.0 != 0
        } else if self.out_proj_fp8.is_some() {
            self.w8a16_gemv_batch16_k.0 != 0
        } else {
            self.w4a16_gemv_batch16_k.0 != 0
        };
        in_ok && out_ok
    }

    /// in_proj for the exact verify: `normed[M,H]` -> `proj[M,in_proj_size]`
    /// through the bit-identical batch-M twin of whichever GEMV decode uses.
    fn verify_exact_in_proj(
        &self,
        normed: DevicePtr,
        proj: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        if let Some(ref w) = self.in_proj_bf16 {
            ops::dense_gemv_batchm(
                ctx.gpu,
                self.dense_gemv_bf16_batchm_k,
                normed,
                w,
                proj,
                m,
                self.in_proj_size as u32,
                h,
                self.in_proj_size as u32,
                stream,
            )
        } else if let Some(ref fp8w) = self.in_proj_fp8 {
            ops::w8a16_gemv_batch4_div(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj,
                m,
                self.in_proj_size as u32,
                h,
                4,
                stream,
            )
        } else {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_gemv_batch16_k,
                normed,
                &self.ssm.in_proj,
                proj,
                m,
                self.in_proj_size as u32,
                h,
                stream,
            )
        }
    }

    /// out_proj for the exact verify: `gated[M,d_inner]` -> `out[M,H]`.
    fn verify_exact_out_proj(
        &self,
        gated: DevicePtr,
        out: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        if let Some(ref w) = self.out_proj_bf16 {
            ops::dense_gemv_batchm(
                ctx.gpu,
                self.dense_gemv_bf16_batchm_k,
                gated,
                w,
                out,
                m,
                h,
                self.d_inner as u32,
                h,
                stream,
            )
        } else if let Some(ref fp8w) = self.out_proj_fp8 {
            ops::w8a16_gemv_batch4_div(
                ctx.gpu,
                self.w8a16_gemv_batch16_k,
                gated,
                fp8w.weight,
                fp8w.row_scale,
                out,
                m,
                h,
                self.d_inner as u32,
                4,
                stream,
            )
        } else {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                self.w4a16_gemv_batch16_k,
                gated,
                &self.ssm.out_proj,
                out,
                m,
                h,
                self.d_inner as u32,
                stream,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::use_fused_mamba_verify;

    #[test]
    fn lightning_product_refuses_fused_recurrence() {
        assert!(!use_fused_mamba_verify(true, true, true));
        assert!(use_fused_mamba_verify(true, false, true));
        assert!(!use_fused_mamba_verify(false, false, true));
        assert!(!use_fused_mamba_verify(true, false, false));
    }
}
