// SPDX-License-Identifier: AGPL-3.0-only

//! Batched Mamba-2 verify: prefill GEMMs + n=1 conv FIR + fused persistent
//! scan. Conv stays sequential so FIR intermediates stay lossless. H dumps
//! go into the contiguous `h_state_intermediates` slab (slot*ni + t).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

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
        let pd_fp8_ok = self.fp8_gemm_t_k.0 != 0;
        self.prefill_in_proj(normed, proj, n, h, false, false, pd_fp8_ok, ctx, stream)?;

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

        let use_fused = self.mamba2_ssm_verify_k.0 != 0
            && std::env::var("ATLAS_NO_MAMBA_VERIFY_FUSED").is_err();
        if use_fused {
            ops::mamba2_ssm_verify(
                ctx.gpu,
                self.mamba2_ssm_verify_k,
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
        self.prefill_out_proj(gated_out, out, n, h, false, false, pd_fp8_ok, ctx, stream)?;
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
}
