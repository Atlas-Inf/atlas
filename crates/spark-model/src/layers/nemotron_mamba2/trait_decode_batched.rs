// SPDX-License-Identifier: AGPL-3.0-only

//! Batched Mamba-2 verify: prefill GEMMs + the proven n=1 conv/SSM loop.
//! Fused `mamba2_ssm_verify` is compiled but not dispatched until it is
//! CUDA-graph safe (ILA 700 on first kgamma capture).

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
        let fp8_a = false;
        let w4a4 = false;
        let pd_fp8_ok = self.fp8_gemm_t_k.0 != 0;
        self.prefill_in_proj(normed, proj, n, h, fp8_a, w4a4, pd_fp8_ok, ctx, stream)?;

        let xbc_tmp = ctx.buffers.ssm_deinterleaved();
        let y_out = ctx.buffers.attn_output();
        let h_bytes = ctx.config.ssm_h_state_bytes();
        let conv_bytes = ctx.config.ssm_conv_state_bytes();

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
            let dt = proj.offset(t * self.in_proj_size * bf16 + (self.d_inner + self.d_xbc) * bf16);
            self.ssm_decode(
                ctx.gpu,
                ssm_state.h_state,
                xbc_tmp,
                xbc_tmp.offset(self.d_inner * bf16),
                xbc_tmp.offset((self.d_inner + gs) * bf16),
                dt,
                y_out.offset(t * self.d_inner * bf16),
                1,
                stream,
            )?;
            if t + 1 < num_tokens {
                if t < ssm_state.h_state_intermediates.len() {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.h_state,
                        ssm_state.h_state_intermediates[t],
                        h_bytes,
                        stream,
                    )?;
                }
                if t < ssm_state.conv_state_intermediates.len() {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t],
                        conv_bytes,
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
        self.prefill_out_proj(gated_out, out, n, h, fp8_a, w4a4, pd_fp8_ok, ctx, stream)?;
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
