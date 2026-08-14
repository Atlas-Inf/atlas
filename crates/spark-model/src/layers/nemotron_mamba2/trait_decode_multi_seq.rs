// SPDX-License-Identifier: AGPL-3.0-only
//! AR multi-seq Mamba-2: batched in/out proj, per-seq conv+scan.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

impl NemotronMamba2Layer {
    #[allow(dead_code)]
    pub(super) fn decode_multi_seq_ar(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &mut [&mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let n = num_seqs as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let gs = self.n_groups * self.state_size;
        let group_size = (self.d_inner / self.n_groups) as u32;
        let pd_fp8_ok = self.fp8_gemm_t_k.0 != 0;

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
        self.prefill_in_proj(normed, proj, n, h, false, false, pd_fp8_ok, ctx, stream)?;

        let xbc_tmp = ctx.buffers.ssm_deinterleaved();
        let y_out = ctx.buffers.attn_output();
        let row_proj = self.in_proj_size * bf16;
        let row_xbc = self.d_xbc * bf16;
        let row_y = self.d_inner * bf16;

        for i in 0..num_seqs {
            let ssm = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
            let proj_i = proj.offset(i * row_proj);
            let xbc_in = proj_i.offset(self.d_inner * bf16);
            let dt_ptr = proj_i.offset((self.d_inner + self.d_xbc) * bf16);
            let xbc_i = xbc_tmp.offset(i * row_xbc);
            let y_i = y_out.offset(i * row_y);
            self.conv1d_update_biased(
                ctx.gpu,
                ssm.conv_state,
                xbc_in,
                xbc_i,
                self.d_xbc as u32,
                self.d_conv as u32,
                1,
                stream,
            )?;
            let x_ptr = xbc_i;
            let b_ptr = xbc_i.offset(self.d_inner * bf16);
            let c_ptr = xbc_i.offset((self.d_inner + gs) * bf16);
            self.ssm_decode(
                ctx.gpu,
                ssm.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                dt_ptr,
                y_i,
                1,
                stream,
            )?;
        }

        let gated = ctx.buffers.norm_output();
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_out,
            proj,
            &self.ssm.ssm_norm,
            gated,
            n,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        let out = ctx.buffers.qkv_output();
        self.prefill_out_proj(gated, out, n, h, false, false, pd_fp8_ok, ctx, stream)?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            out,
            n * h as u32,
            stream,
        )?;
        Ok(())
    }
}
