// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-sequence K-row verify for Lightning Mamba-2.
//!
//! One in_proj / out_proj over R = Σ ks rows. Conv + SSM stay per-seq
//! (stateful). Same math as `decode_batched_verify` per sequence.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

impl NemotronMamba2Layer {
    pub(super) fn decode_verify_multi_loop<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            states.len() == n_seqs && ks.len() == n_seqs,
            "decode_verify_multi: states/ks/n mismatch"
        );
        if std::env::var("ATLAS_DFLASH_MAMBA_MULTI_LOOP").is_ok() {
            let h = ctx.config.hidden_size;
            let bf16 = 2usize;
            let mut off = 0usize;
            for i in 0..n_seqs {
                let k = ks[i];
                let row = off * h * bf16;
                self.decode_batched_verify(
                    hidden.offset(row),
                    residual.offset(row),
                    k,
                    states[i],
                    ctx,
                    stream,
                )?;
                off += k;
            }
            return Ok(());
        }
        self.decode_verify_multi_fused(hidden, residual, n_seqs, ks, states, ctx, stream)
    }

    fn decode_verify_multi_fused<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let r_total: usize = ks.iter().sum();
        let n = r_total as u32;
        let gs = self.n_groups * self.state_size;
        let row_bytes = self.d_xbc * bf16;

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

        let mut off = 0usize;
        for i in 0..n_seqs {
            let k = ks[i];
            let ssm_state = states[i]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
            for t in 0..k {
                let xbc_in = proj.offset(
                    (off + t) * self.in_proj_size * bf16 + self.d_inner * bf16,
                );
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
                ctx.gpu.copy_d2d_async(
                    xbc_tmp,
                    packed.offset((off + t) * row_bytes),
                    row_bytes,
                    stream,
                )?;
                if t + 1 < k && t < ssm_state.conv_state_intermediates.len() {
                    ctx.gpu.copy_d2d_async(
                        ssm_state.conv_state,
                        ssm_state.conv_state_intermediates[t],
                        conv_bytes,
                        stream,
                    )?;
                }
            }

            let x_ptr = packed.offset(off * row_bytes);
            let b_ptr = x_ptr.offset(self.d_inner * bf16);
            let c_ptr = x_ptr.offset((self.d_inner + gs) * bf16);
            let dt_ptr = proj.offset(
                off * self.in_proj_size * bf16 + (self.d_inner + self.d_xbc) * bf16,
            );
            let n_inter = ssm_state
                .h_state_intermediates
                .len()
                .min(k.saturating_sub(1));
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
                    y_out.offset(off * self.d_inner * bf16),
                    h_inter,
                    n_inter as u32,
                    inter_stride,
                    1,
                    k as u32,
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
                for t in 0..k {
                    let xt = packed.offset((off + t) * row_bytes);
                    self.ssm_decode(
                        ctx.gpu,
                        ssm_state.h_state,
                        xt,
                        xt.offset(self.d_inner * bf16),
                        xt.offset((self.d_inner + gs) * bf16),
                        proj.offset(
                            (off + t) * self.in_proj_size * bf16
                                + (self.d_inner + self.d_xbc) * bf16,
                        ),
                        y_out.offset((off + t) * self.d_inner * bf16),
                        1,
                        stream,
                    )?;
                    if t + 1 < k && t < ssm_state.h_state_intermediates.len() {
                        ctx.gpu.copy_d2d_async(
                            ssm_state.h_state,
                            ssm_state.h_state_intermediates[t],
                            h_bytes,
                            stream,
                        )?;
                    }
                }
            }
            off += k;
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
            (r_total * h) as u32,
            stream,
        )?;
        Ok(())
    }
}
