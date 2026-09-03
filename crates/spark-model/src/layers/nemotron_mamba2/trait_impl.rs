// SPDX-License-Identifier: AGPL-3.0-only

//! `impl TransformerLayer for NemotronMamba2Layer`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::{nemotron_decode_policy, ops};

fn use_batched_mamba_verify(
    num_tokens: usize,
    lightning_exact: bool,
    fused_not_disabled: bool,
) -> bool {
    num_tokens > 1 && !lightning_exact && fused_not_disabled
}

impl TransformerLayer for NemotronMamba2Layer {
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        // 1. RMS norm + save residual
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            1,
            h as u32,
            eps,
            stream,
        )?;

        // 2. in_proj GEMV: normed[hidden_size] -> proj[in_proj_size]
        //    Layout: [z(d_inner) | xBC(d_xbc) | dt(num_heads)]
        let proj = ctx.buffers.ssm_qkvz();
        // Use FP8 GEMV if available (skips double-quantization lossy path)
        if let Some(ref w) = self.in_proj_bf16 {
            // Native BF16: `ssm.in_proj` is NULL here (never quantized).
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                normed,
                w,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else if let Some(ref fp8w) = self.in_proj_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                normed,
                &self.ssm.in_proj,
                proj,
                self.in_proj_size as u32,
                h as u32,
                stream,
            )?;
        }

        // Pointers into projection output (BF16, 2 bytes per element)
        let z_ptr = proj;
        let xbc_ptr = proj.offset(self.d_inner * 2);
        let dt_ptr = proj.offset((self.d_inner + self.d_xbc) * 2);

        // 3. Conv1d update on xBC (with bias, fused SiLU)
        let xbc_out = ctx.buffers.ssm_deinterleaved();
        self.conv1d_update_biased(
            ctx.gpu,
            ssm_state.conv_state,
            xbc_ptr,
            xbc_out,
            self.d_xbc as u32,
            self.d_conv as u32,
            1,
            stream,
        )?;

        // 4. Split xBC_out into x, B, C (BF16 offsets)
        let x_ptr = xbc_out;
        let gs = self.n_groups * self.state_size;
        let b_ptr = xbc_out.offset(self.d_inner * 2);
        let c_ptr = xbc_out.offset((self.d_inner + gs) * 2);

        // 5. SSM decode: state update + y output
        let y_ptr = ctx.buffers.attn_output();
        self.ssm_decode(
            ctx.gpu,
            ssm_state.h_state,
            x_ptr,
            b_ptr,
            c_ptr,
            dt_ptr,
            y_ptr,
            1,
            stream,
        )?;

        // 6. Gated RMS norm: rms_norm(y, ssm_norm) * silu(z)
        //    y is [d_inner], z is [d_inner], gate_stride = in_proj_size (z at start of proj)
        let gated_out = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_ptr,
            z_ptr,
            &self.ssm.ssm_norm,
            gated_out,
            1,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        // 7. out_proj GEMV: gated_out[d_inner] -> out[hidden_size]
        // Use qkv_output (NOT ssm_qkvz) — ssm_qkvz still holds z_ptr being read
        // by gated_rms_norm above. Writing out_proj to the same buffer creates a
        // write-after-read race that corrupts the gate signal → all-zero output.
        let out = ctx.buffers.qkv_output();
        if let Some(ref w) = self.out_proj_bf16 {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gated_out,
                w,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if let Some(ref fp8w) = self.out_proj_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                gated_out,
                fp8w.weight,
                fp8w.row_scale,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else {
            ops::w4a16_decode_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                self.w4a16_gemv_sw_k,
                ctx.levers.gemv_sw,
                gated_out,
                &self.ssm.out_proj,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        }

        // 8. Residual add: hidden += out_proj_result (hidden unchanged by rms_norm_residual)
        ops::residual_add(ctx.gpu, self.residual_add_k, hidden, out, h as u32, stream)?;

        Ok(())
    }

    fn decode_batched(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if use_batched_mamba_verify(
            num_tokens,
            ctx.levers.lightning_mamba_exact_recurrence,
            std::env::var("ATLAS_NO_MAMBA_VERIFY_FUSED").is_err(),
        ) {
            return self.decode_batched_verify(hidden, residual, num_tokens, state, ctx, stream);
        }
        let h = ctx.config.hidden_size;
        let h_bytes = ctx.config.ssm_h_state_bytes();
        let conv_bytes = ctx.config.ssm_conv_state_bytes();
        for t in 0..num_tokens {
            let offset = t * h * 2;
            self.decode(
                hidden.offset(offset),
                residual.offset(offset),
                state,
                kv_cache,
                seq_len + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
            // MTP reject rewinds to intermediate[num_accepted-1]. Qwen GDN
            // writes those inside the fused kernel; Mamba-2 decode does not.
            // Snapshot after every token except the last (full-accept keeps
            // live h_state). Skip when the slot has no MTP intermediates.
            if t + 1 < num_tokens {
                let ssm = state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
                if t < ssm.h_state_intermediates.len() && t < ssm.conv_state_intermediates.len() {
                    ctx.gpu.copy_d2d_async(
                        ssm.h_state,
                        ssm.h_state_intermediates[t],
                        h_bytes,
                        stream,
                    )?;
                    ctx.gpu.copy_d2d_async(
                        ssm.conv_state,
                        ssm.conv_state_intermediates[t],
                        conv_bytes,
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn decode_verify_multi<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        n_seqs: usize,
        ks: &[usize],
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _wy_tables: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_verify_multi_loop(hidden, residual, n_seqs, ks, states, _kv_cache, ctx, stream)
    }

    fn decode_multi_seq<'a, 'b: 'a>(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_seqs: usize,
        states: &'a mut [&'b mut (dyn LayerState + 'static)],
        _kv_cache: &mut PagedKvCache,
        _seq_lens: &[usize],
        _block_tables: &[Vec<u32>],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // AR C>1: batched in/out projection is the production default for
        // both hybrid mixers. Set ATLAS_LIGHTNING_DECODE_MULTI=0 for the
        // serial diagnostic fallback; component values are narrow overrides.
        if nemotron_decode_policy::decode_multi_seq_batched(
            std::env::var("ATLAS_LIGHTNING_DECODE_MULTI")
                .ok()
                .as_deref(),
            std::env::var("ATLAS_LIGHTNING_MAMBA_MULTI").ok().as_deref(),
        ) {
            self.decode_multi_seq_ar(hidden, residual, num_seqs, states, ctx, stream)
        } else {
            // Default serial diagnostic path: one per-sequence decode().
            let h = ctx.config.hidden_size;
            for i in 0..num_seqs {
                let offset = i * h * 2;
                let mut bt = _block_tables[i].clone();
                let mut stub_disk = Vec::<u32>::new();
                let mut stub_off = Vec::<u32>::new();
                self.decode(
                    hidden.offset(offset),
                    residual.offset(offset),
                    states[i],
                    _kv_cache,
                    _seq_lens[i],
                    &mut bt,
                    &mut stub_disk,
                    &mut stub_off,
                    ctx,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    fn prefill(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len_start: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _kv_write_start: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.prefill_ssm(hidden, residual, num_tokens, state, ctx, stream)
    }

    fn alloc_state(&self, gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let h_state = gpu.alloc(self.h_state_bytes)?;
        gpu.memset(h_state, 0, self.h_state_bytes)?;
        let conv_state = gpu.alloc(self.conv_state_bytes)?;
        gpu.memset(conv_state, 0, self.conv_state_bytes)?;
        Ok(Box::new(SsmLayerState {
            h_state,
            conv_state,
            h_state_checkpoint: None,
            conv_state_checkpoint: None,
            h_state_intermediates: Vec::new(),
            conv_state_intermediates: Vec::new(),
            h_is_f16: false,
            // Stage-3 narrowing is GDN-only (`ssm_h_fp16_preconditions`
            // refuses a non-GDN SSM stack), so this state is always FP32-wide.
            h_prefill_stage: None,
            ple: None,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::use_batched_mamba_verify;

    #[test]
    fn lightning_exact_routes_every_mamba_stage_through_literal_m1_decode() {
        assert!(!use_batched_mamba_verify(4, true, true));
        assert!(use_batched_mamba_verify(4, false, true));
        assert!(!use_batched_mamba_verify(1, false, true));
        assert!(!use_batched_mamba_verify(4, false, false));
    }
}
