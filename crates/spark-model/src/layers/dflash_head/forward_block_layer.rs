// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer body of `BlockDiffusionDraftHead::forward_block`. Extracted
//! from `forward_block.rs` so the parent file fits the 500-LoC budget.
//! Contains the 12-step kernel chain (input_layernorm → q/k/v projections
//! → ctx K/V override → q_norm/k_norm → RoPE → attention → o_proj →
//! residual → post_attention_layernorm → MLP gate/up → silu_mul →
//! down_proj → residual). Called once per drafter layer from
//! `forward_block`'s Step 3 loop.

use anyhow::Result;

use super::{BlockDiffusionDraftHead, DflashLayer, DflashScratch};
use crate::layer::ForwardContext;

/// Inputs passed to the per-layer kernel chain. Holds local computations
/// from the surrounding `forward_block` body so the helper can be called
/// without re-deriving them in every layer iteration.
#[allow(clippy::too_many_arguments)]
pub(super) struct LayerArgs {
    pub layer_idx: usize,
    pub n_attn: u32,
    pub eff_ctx: usize,
    pub h: u32,
    pub q_dim: u32,
    pub kv_dim: u32,
    pub inter: u32,
    pub bf16: usize,
    pub inv_sqrt_d: f32,
    pub stream: u64,
}

impl BlockDiffusionDraftHead {
    /// Run one drafter transformer layer. Mutates `scratch.*` buffers
    /// in place, leaving `stream_buf` updated with the layer's output.
    pub(super) fn forward_block_layer(
        &self,
        layer: &DflashLayer,
        args: &LayerArgs,
        ctx: &ForwardContext,
        debug_dump: bool,
        scratch: &DflashScratch,
    ) -> Result<()> {
        use crate::layers::ops;

        let LayerArgs {
            layer_idx,
            n_attn,
            eff_ctx,
            h,
            q_dim,
            kv_dim,
            inter,
            bf16,
            inv_sqrt_d,
            stream,
        } = *args;
        let gpu = ctx.gpu;

        let dump_bf16 = |label: &str, ptr: spark_runtime::gpu::DevicePtr, n: usize| -> Result<()> {
            if !debug_dump {
                return Ok(());
            }
            let mut buf = vec![0u8; n * 2];
            gpu.synchronize(stream)?;
            gpu.copy_d2h(ptr, &mut buf)?;
            let vals: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("DFLASH DUMP {label} [{n}]: {:?}", &vals);
            Ok(())
        };

        // 3a. input_layernorm.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            scratch.stream_buf,
            &layer.input_layernorm,
            scratch.norm_buf,
            n_attn,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        let attn_conv_out_delta = if let Some(ref conv) = layer.attention_conv {
            let out_delta = conv.prepare(
                gpu,
                self.kernels.dense_gemm_pipelined,
                self.kernels.dflash2_conv,
                scratch.norm_buf,
                scratch.dflash2_conv_delta,
                scratch.dflash2_conv_out,
                n_attn,
                stream,
            )?;
            gpu.copy_d2d_async(
                scratch.dflash2_conv_out,
                scratch.norm_buf,
                (n_attn * h * bf16 as u32) as usize,
                stream,
            )?;
            Some(out_delta)
        } else {
            None
        };

        // 3b. q/k/v projections from norm_buf (n_attn rows).
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            scratch.norm_buf,
            &layer.q_proj,
            scratch.q_buf,
            n_attn,
            q_dim,
            h,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            scratch.norm_buf,
            &layer.k_proj,
            scratch.k_buf,
            n_attn,
            kv_dim,
            h,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            scratch.norm_buf,
            &layer.v_proj,
            scratch.v_buf,
            n_attn,
            kv_dim,
            h,
            stream,
        )?;

        // 3b'. Ctx K/V override (skip input_layernorm; project fc_proj
        // directly through layer.k_proj/v_proj for ctx slots).
        if eff_ctx > 0 {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                scratch.fc_proj,
                &layer.k_proj,
                scratch.k_buf,
                eff_ctx as u32,
                kv_dim,
                h,
                stream,
            )?;
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.kernels.dense_gemm_pipelined,
                scratch.fc_proj,
                &layer.v_proj,
                scratch.v_buf,
                eff_ctx as u32,
                kv_dim,
                h,
                stream,
            )?;
            // Force ctx-slot Q to zeros — Q-side ctx contributes nothing
            // meaningful (gets discarded at lm_head extraction).
            gpu.memset(scratch.q_buf, 0, eff_ctx * q_dim as usize * bf16)?;
        }

        if layer_idx == 0 {
            dump_bf16("layer0.k_buf[ctx0].pre_k_norm", scratch.k_buf, 10)?;
            dump_bf16("layer0.v_buf[ctx0]", scratch.v_buf, 10)?;
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].pre_q_norm",
                scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].pre_k_norm",
                scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
        }

        // 3c. q_norm / k_norm — per-head RMSNorm over head_dim slices.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            scratch.q_buf,
            &layer.q_norm,
            scratch.q_buf,
            n_attn * self.num_q_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            scratch.k_buf,
            &layer.k_norm,
            scratch.k_buf,
            n_attn * self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        if layer_idx == 0 {
            dump_bf16("layer0.k_buf[ctx0].post_k_norm", scratch.k_buf, 10)?;
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].post_q_norm",
                scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].post_k_norm",
                scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
        }

        // 3d. yarn RoPE — n_attn positions.
        ops::rope_yarn(
            gpu,
            self.kernels.rope_qwen3,
            scratch.q_buf,
            scratch.k_buf,
            scratch.position_ids,
            n_attn,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            let noise_k_offset = eff_ctx * kv_dim as usize * bf16;
            dump_bf16(
                "layer0.q_buf[noise0].post_rope",
                scratch.q_buf.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.k_buf[noise0].post_rope",
                scratch.k_buf.offset(noise_k_offset),
                10,
            )?;
            dump_bf16("layer0.k_buf[ctx0].post_rope", scratch.k_buf, 10)?;
        }

        // 3e. attention — causal/SWA from the drafter config (Lightning:
        // dflash_config.causal=true, swa_window_size=1024).
        ops::prefill_attention(
            gpu,
            self.kernels.prefill_attn,
            scratch.q_buf,
            scratch.k_buf,
            scratch.v_buf,
            scratch.attn_out,
            n_attn,
            1,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            inv_sqrt_d,
            self.attn_causal(),
            self.attn_sliding_window(),
            stream,
        )?;
        if layer_idx == 0 {
            let noise_q_offset = eff_ctx * q_dim as usize * bf16;
            dump_bf16(
                "layer0.attn_out[noise0]",
                scratch.attn_out.offset(noise_q_offset),
                10,
            )?;
            dump_bf16(
                "layer0.attn_out[noise0][1000..1010]",
                scratch.attn_out.offset(noise_q_offset + 1000 * bf16),
                10,
            )?;
            dump_bf16(
                "layer0.attn_out[noise0][4086..4096]",
                scratch.attn_out.offset(noise_q_offset + 4086 * bf16),
                10,
            )?;
        }

        // 3f. o_proj.
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            scratch.attn_out,
            &layer.o_proj,
            scratch.stream_acc,
            n_attn,
            h,
            q_dim,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_acc[noise0].post_o_proj",
                scratch.stream_acc.offset(noise_offset),
                10,
            )?;
            dump_bf16(
                "layer0.stream_buf[noise0].pre_residual",
                scratch.stream_buf.offset(noise_offset),
                10,
            )?;
        }

        if let (Some(ref conv), Some(out_delta)) =
            (layer.attention_conv.as_ref(), attn_conv_out_delta)
        {
            conv.finish(
                gpu,
                self.kernels.dflash2_conv,
                scratch.stream_acc,
                out_delta,
                scratch.dflash2_conv_out,
                n_attn,
                stream,
            )?;
            gpu.copy_d2d_async(
                scratch.dflash2_conv_out,
                scratch.stream_acc,
                (n_attn * h * bf16 as u32) as usize,
                stream,
            )?;
        }

        // 3g. residual: stream_buf += stream_acc (n_attn rows).
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            scratch.stream_buf,
            scratch.stream_acc,
            n_attn * h,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_buf[noise0].post_attn_residual",
                scratch.stream_buf.offset(noise_offset),
                10,
            )?;
        }

        // 3h. post_attention_layernorm.
        ops::rms_norm(
            gpu,
            self.kernels.rms_norm,
            scratch.stream_buf,
            &layer.post_attention_layernorm,
            scratch.norm_buf,
            n_attn,
            h,
            self.rms_norm_eps,
            stream,
        )?;

        let mlp_conv_out_delta = if let Some(ref conv) = layer.mlp_conv {
            let out_delta = conv.prepare(
                gpu,
                self.kernels.dense_gemm_pipelined,
                self.kernels.dflash2_conv,
                scratch.norm_buf,
                scratch.dflash2_conv_delta,
                scratch.dflash2_conv_out,
                n_attn,
                stream,
            )?;
            gpu.copy_d2d_async(
                scratch.dflash2_conv_out,
                scratch.norm_buf,
                (n_attn * h * bf16 as u32) as usize,
                stream,
            )?;
            Some(out_delta)
        } else {
            None
        };

        // 3i. MLP: gate + up.
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            scratch.norm_buf,
            &layer.gate_proj,
            scratch.mlp_intermediate,
            n_attn,
            inter,
            h,
            stream,
        )?;
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            scratch.norm_buf,
            &layer.up_proj,
            scratch.mlp_up,
            n_attn,
            inter,
            h,
            stream,
        )?;

        // 3j. silu_mul.
        ops::silu_mul(
            gpu,
            self.kernels.silu_mul,
            scratch.mlp_intermediate,
            scratch.mlp_up,
            scratch.mlp_intermediate,
            n_attn * inter,
            stream,
        )?;

        // 3k. down_proj.
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            scratch.mlp_intermediate,
            &layer.down_proj,
            scratch.stream_acc,
            n_attn,
            h,
            inter,
            stream,
        )?;

        if let (Some(ref conv), Some(out_delta)) = (layer.mlp_conv.as_ref(), mlp_conv_out_delta) {
            conv.finish(
                gpu,
                self.kernels.dflash2_conv,
                scratch.stream_acc,
                out_delta,
                scratch.dflash2_conv_out,
                n_attn,
                stream,
            )?;
            gpu.copy_d2d_async(
                scratch.dflash2_conv_out,
                scratch.stream_acc,
                (n_attn * h * bf16 as u32) as usize,
                stream,
            )?;
        }

        // 3l. residual.
        ops::residual_add(
            gpu,
            self.kernels.residual_add,
            scratch.stream_buf,
            scratch.stream_acc,
            n_attn * h,
            stream,
        )?;
        if layer_idx == 0 {
            let noise_offset = eff_ctx * self.hidden_size * bf16;
            dump_bf16(
                "layer0.stream_buf[noise0].post_layer",
                scratch.stream_buf.offset(noise_offset),
                10,
            )?;
        }

        Ok(())
    }
}
