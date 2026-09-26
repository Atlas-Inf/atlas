// SPDX-License-Identifier: AGPL-3.0-only

//! One non-authoritative native `[B, gamma]` drafter layer.
//!
//! The caller owns admission and cache-readiness gates. This method mirrors the
//! serial layer operation order and leaves the next layer input in
//! `batch_query_embed`; final logits/Markov and returned drafts remain serial.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    pub(super) fn run_batched_layer_stage(
        &self,
        layer_idx: usize,
        batch_rows: u32,
        batch_size: u32,
        max_kv_len: u32,
        // Group γ (per-sequence under adaptive; configured γ otherwise).
        gamma: u32,
        serial_block_tables: Option<&[u64]>,
        serial_attention_args: Option<spark_runtime::gpu::DevicePtr>,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| anyhow::anyhow!("DFlash batched layer {layer_idx} is missing"))?;
        let hidden = u32::try_from(self.hidden_size)
            .map_err(|_| anyhow::anyhow!("DFlash hidden width exceeds u32"))?;
        let q_dim = u32::try_from(self.num_q_heads * self.head_dim)
            .map_err(|_| anyhow::anyhow!("DFlash q width exceeds u32"))?;
        let kv_dim = u32::try_from(self.num_kv_heads * self.head_dim)
            .map_err(|_| anyhow::anyhow!("DFlash KV width exceeds u32"))?;
        let intermediate = u32::try_from(self.intermediate_size)
            .map_err(|_| anyhow::anyhow!("DFlash MLP width exceeds u32"))?;
        let hidden_elements = batch_rows
            .checked_mul(hidden)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch hidden elements overflow"))?;
        let mlp_elements = batch_rows
            .checked_mul(intermediate)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch MLP elements overflow"))?;

        crate::layers::ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            self.batch_query_embed,
            &layer.input_layernorm,
            self.batch_norm,
            batch_rows,
            hidden,
            self.rms_norm_eps,
            stream,
        )?;
        if let Some(ref conv) = layer.attention_conv {
            self.staged_conv_prepare(
                conv,
                self.batch_norm,
                batch_size,
                hidden,
                gamma,
                ctx,
                stream,
            )?;
        }
        for (weight, fp8, nvfp4, output, width) in [
            (
                &layer.q_proj,
                &layer.q_proj_fp8,
                &layer.q_proj_nvfp4,
                self.batch_q,
                q_dim,
            ),
            (
                &layer.k_proj,
                &layer.k_proj_fp8,
                &layer.k_proj_nvfp4,
                self.batch_k,
                kv_dim,
            ),
            (
                &layer.v_proj,
                &layer.v_proj_fp8,
                &layer.v_proj_nvfp4,
                self.batch_v,
                kv_dim,
            ),
        ] {
            self.run_staged_projection(
                batch_size,
                self.batch_norm,
                weight,
                fp8,
                nvfp4,
                output,
                gamma,
                width,
                hidden,
                ctx,
                stream,
            )?;
        }
        let q_rows = batch_rows
            .checked_mul(self.num_q_heads as u32)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch q-norm rows overflow"))?;
        let k_rows = batch_rows
            .checked_mul(self.num_kv_heads as u32)
            .ok_or_else(|| anyhow::anyhow!("DFlash batch k-norm rows overflow"))?;
        crate::layers::ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            self.batch_q,
            &layer.q_norm,
            self.batch_q,
            q_rows,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        crate::layers::ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            self.batch_k,
            &layer.k_norm,
            self.batch_k,
            k_rows,
            self.head_dim as u32,
            self.rms_norm_eps,
            stream,
        )?;
        crate::layers::ops::rope_yarn(
            ctx.gpu,
            self.kernels.rope_qwen3,
            self.batch_q,
            self.batch_k,
            self.batch_position_ids,
            batch_rows,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            self.rotary_dim as u32,
            self.yarn_inv_freq,
            self.rope_theta,
            stream,
        )?;
        // Attention sinks are required only by the Lightning product's
        // batched-sink kernel; generic DFlash2 passes NULL exactly like its
        // serial Option-B layer does.
        let sinks = if self.startup.native_batch_authoritative {
            layer
                .attention_sink_bias
                .as_ref()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "DFlash batched layer {layer_idx} lacks required attention sinks"
                    )
                })?
                .weight
        } else {
            layer
                .attention_sink_bias
                .as_ref()
                .map(|sinks| sinks.weight)
                .unwrap_or(DevicePtr::NULL)
        };
        let (k_pool, v_pool) = {
            let cache = self.kv_cache.lock();
            (cache.k_pool_ptr(layer_idx), cache.v_pool_ptr(layer_idx))
        };
        crate::layers::ops::reshape_and_cache(
            ctx.gpu,
            self.kernels.reshape_cache_bf16,
            self.batch_k,
            self.batch_v,
            k_pool,
            v_pool,
            self.batch_slot_mapping,
            batch_rows,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            16,
            kv_dim,
            kv_dim,
            0,
            stream,
        )?;
        self.run_staged_attention(
            gamma,
            layer_idx,
            batch_size,
            max_kv_len,
            serial_block_tables,
            serial_attention_args,
            sinks,
            k_pool,
            v_pool,
            ctx,
            stream,
        )?;
        self.run_staged_projection(
            batch_size,
            self.batch_attn_out,
            &layer.o_proj,
            &layer.o_proj_fp8,
            &layer.o_proj_nvfp4,
            self.batch_attn_proj,
            gamma,
            hidden,
            q_dim,
            ctx,
            stream,
        )?;
        if let Some(ref conv) = layer.attention_conv {
            self.staged_conv_finish(
                conv,
                self.batch_attn_proj,
                batch_size,
                hidden,
                gamma,
                ctx,
                stream,
            )?;
        }
        crate::layers::ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            self.batch_query_embed,
            self.batch_attn_proj,
            hidden_elements,
            stream,
        )?;
        crate::layers::ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            self.batch_query_embed,
            &layer.post_attention_layernorm,
            self.batch_norm,
            batch_rows,
            hidden,
            self.rms_norm_eps,
            stream,
        )?;
        if let Some(ref conv) = layer.mlp_conv {
            self.staged_conv_prepare(
                conv,
                self.batch_norm,
                batch_size,
                hidden,
                gamma,
                ctx,
                stream,
            )?;
        }
        for (weight, fp8, nvfp4, output) in [
            (
                &layer.gate_proj,
                &layer.gate_proj_fp8,
                &layer.gate_proj_nvfp4,
                self.batch_mlp_gate,
            ),
            (
                &layer.up_proj,
                &layer.up_proj_fp8,
                &layer.up_proj_nvfp4,
                self.batch_mlp_up,
            ),
        ] {
            self.run_staged_projection(
                batch_size,
                self.batch_norm,
                weight,
                fp8,
                nvfp4,
                output,
                gamma,
                intermediate,
                hidden,
                ctx,
                stream,
            )?;
        }
        crate::layers::ops::silu_mul(
            ctx.gpu,
            self.kernels.silu_mul,
            self.batch_mlp_gate,
            self.batch_mlp_up,
            self.batch_mlp_gate,
            mlp_elements,
            stream,
        )?;
        self.run_staged_projection(
            batch_size,
            self.batch_mlp_gate,
            &layer.down_proj,
            &layer.down_proj_fp8,
            &layer.down_proj_nvfp4,
            self.batch_mlp_down,
            gamma,
            hidden,
            intermediate,
            ctx,
            stream,
        )?;
        if let Some(ref conv) = layer.mlp_conv {
            self.staged_conv_finish(
                conv,
                self.batch_mlp_down,
                batch_size,
                hidden,
                gamma,
                ctx,
                stream,
            )?;
        }
        crate::layers::ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            self.batch_query_embed,
            self.batch_mlp_down,
            hidden_elements,
            stream,
        )
    }
}
