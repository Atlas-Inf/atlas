// SPDX-License-Identifier: AGPL-3.0-only

//! One non-authoritative native `[B, gamma]` drafter layer.
//!
//! The caller owns admission and cache-readiness gates. This method mirrors the
//! serial layer operation order and leaves the next layer input in
//! `batch_query_embed`; final logits/Markov and returned drafts remain serial.

use anyhow::Result;

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    pub(super) fn run_batched_layer_stage(
        &self,
        layer_idx: usize,
        batch_rows: u32,
        batch_size: u32,
        max_kv_len: u32,
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
        for (weight, output, width) in [
            (&layer.q_proj, self.batch_q, q_dim),
            (&layer.k_proj, self.batch_k, kv_dim),
            (&layer.v_proj, self.batch_v, kv_dim),
        ] {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.kernels.dense_gemm_pipelined,
                self.batch_norm,
                weight,
                output,
                batch_rows,
                width,
                hidden,
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
        let sinks = layer.attention_sink_bias.as_ref().ok_or_else(|| {
            anyhow::anyhow!("DFlash batched layer {layer_idx} lacks required attention sinks")
        })?;
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
        crate::layers::ops::prefill_attention_paged_batched_sink(
            ctx.gpu,
            self.kernels.prefill_attn_dflash_bf16_batched_sink,
            self.batch_q,
            k_pool,
            v_pool,
            self.batch_attn_out,
            self.batch_block_table_ptrs,
            batch_size,
            self.batch_cu_seqlens,
            self.batch_kv_lens,
            self.gamma as u32,
            max_kv_len,
            0,
            self.num_q_heads as u32,
            self.num_kv_heads as u32,
            self.head_dim as u32,
            16,
            self.attn_sliding_window(),
            1.0 / (self.head_dim as f32).sqrt(),
            sinks.weight,
            stream,
        )?;
        crate::layers::ops::dense_gemm_bf16_pipelined(
            ctx.gpu,
            self.kernels.dense_gemm_pipelined,
            self.batch_attn_out,
            &layer.o_proj,
            self.batch_attn_proj,
            batch_rows,
            hidden,
            q_dim,
            stream,
        )?;
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
        for (weight, output) in [
            (&layer.gate_proj, self.batch_mlp_gate),
            (&layer.up_proj, self.batch_mlp_up),
        ] {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.kernels.dense_gemm_pipelined,
                self.batch_norm,
                weight,
                output,
                batch_rows,
                intermediate,
                hidden,
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
        crate::layers::ops::dense_gemm_bf16_pipelined(
            ctx.gpu,
            self.kernels.dense_gemm_pipelined,
            self.batch_mlp_gate,
            &layer.down_proj,
            self.batch_mlp_down,
            batch_rows,
            hidden,
            intermediate,
            stream,
        )?;
        crate::layers::ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            self.batch_query_embed,
            self.batch_mlp_down,
            hidden_elements,
            stream,
        )
    }

    /// Stage final norm, shared LM head, and unbiased per-row argmax. The token
    /// buffer is consumed only by the forthcoming batch-wide Markov stage.
    pub(super) fn run_batched_tail_base(
        &self,
        batch_rows: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let hidden = u32::try_from(self.hidden_size)
            .map_err(|_| anyhow::anyhow!("DFlash hidden width exceeds u32"))?;
        let vocab = u32::try_from(self.vocab_size)
            .map_err(|_| anyhow::anyhow!("DFlash vocab exceeds u32"))?;
        crate::layers::ops::rms_norm(
            ctx.gpu,
            self.kernels.rms_norm,
            self.batch_query_embed,
            &self.norm,
            self.batch_norm,
            batch_rows,
            hidden,
            self.rms_norm_eps,
            stream,
        )?;
        if matches!(self.quant, super::DflashQuantization::Fp8Weights) {
            if let Some(fp8) = self.lm_head_shared_fp8.as_ref() {
                crate::layers::ops::fp8_gemm_n128_row_scaled(
                    ctx.gpu,
                    self.kernels.fp8_gemm_n128_row_scaled,
                    self.batch_norm,
                    fp8,
                    self.batch_logits,
                    batch_rows,
                    vocab,
                    hidden,
                    stream,
                )?;
            } else {
                crate::layers::ops::dense_gemm_bf16_pipelined(
                    ctx.gpu,
                    self.kernels.dense_gemm_pipelined,
                    self.batch_norm,
                    &crate::weight_map::DenseWeight {
                        weight: self.lm_head_shared,
                    },
                    self.batch_logits,
                    batch_rows,
                    vocab,
                    hidden,
                    stream,
                )?;
            }
        } else if let Some(nvfp4) = self.lm_head_nvfp4.as_ref() {
            anyhow::ensure!(
                self.kernels.w4a16_gemm.0 != 0,
                "DFlash batched NVFP4 LM head kernel is unresolved"
            );
            crate::layers::ops::w4a16_gemm(
                ctx.gpu,
                self.kernels.w4a16_gemm,
                self.batch_norm,
                nvfp4,
                self.batch_logits,
                batch_rows,
                vocab,
                hidden,
                stream,
            )?;
        } else {
            crate::layers::ops::dense_gemm_bf16_pipelined(
                ctx.gpu,
                self.kernels.dense_gemm_pipelined,
                self.batch_norm,
                &crate::weight_map::DenseWeight {
                    weight: self.lm_head_shared,
                },
                self.batch_logits,
                batch_rows,
                vocab,
                hidden,
                stream,
            )?;
        }
        crate::layers::ops::argmax_bf16_batch(
            ctx.gpu,
            self.kernels.argmax_batch,
            self.batch_logits,
            self.batch_tokens,
            vocab,
            batch_rows,
            vocab,
            stream,
        )
    }

    /// Apply DSpark Markov bias depth-serial and batch-wide. Row 0 remains the
    /// unbiased anchor; rows 1..gamma are overwritten in `batch_tokens`.
    pub(super) fn run_batched_markov(
        &self,
        batch_size: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.markov_rank == 0 {
            return Ok(());
        }
        let (w1, w2) = self
            .markov_w1
            .as_ref()
            .zip(self.markov_w2.as_ref())
            .ok_or_else(|| anyhow::anyhow!("DFlash batched Markov weights are missing"))?;
        anyhow::ensure!(
            !self.batch_markov_embed.is_null() && !self.batch_markov_bias.is_null(),
            "DFlash batched Markov scratch is null"
        );
        let rank = u32::try_from(self.markov_rank)
            .map_err(|_| anyhow::anyhow!("DFlash Markov rank exceeds u32"))?;
        let vocab = u32::try_from(self.vocab_size)
            .map_err(|_| anyhow::anyhow!("DFlash vocab exceeds u32"))?;
        let gamma =
            u32::try_from(self.gamma).map_err(|_| anyhow::anyhow!("DFlash gamma exceeds u32"))?;
        let row_stride = gamma
            .checked_mul(vocab)
            .ok_or_else(|| anyhow::anyhow!("DFlash Markov row stride overflow"))?;
        for depth in 1..self.gamma {
            crate::layers::ops::batched_embed(
                ctx.gpu,
                self.kernels.batched_embed,
                self.batch_markov_prev,
                w1.weight,
                self.batch_markov_embed,
                batch_size,
                rank,
                stream,
            )?;
            crate::layers::ops::dense_gemv_batchm(
                ctx.gpu,
                self.kernels.dense_gemv_batchm,
                self.batch_markov_embed,
                w2,
                self.batch_markov_bias,
                batch_size,
                vocab,
                rank,
                vocab,
                stream,
            )?;
            crate::layers::ops::dflash_batch_add_depth_bias(
                ctx.gpu,
                self.kernels.batch_markov_add_bias,
                self.batch_logits,
                self.batch_markov_bias,
                batch_size,
                gamma,
                vocab,
                depth as u32,
                stream,
            )?;
            let logits_offset = depth
                .checked_mul(self.vocab_size)
                .and_then(|elements| elements.checked_mul(2))
                .ok_or_else(|| anyhow::anyhow!("DFlash Markov logits offset overflow"))?;
            crate::layers::ops::argmax_bf16_batch(
                ctx.gpu,
                self.kernels.argmax_batch,
                self.batch_logits.offset(logits_offset),
                self.batch_markov_prev,
                vocab,
                batch_size,
                row_stride,
                stream,
            )?;
            crate::layers::ops::dflash_batch_store_depth_tokens(
                ctx.gpu,
                self.kernels.batch_markov_store_tokens,
                self.batch_tokens,
                self.batch_markov_prev,
                batch_size,
                gamma,
                depth as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
