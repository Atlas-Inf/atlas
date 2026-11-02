// SPDX-License-Identifier: AGPL-3.0-only

//! KV-only MLA prefill for the MTP prompt context: writes the compressed
//! latent straight to the paged cache (split out of `cache_skip_mla.rs`).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// MTP prompt-context fast path for compressed MLA. The predictor only
    /// needs its cache populated before autoregressive proposal; none of the
    /// layer output is consumed. For GLM-5 (NoPE, `mla.rope == 0`) K/V is a
    /// pure function of the combined input row, so skip Q, attention, O, and
    /// MoE entirely and write the compressed latent directly to paged cache.
    pub(crate) fn prefill_mla_kv_only_impl(
        &self,
        hidden: DevicePtr,
        num_tokens: usize,
        kv_cache: &mut PagedKvCache,
        slots: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(mla) = self.mla.as_ref() else {
            return Ok(false);
        };
        if mla.rope != 0 || num_tokens == 0 {
            return Ok(false);
        }

        if super::super::glm_long_context::enabled(&ctx.config.model_type) {
            anyhow::ensure!(
                crate::speculative::glm_repair_policy::long_lane_enabled()
                    && !ctx.graph_capture
                    && !ctx.gpu.stream_is_capturing(stream)
                    && mla.glm_indexer.is_some()
                    && kv_cache.sparse_index_config().is_some()
                    && self.kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Bf16,
                "GLM long MTP KV writer requires eager repaired BF16 indexed MLA"
            );
        }
        let n = num_tokens as u32;
        let h = ctx.config.hidden_size as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            hidden,
            &self.input_norm,
            normed,
            n,
            h,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;

        let kv_latent = ctx.buffers.expert_gate_out();
        self.mla_prefill_dense(normed, &mla.wkv_a, kv_latent, n, kv_lora, h, ctx, stream)?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_w_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n,
            kv_lora,
            ctx.config.rms_norm_eps as f32,
            stream,
        )?;

        // With a zero-width RoPE partition the compressed cache row is the KV
        // latent itself for both K and V. Keep the ordinary assembly kernel so
        // the cache layout remains identical to decode's established path.
        let k_cache = ctx.buffers.expert_up_out();
        let v_cache = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,
            ctx.buffers.ssm_ba(),
            k_cache,
            v_cache,
            n,
            kv_lora,
            0,
            kv_lora,
            stream,
        )?;
        self.write_kv_cache(
            ctx.gpu,
            k_cache,
            v_cache,
            kv_cache,
            slots,
            n,
            1,
            kv_lora,
            kv_cache.block_size() as u32,
            kv_lora,
            kv_lora,
            stream,
            false,
        )?;
        if super::super::glm_long_context::enabled(&ctx.config.model_type) {
            // The KV-only caller owns explicit slots but intentionally has no
            // attention metadata. Index population consumes only slot + rows.
            let index_ctx = ForwardContext {
                attn_metadata: Some(crate::layer::AttnMetadataDev {
                    positions: DevicePtr::NULL,
                    positions_h: DevicePtr::NULL,
                    positions_w: DevicePtr::NULL,
                    slot: slots,
                    seq_len: DevicePtr::NULL,
                    block_table: DevicePtr::NULL,
                    max_blocks_per_seq: 0,
                    num_seqs: n,
                    seq_slot: DevicePtr::NULL,
                    moe_row_adapter: DevicePtr::NULL,
                }),
                midchunk_capture: None,
                ..*ctx
            };
            self.glm_index_prefill_cache_update(normed, n, 0, kv_cache, &index_ctx, stream, None)?;
        }
        Ok(true)
    }
}
