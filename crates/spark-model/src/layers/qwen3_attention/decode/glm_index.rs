// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 semantic-index maintenance and selection for single-token decode.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache, SparseIndexCacheDtype};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Shared eager/graph maintenance. Position and pool phase come from the
    /// uploaded slot, so every call in this chain is safe to capture.
    pub(in crate::layers::qwen3_attention) fn glm_index_decode_update(
        &self,
        normed: DevicePtr,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mla = self.mla.as_ref().expect("GLM decode index without MLA");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("GLM decode index without checkpoint weights");
        let spec = kv_cache
            .sparse_index_config()
            .ok_or_else(|| anyhow::anyhow!("GLM semantic-index cache is not attached"))?;
        ensure!(
            spec.dtype == SparseIndexCacheDtype::Bf16
                && matches!(
                    kv_cache.dtype_for_layer(self.attn_layer_idx),
                    KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128
                ),
            "GLM sparse decode requires a BF16 index cache and a BF16 or fp8_g128 KV cache"
        );
        ensure!(
            self.glm_index_layernorm_k.0 != 0
                && self.glm_index_tail_write_k.0 != 0
                && self.glm_index_kpool_finalize_k.0 != 0,
            "GLM decode index-maintenance kernels are unavailable"
        );

        let dim = spec.head_dim as u32;
        let h = ctx.config.hidden_size as u32;
        let keys = ctx.buffers.ssm_qkvz();
        let gates = keys.offset(spec.head_dim * 2);
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &indexer.wk,
            keys,
            dim,
            h,
            stream,
        )?;
        ops::glm_index_layernorm(
            ctx.gpu,
            self.glm_index_layernorm_k,
            keys,
            indexer.k_norm_weight.weight,
            indexer.k_norm_bias.weight,
            1,
            dim,
            1e-6,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &indexer.kpool_gate,
            gates,
            dim,
            h,
            stream,
        )?;
        let meta = ctx
            .attn_metadata
            .expect("GLM decode index requires slot metadata");
        ops::glm_index_tail_write(
            ctx.gpu,
            self.glm_index_tail_write_k,
            keys,
            gates,
            kv_cache.sparse_index_tail_pool_ptr(self.attn_layer_idx),
            kv_cache.sparse_index_tail_map_ptr(),
            meta.slot,
            1,
            kv_cache.block_size() as u32,
            spec.tokens_per_pool as u32,
            dim,
            kv_cache.sparse_index_tail_block_stride_bytes(self.attn_layer_idx) as u64,
            stream,
        )?;
        ops::glm_index_kpool_finalize(
            ctx.gpu,
            self.glm_index_kpool_finalize_k,
            kv_cache.sparse_index_tail_pool_ptr(self.attn_layer_idx),
            kv_cache.sparse_index_tail_map_ptr(),
            indexer.kpool_ape.weight,
            kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
            meta.slot,
            1,
            kv_cache.block_size() as u32,
            spec.tokens_per_pool as u32,
            dim,
            kv_cache.sparse_index_tail_block_stride_bytes(self.attn_layer_idx) as u64,
            kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
            stream,
        )?;
        Ok(())
    }

    /// Maintain semantic history, then select IDs above the exact threshold.
    /// This eager variant embeds the host position and must not be captured.
    pub(in crate::layers::qwen3_attention) fn glm_index_decode_update_and_select(
        &self,
        normed: DevicePtr,
        q_latent: DevicePtr,
        pos: u32,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, u32)>> {
        self.glm_index_decode_update_and_select_with_query(
            normed, q_latent, pos, kv_cache, ctx, stream, None,
        )
    }

    /// Variant used by the repaired K3 query stage.  Index maintenance and
    /// selection stay exactly where the scalar consumer performs them; only
    /// the stateless index-Q projection can be supplied from retained scratch.
    pub(in crate::layers::qwen3_attention) fn glm_index_decode_update_and_select_with_query(
        &self,
        normed: DevicePtr,
        q_latent: DevicePtr,
        pos: u32,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        precomputed_query: Option<DevicePtr>,
    ) -> Result<Option<(DevicePtr, u32)>> {
        if let Some(query) = precomputed_query {
            ensure!(
                !query.is_null() && query.0 % 16 == 0,
                "GLM sparse decode precomputed index query is invalid"
            );
        }
        self.glm_index_decode_update(normed, kv_cache, ctx, stream)?;
        let mla = self.mla.as_ref().expect("GLM decode index without MLA");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("GLM decode index without weights");
        let spec = kv_cache
            .sparse_index_config()
            .expect("validated semantic cache");
        let meta = ctx
            .attn_metadata
            .expect("GLM decode index requires metadata");
        let h = ctx.config.hidden_size as u32;
        let seq_len = pos + 1;
        let topk = ctx.config.index_topk as u32;
        if seq_len <= topk {
            return Ok(None);
        }
        ensure!(
            self.glm_index_logits_k.0 != 0 && self.glm_index_topk_expand_k.0 != 0,
            "GLM sparse decode selection kernels are unavailable"
        );
        let index_heads = ctx.config.index_n_heads as u32;
        let index_dim = ctx.config.index_head_dim as u32;
        let pool_size = spec.tokens_per_pool as u32;
        let logits_stride = seq_len.div_ceil(pool_size);
        let output_width = topk + pool_size - 1;
        let query = if let Some(query) = precomputed_query {
            query
        } else {
            let query = ctx.buffers.ssm_deinterleaved();
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_k,
                q_latent,
                &indexer.wq_b,
                query,
                index_heads * index_dim,
                mla.q_lora_rank as u32,
                stream,
            )?;
            query
        };
        let weights = ctx.buffers.ssm_gates();
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &indexer.weights_proj,
            weights,
            index_heads,
            h,
            stream,
        )?;
        super::super::glm_index_capacity::tile_rows(
            1,
            logits_stride as usize,
            ctx.buffers.sizes().expert_down_out,
        )?;
        let logits = ctx.buffers.expert_down_out();
        ops::glm_index_logits(
            ctx.gpu,
            self.glm_index_logits_decode_k,
            query,
            weights,
            kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
            logits,
            meta.block_table,
            1,
            pos,
            logits_stride,
            index_heads,
            index_dim,
            pool_size,
            kv_cache.block_size() as u32,
            kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
            1,
            8,
            stream,
        )?;
        // Cache writeback has completed in stream order, so the QKV arena can
        // safely hold the compact token IDs until attention consumes them.
        let selected = ctx.buffers.qkv_output();
        ops::glm_index_topk_expand(
            ctx.gpu,
            self.glm_index_topk_expand_k,
            logits,
            selected,
            1,
            pos,
            logits_stride,
            topk,
            pool_size,
            output_width,
            stream,
        )?;
        Ok(Some((selected, output_width)))
    }
}
