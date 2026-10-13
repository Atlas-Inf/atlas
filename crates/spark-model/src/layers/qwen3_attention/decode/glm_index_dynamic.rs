// SPDX-License-Identifier: AGPL-3.0-only

//! Device-driven fixed-shape selector for independent C2/C3 graph decode.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Always enqueue both attentions. Device guards ensure exactly one writes
    /// this row; the original dense arithmetic is retained below the threshold.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::qwen3_attention) fn glm_index_dynamic_attention(
        &self,
        normed: DevicePtr,
        q_latent: DevicePtr,
        query: DevicePtr,
        output: DevicePtr,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        shape: ops::GlmDynamicShape,
        num_heads: u32,
        inv_sqrt_d: f32,
        stream: u64,
    ) -> Result<()> {
        let (selected, dense_len) =
            self.glm_index_decode_dynamic_select(normed, q_latent, kv_cache, ctx, shape, stream)?;
        let meta = ctx.attn_metadata.expect("validated GLM dynamic metadata");
        ops::paged_decode_attn_bf16(
            ctx.gpu,
            self.paged_decode_mla_k,
            query,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            output,
            meta.block_table,
            dense_len,
            meta.max_blocks_per_seq,
            1,
            num_heads,
            1,
            512,
            kv_cache.block_size() as u32,
            inv_sqrt_d,
            num_heads * 512,
            0,
            stream,
        )?;
        ops::glm_sparse_mla_dynamic(
            ctx.gpu,
            self.glm_sparse_attn_dynamic_k,
            query,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            kv_cache.v_pool_ptr(self.attn_layer_idx),
            selected,
            output,
            meta.block_table,
            meta.seq_len,
            num_heads,
            shape,
            inv_sqrt_d,
            stream,
        )
    }

    /// Returns fixed (selected IDs, dense length) scratch addresses. Neither
    /// the graph topology nor its arguments depend on the current host length.
    #[allow(clippy::too_many_arguments)]
    fn glm_index_decode_dynamic_select(
        &self,
        normed: DevicePtr,
        q_latent: DevicePtr,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        shape: ops::GlmDynamicShape,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        self.glm_index_decode_update(normed, kv_cache, ctx, stream)?;
        let mla = self.mla.as_ref().expect("validated GLM dynamic MLA");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("validated GLM dynamic index");
        let meta = ctx.attn_metadata.expect("validated GLM dynamic metadata");
        let query = ctx.buffers.ssm_deinterleaved();
        let weights = ctx.buffers.ssm_gates();
        // Deliberately unconditional: dense rows pay these two projections,
        // but crossing 2048 never changes the captured graph.
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            q_latent,
            &indexer.wq_b,
            query,
            32 * 128,
            1536,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            self.dense_gemv_k,
            normed,
            &indexer.weights_proj,
            weights,
            32,
            4096,
            stream,
        )?;
        let logits = ctx.buffers.expert_down_out();
        ops::glm_index_logits_dynamic(
            ctx.gpu,
            self.glm_index_logits_dynamic_k,
            query,
            weights,
            kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
            logits,
            meta.block_table,
            meta.seq_len,
            shape,
            kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
            stream,
        )?;
        let selected = ctx.buffers.qkv_output();
        let dense_len = selected.offset(ops::GLM_DYNAMIC_DENSE_OFFSET);
        ops::glm_index_topk_expand_dynamic(
            ctx.gpu,
            self.glm_index_topk_dynamic_k,
            logits,
            selected,
            meta.seq_len,
            dense_len,
            shape,
            stream,
        )?;
        Ok((selected, dense_len))
    }
}
