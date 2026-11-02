// SPDX-License-Identifier: AGPL-3.0-only
//! Eager causal semantic indexing for repaired K3 verification.
use super::*;
use crate::layer::ForwardContext;
use crate::layers::qwen3_attention::glm_long_context;
use anyhow::ensure;
use spark_runtime::kv_cache::{KvCacheDtype, SparseIndexCacheDtype};

impl Qwen3AttentionLayer {
    pub(super) fn validate_glm_long_verify(
        &self,
        c: &MultiSeqCtx<'_>,
        cache: &PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<bool> {
        if !glm_long_context::enabled(&c.fwd.config.model_type) {
            return Ok(false);
        }
        use crate::speculative::glm_repair_policy as policy;
        // Repaired MTP verifies exactly K3; the DFlash lane verifies its block.
        let rows_ok = if policy::enabled() {
            c.n == 3
        } else {
            policy::dflash_enabled() && (2..=policy::MAX_DFLASH_VERIFY_ROWS).contains(&c.n)
        };
        ensure!(
            rows_ok
                && c.seq_lens.len() == c.n
                && c.fwd.config.tp_world_size == 2
                && c.fwd.config.ep_world_size == 2
                && !c.fwd.graph_capture
                && !c.fwd.gpu.stream_is_capturing(c.stream),
            "GLM long verify requires repaired eager K3 or the DFlash lane, TP2/EP2"
        );
        let mla = self.mla.as_ref().expect("MLA dispatch owns weights");
        ensure!(
            mla.glm_indexer.is_some()
                && mla.rope == 0
                && mla.o_lora_rank == 0
                && mla.kv_lora_rank == 512
                && mla.q_lora_rank == 1536
                && c.h == 4096
                && c.hd == 256
                && matches!(c.nq, 32 | 64)
                && c.fwd.config.index_topk == 2048
                && c.fwd.config.index_kpool == 4
                && c.fwd.config.index_head_dim == 128
                && c.fwd.config.index_n_heads == 32,
            "GLM long MTP requires indexed NoPE512 checkpoint geometry"
        );
        let index = cache
            .sparse_index_config()
            .ok_or_else(|| anyhow::anyhow!("GLM long MTP needs semantic index storage"))?;
        ensure!(
            self.kv_dtype == KvCacheDtype::Bf16
                && cache.dtype_for_layer(self.attn_layer_idx) == KvCacheDtype::Bf16
                && index.dtype == SparseIndexCacheDtype::Bf16
                && index.head_dim == 128
                && index.tokens_per_pool == 4,
            "GLM long MTP requires BF16 K/V and semantic index"
        );
        ensure!(
            meta.num_seqs as usize == c.n
                && !meta.slot.is_null()
                && !meta.seq_len.is_null()
                && !meta.block_table.is_null(),
            "GLM long verify metadata incomplete"
        );
        let shape = ops::GlmDynamicShape::new(meta.max_blocks_per_seq, c.bs)?;
        shape.validate_positions(
            c.seq_lens.iter().copied(),
            c.n,
            c.fwd.config.max_position_embeddings,
        )?;
        ensure!(
            c.seq_lens
                .windows(2)
                .all(|p| p[0].checked_add(1) == Some(p[1])),
            "GLM long verify positions must be causal and consecutive"
        );
        shape.validate_arenas(
            c.fwd.buffers.sizes().expert_down_out,
            c.fwd.buffers.sizes().qkv_output,
        )?;
        ensure!(
            self.glm_index_layernorm_k.0 != 0
                && self.glm_index_tail_write_k.0 != 0
                && self.glm_index_kpool_finalize_k.0 != 0
                && self.glm_index_logits_decode_k.0 != 0
                && self.glm_index_topk_expand_k.0 != 0
                && self.glm_sparse_attn_decode_k.0 != 0,
            "GLM long MTP semantic kernels unavailable"
        );
        self.validate_glm_split_scratch(c, cache, meta)?;
        Ok(true)
    }

    fn validate_glm_split_scratch(
        &self,
        c: &MultiSeqCtx<'_>,
        cache: &PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        if !ops::glm_sparse_decode_split_enabled(&c.fwd.config.model_type)? {
            return Ok(());
        }
        ensure!(
            c.nq == 32 && c.bs == 16,
            "GLM split requires local32 heads/block16 before cache writes"
        );
        let b = c.fwd.buffers;
        let s = b.sizes();
        let pool_bytes = |stride: usize| {
            cache
                .num_blocks()
                .checked_mul(stride)
                .ok_or_else(|| anyhow::anyhow!("GLM split cache span overflow"))
        };
        let table_bytes = (meta.max_blocks_per_seq as usize)
            .checked_mul(3 * 4)
            .ok_or_else(|| anyhow::anyhow!("GLM split block table span overflow"))?;
        let live = [
            (c.normed, c.n * 4096 * 2),
            (b.expert_up_out(), s.expert_up_out),
            (b.expert_down_out(), s.expert_down_out),
            (b.qkv_output(), s.qkv_output),
            (b.attn_output(), s.attn_output),
            (b.moe_output(), s.moe_output),
            (b.ssm_qkvz(), s.ssm_qkvz),
            (b.ssm_ba(), s.ssm_ba),
            (b.ssm_deinterleaved(), s.ssm_deinterleaved),
            (b.ssm_gates(), s.ssm_gates),
            (b.ssm_conv_out_f32(), s.ssm_conv_out_f32),
            (meta.block_table, table_bytes),
            (
                cache.k_pool_ptr(self.attn_layer_idx),
                pool_bytes(cache.k_block_stride_bytes_for_layer(self.attn_layer_idx))?,
            ),
            (
                cache.v_pool_ptr(self.attn_layer_idx),
                pool_bytes(cache.v_block_stride_bytes_for_layer(self.attn_layer_idx))?,
            ),
        ];
        // expert_gate_out's single-row KV latent is dead after cache writeback.
        // Index selection uses expert_down_out/qkv_output; absorbed Q remains
        // expert_up_out, and retained K3 V rows remain ssm_qkvz+512. Same stream
        // completes split+merge before the next row reuses any scratch.
        ops::validate_glm_sparse_decode_split_scratch(
            &c.fwd.config.model_type,
            b.expert_gate_out(),
            s.expert_gate_out,
            &live,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_long_verify_attention(
        &self,
        c: &MultiSeqCtx<'_>,
        cache: &PagedKvCache,
        meta: AttnMetadataDev,
        normed: DevicePtr,
        q_latent: DevicePtr,
        query: DevicePtr,
        output: DevicePtr,
        position: usize,
        d: &MlaDims,
        stream: u64,
        index_query: Option<DevicePtr>,
    ) -> Result<bool> {
        if !glm_long_context::enabled(&c.fwd.config.model_type) {
            return Ok(false);
        }
        let row_ctx = ForwardContext {
            attn_metadata: Some(meta),
            midchunk_capture: None,
            ..*c.fwd
        };
        // q_latent survives the zero-RoPE chain. Index scratch overwrites only
        // consumed Q expansion / K/V assembly; normed and absorbed Q stay live.
        let selected = self.glm_index_decode_update_and_select_with_query(
            normed,
            q_latent,
            position as u32,
            cache,
            &row_ctx,
            stream,
            index_query,
        )?;
        let Some((ids, width)) = selected else {
            return Ok(false);
        };
        // The preceding single-row mla_cache_assemble writes the same NoPE
        // latent to both BF16 cache sides. Keep this row's causal index result
        // and block table; only the attention arithmetic is an opt-in change.
        let attention = ops::GlmSparsePrefillTc {
            config: c.fwd.config,
            dtype: self.kv_dtype,
            identical_kv_latent: true,
            query,
            k_cache: cache.k_pool_ptr(self.attn_layer_idx),
            v_cache: cache.v_pool_ptr(self.attn_layer_idx),
            indices: ids,
            output,
            block_table: meta.block_table,
            rows: 1,
            heads: d.nq,
            head_dim: d.mla_cache_dim,
            index_width: width,
            block_size: c.bs,
            scale: d.inv_sqrt_d,
        };
        if ops::try_glm_sparse_decode_split(
            c.fwd.gpu,
            &attention,
            c.fwd.buffers.expert_gate_out(),
            c.fwd.buffers.sizes().expert_gate_out,
            stream,
        )? || ops::try_glm_sparse_decode_tc(c.fwd.gpu, &attention, stream)?
        {
            return Ok(true);
        }
        ops::glm_sparse_mla_prefill(
            c.fwd.gpu,
            self.glm_sparse_attn_decode_k,
            query,
            cache.k_pool_ptr(self.attn_layer_idx),
            cache.v_pool_ptr(self.attn_layer_idx),
            ids,
            output,
            meta.block_table,
            1,
            d.nq,
            d.mla_cache_dim,
            width,
            c.bs,
            1,
            d.inv_sqrt_d,
            stream,
        )?;
        Ok(true)
    }
}
