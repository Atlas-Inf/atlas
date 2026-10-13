// SPDX-License-Identifier: AGPL-3.0-only

//! Semantic indexing for independent GLM decode rows. Q absorption and
//! KV writes finish before selectors borrow their scratch; V extraction waits
//! until every selector has released ssm_qkvz. No persistent memory is added.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, KernelHandle};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache, SparseIndexCacheDtype};

use super::ctx::MultiSeqCtx;
use super::mla_gemv::MlaDims;
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;
use crate::layers::qwen3_attention::{MlaWeights, Qwen3AttentionLayer};

fn validate_positions(
    positions: &[usize],
    rows: usize,
    model_limit: usize,
    c4_sparse: bool,
) -> Result<u32> {
    ensure!(
        matches!(rows, 2 | 3) || (rows == 4 && c4_sparse),
        "GLM sparse decode requires C2/C3 or opted-in eager C4"
    );
    if c4_sparse {
        crate::model::glm_c4::validate_positions_for_mode(positions.iter().copied(), rows, true)?;
    }
    ensure!(
        positions.len() == rows,
        "GLM sparse host-position count mismatch"
    );
    let max_position = positions.iter().copied().max().unwrap_or(0);
    ensure!(
        max_position < model_limit,
        "GLM sparse position exceeds model context"
    );
    let max_position = u32::try_from(max_position)?;
    ensure!(
        max_position < u32::MAX,
        "GLM sparse sequence length overflows u32"
    );
    Ok(max_position)
}

fn row_metadata(meta: AttnMetadataDev, row: usize) -> AttnMetadataDev {
    AttnMetadataDev {
        positions: meta.positions.offset(row * 4),
        positions_h: meta.positions_h.offset(row * 4),
        positions_w: meta.positions_w.offset(row * 4),
        slot: meta.slot.offset(row * 8),
        seq_len: meta.seq_len.offset(row * 4),
        block_table: meta
            .block_table
            .offset(row * meta.max_blocks_per_seq as usize * 4),
        max_blocks_per_seq: meta.max_blocks_per_seq,
        num_seqs: 1,
        seq_slot: DevicePtr::NULL,
        moe_row_adapter: DevicePtr::NULL,
    }
}

impl Qwen3AttentionLayer {
    pub(super) fn validate_glm_multi_seq_sparse(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        mla: &MlaWeights,
    ) -> Result<()> {
        let config = c.fwd.config;
        let c4_sparse = crate::model::glm_c4::sparse_enabled(&config.model_type);
        let max_position =
            validate_positions(c.seq_lens, c.n, config.max_position_embeddings, c4_sparse)?;
        let dynamic = crate::layers::qwen3_attention::glm_multi_seq_sparse_graphs_enabled(
            &config.model_type,
        )?;
        ensure!(
            !c4_sparse || (!dynamic && !c.fwd.graph_capture),
            "GLM C4 sparse decode is eager-only"
        );
        ensure!(
            !c.fwd.graph_capture || dynamic,
            "GLM multi-sequence sparse capture requires the device-length graph opt-in"
        );
        ensure!(
            config.model_type == "glm5_next"
                && c.h == 4096
                && c.hd == 256
                && matches!(c.nq, 32 | 64)
                && mla.nope == 256
                && mla.v_dim == 256
                && mla.kv_lora_rank == 512
                && mla.q_lora_rank == 1536
                && mla.rope == 0
                && mla.o_lora_rank == 0
                && mla.glm_indexer.is_some()
                && config.index_n_heads == 32
                && config.index_head_dim == 128
                && config.index_topk == 2048
                && config.index_kpool == 4,
            "GLM multi-sequence sparse decode requires the validated MLA/index geometry"
        );
        let spec = kv_cache
            .sparse_index_config()
            .ok_or_else(|| anyhow::anyhow!("GLM semantic-index cache is not attached"))?;
        ensure!(
            self.kv_dtype == KvCacheDtype::Bf16
                && kv_cache.dtype_for_layer(self.attn_layer_idx) == KvCacheDtype::Bf16
                && spec.dtype == SparseIndexCacheDtype::Bf16
                && spec.head_dim == 128
                && spec.tokens_per_pool == 4
                && c.bs > 0
                && c.bs.is_multiple_of(4),
            "GLM multi-sequence sparse decode requires BF16 KV/index and four-token pools"
        );
        ensure!(
            meta.num_seqs as usize == c.n
                && meta.max_blocks_per_seq > 0
                && !meta.slot.is_null()
                && !meta.seq_len.is_null()
                && !meta.block_table.is_null(),
            "GLM multi-sequence sparse metadata is incomplete"
        );
        ensure!(
            self.mla_cache_assemble_batched_k.0 != 0
                && self.glm_sparse_attn_decode_k.0 != 0
                && self.paged_decode_mla_k.0 != 0
                && self.glm_index_layernorm_k.0 != 0
                && self.glm_index_tail_write_k.0 != 0
                && self.glm_index_kpool_finalize_k.0 != 0
                && self.glm_index_logits_k.0 != 0
                && self.glm_index_logits_decode_k.0 != 0
                && self.glm_index_topk_expand_k.0 != 0,
            "GLM multi-sequence sparse kernels are unavailable"
        );
        let sizes = c.fwd.buffers.sizes();
        let shape = ops::GlmDynamicShape::new(meta.max_blocks_per_seq, c.bs)?;
        if dynamic {
            shape.validate_positions(
                c.seq_lens.iter().copied(),
                c.n,
                config.max_position_embeddings,
            )?;
            shape.validate_arenas(sizes.expert_down_out, sizes.qkv_output)?;
            ensure!(
                self.glm_index_logits_dynamic_k.0 != 0
                    && self.glm_index_topk_dynamic_k.0 != 0
                    && self.glm_sparse_attn_dynamic_k.0 != 0,
                "GLM device-length sparse kernels are unavailable"
            );
        }
        let absorbed = c.n * c.nq as usize * 512 * 2;
        let expanded = c.n * c.nq as usize * 256 * 2;
        let logits = if dynamic {
            shape.score_bytes()
        } else {
            ((max_position as usize + 1).div_ceil(4)) * 4
        };
        for (name, available, required) in [
            ("Q latent", sizes.ssm_ba, c.n * 1536 * 2),
            (
                "expanded Q / index query",
                sizes.ssm_deinterleaved,
                expanded.max(32 * 128 * 2),
            ),
            ("KV latent", sizes.expert_gate_out, c.n * 512 * 2),
            ("absorbed Q", sizes.expert_up_out, absorbed),
            ("attention output", sizes.attn_output, absorbed),
            (
                "KV entries / selected IDs",
                sizes.qkv_output,
                (c.n * 2 * 512 * 2).max(2051 * 4),
            ),
            (
                "V extraction / index keys",
                sizes.ssm_qkvz,
                expanded.max(2 * 128 * 2),
            ),
            ("index weights", sizes.ssm_gates, 32 * 2),
            ("index scores", sizes.expert_down_out, logits),
        ] {
            ensure!(
                available >= required,
                "GLM sparse {name} scratch requires {required} bytes, arena has {available}"
            );
        }
        Ok(())
    }

    /// Batched absorbed Q has already been computed when `batch_kernel` exists.
    /// Scalar fallback also finishes every Q row before any index query writes.
    pub(super) fn ms_glm_mla_sparse_attention(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
        mla: &MlaWeights,
        dims: &MlaDims,
        batch_kernel: KernelHandle,
    ) -> Result<()> {
        let buffers = c.fwd.buffers;
        let gpu = c.fwd.gpu;
        let rows = c.n as u32;
        let stream = c.stream;
        let q_full = buffers.ssm_deinterleaved();
        let q_absorbed = buffers.expert_up_out();
        let absorbed_row = c.nq as usize * 512 * 2;
        let expanded_row = c.nq as usize * 256 * 2;
        if batch_kernel.0 == 0 {
            for row in 0..c.n {
                self.ms_mla_q_absorb(
                    c,
                    mla,
                    dims,
                    q_full.offset(row * expanded_row),
                    q_absorbed.offset(row * absorbed_row),
                    stream,
                )?;
            }
        }

        // The selector uses the base of qkv_output, so all cache entries must
        // be written first, even when projection batching is disabled.
        let k_entries = buffers.qkv_output();
        let v_entries = k_entries.offset(c.n * 512 * 2);
        ops::mla_cache_assemble_batched(
            gpu,
            self.mla_cache_assemble_batched_k,
            buffers.expert_gate_out(),
            DevicePtr::NULL,
            k_entries,
            v_entries,
            rows,
            512,
            0,
            512,
            stream,
        )?;
        self.write_kv_cache(
            gpu, k_entries, v_entries, kv_cache, meta.slot, rows, 1, 512, c.bs, 512, 512, stream,
            false,
        )?;

        let attn_out = buffers.attn_output();
        let dynamic = crate::layers::qwen3_attention::glm_multi_seq_sparse_graphs_enabled(
            &c.fwd.config.model_type,
        )?;
        let shape = ops::GlmDynamicShape::new(meta.max_blocks_per_seq, c.bs)?;
        for row in 0..c.n {
            let meta_i = row_metadata(meta, row);
            let row_ctx = ForwardContext {
                ssm_batch: None,
                attn_metadata: Some(meta_i),
                midchunk_capture: None,
                ..*c.fwd
            };
            if dynamic {
                self.glm_index_dynamic_attention(
                    c.normed.offset(row * c.h * 2),
                    buffers.ssm_ba().offset(row * 1536 * 2),
                    q_absorbed.offset(row * absorbed_row),
                    attn_out.offset(row * absorbed_row),
                    kv_cache,
                    &row_ctx,
                    shape,
                    c.nq,
                    dims.inv_sqrt_d,
                    stream,
                )?;
                continue;
            }
            // Always maintain the index, including dense rows below 2048.
            // This preserves pool history across threshold and C3/C2/C1 drains.
            let selected = self.glm_index_decode_update_and_select(
                c.normed.offset(row * c.h * 2),
                buffers.ssm_ba().offset(row * 1536 * 2),
                c.seq_lens[row] as u32,
                kv_cache,
                &row_ctx,
                stream,
            )?;
            let query = q_absorbed.offset(row * absorbed_row);
            let output = attn_out.offset(row * absorbed_row);
            if let Some((indices, width)) = selected {
                ops::glm_sparse_mla_prefill(
                    gpu,
                    self.glm_sparse_attn_decode_k,
                    query,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    indices,
                    output,
                    meta_i.block_table,
                    1,
                    c.nq,
                    512,
                    width,
                    c.bs,
                    1,
                    dims.inv_sqrt_d,
                    stream,
                )?;
            } else {
                ops::paged_decode_attn_bf16(
                    gpu,
                    self.paged_decode_mla_k,
                    query,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    output,
                    meta_i.block_table,
                    meta_i.seq_len,
                    meta_i.max_blocks_per_seq,
                    1,
                    c.nq,
                    1,
                    512,
                    c.bs,
                    dims.inv_sqrt_d,
                    c.nq * 512,
                    0,
                    stream,
                )?;
            }
        }

        // Index keys/gates overwrite ssm_qkvz, so V extraction starts only
        // after the last row's selector and attention have completed in order.
        let extracted = buffers.ssm_qkvz();
        if batch_kernel.0 != 0 {
            ops::mla_batched_gemv_batchm(
                gpu,
                batch_kernel,
                attn_out,
                mla.w_uv.weight,
                extracted,
                256,
                512,
                c.nq,
                512,
                256,
                c.nq * 512,
                c.nq * 256,
                stream,
            )?;
        } else {
            for row in 0..c.n {
                self.ms_mla_v_extract(
                    c,
                    mla,
                    dims,
                    attn_out.offset(row * absorbed_row),
                    extracted.offset(row * expanded_row),
                    stream,
                )?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn independent_positions_preserve_threshold_and_tail_phases() {
        assert_eq!(
            validate_positions(&[2047, 2048, 2051], 3, 16384, false).unwrap(),
            2051
        );
        assert_eq!(
            validate_positions(&[16383, 7], 2, 16384, false).unwrap(),
            16383
        );
        assert!(validate_positions(&[2047], 2, 16384, false).is_err());
        assert!(validate_positions(&[0, 1, 2, 3, 4], 5, 16384, false).is_err());
        assert!(validate_positions(&[16384, 7], 2, 16384, false).is_err());
        assert!(validate_positions(&[u32::MAX as usize, 0], 2, usize::MAX, false).is_err());
    }

    #[test]
    fn eager_c4_positions_are_opt_in_bounded_and_preserve_drain_rows() {
        let positions = [2047, 2048, 2051, 16383];
        assert_eq!(
            validate_positions(&positions, 4, 16384, true).unwrap(),
            16383
        );
        assert!(validate_positions(&positions, 4, 16384, false).is_err());
        assert!(validate_positions(&positions, 4, 8192, true).is_err());
        for rows in 2..=4 {
            assert!(validate_positions(&positions[..rows], rows, 16384, true).is_ok());
            assert!(validate_positions(&[16384, 0, 1, 2][..rows], rows, 1_048_576, true).is_err());
        }
        assert!(validate_positions(&positions[..3], 4, 16384, true).is_err());
    }

    #[test]
    fn metadata_offsets_keep_fragmented_rows_separate() {
        let meta = AttnMetadataDev {
            positions: DevicePtr(1000),
            positions_h: DevicePtr(2000),
            positions_w: DevicePtr(3000),
            slot: DevicePtr(4000),
            seq_len: DevicePtr(5000),
            block_table: DevicePtr(6000),
            max_blocks_per_seq: 1024,
            num_seqs: 3,
            seq_slot: DevicePtr::NULL,
            moe_row_adapter: DevicePtr::NULL,
        };
        let row = row_metadata(meta, 2);
        assert_eq!(row.positions.0, 1008);
        assert_eq!(row.slot.0, 4016);
        assert_eq!(row.seq_len.0, 5008);
        assert_eq!(row.block_table.0, 6000 + 2 * 1024 * 4);
        assert_eq!(row.num_seqs, 1);
    }
}
