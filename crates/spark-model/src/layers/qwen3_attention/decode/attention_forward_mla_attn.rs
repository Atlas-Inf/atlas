// SPDX-License-Identifier: AGPL-3.0-only

//! Step 8 of the absorbed-MLA decode chain: GLM sparse-index selection and
//! the paged (dense, sparse or tensor-core sparse) decode attention.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::{MlaWeights, Qwen3AttentionLayer};
use crate::layer::{AttnMetadataDev, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// GLM semantic-index selection for the decode row, plus whether the
    /// cache holds the `fp8_g128` latent.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_decode_sparse_indices(
        &self,
        mla: &MlaWeights,
        meta: AttnMetadataDev,
        normed: DevicePtr,
        q_latent: DevicePtr,
        pos: Option<u32>,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(Option<(DevicePtr, u32)>, bool)> {
        let sparse_indices = if mla.glm_indexer.is_some() {
            pos.map(|token_pos| {
                self.glm_index_decode_update_and_select(
                    normed, q_latent, token_pos, kv_cache, ctx, stream,
                )
            })
            .transpose()?
            .flatten()
        } else {
            None
        };
        // An fp8_g128 latent has one reader, the TC kernel: dense rows select
        // every cached token (device length, so decode graphs stay valid).
        let fp8 = self.kv_dtype == spark_runtime::kv_cache::KvCacheDtype::Fp8G128;
        let sparse_indices = match sparse_indices {
            None if fp8 => {
                let indices = ctx.buffers.expert_gate_out();
                ops::glm_index_fill_causal_dev(
                    ctx.gpu,
                    self.glm_index_fill_causal_dev_k,
                    indices,
                    meta.seq_len,
                    2051,
                    stream,
                )?;
                Some((indices, 2051))
            }
            other => other,
        };
        Ok((sparse_indices, fp8))
    }

    /// Paged MLA decode attention of the absorbed query into `attn_out`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mla_decode_paged_attn(
        &self,
        ctx: &ForwardContext,
        kv_cache: &PagedKvCache,
        meta: AttnMetadataDev,
        sparse_indices: Option<(DevicePtr, u32)>,
        fp8: bool,
        q_absorbed_buf: DevicePtr,
        attn_out: DevicePtr,
        nq: u32,
        mla_cache_dim: u32,
        bs: usize,
        inv_sqrt_d: f32,
        stream: u64,
    ) -> Result<()> {
        if let Some((indices, index_width)) = sparse_indices {
            if self.try_glm_sparse_tc_decode_heads(
                ctx,
                q_absorbed_buf,
                kv_cache,
                indices,
                index_width,
                attn_out,
                meta.block_table,
                nq,
                mla_cache_dim,
                inv_sqrt_d,
                stream,
            )? {
                Ok(())
            } else {
                anyhow::ensure!(
                    !fp8,
                    "fp8_g128 GLM decode requires ATLAS_GLM_SPARSE_DECODE_TC=1"
                );
                ops::glm_sparse_mla_prefill(
                    ctx.gpu,
                    self.glm_sparse_attn_decode_k,
                    q_absorbed_buf,
                    kv_cache.k_pool_ptr(self.attn_layer_idx),
                    kv_cache.v_pool_ptr(self.attn_layer_idx),
                    indices,
                    attn_out,
                    meta.block_table,
                    1,
                    nq,
                    mla_cache_dim,
                    index_width,
                    kv_cache.block_size() as u32,
                    1,
                    inv_sqrt_d,
                    stream,
                )
            }
        } else {
            ops::paged_decode_attn_bf16(
                ctx.gpu,
                self.paged_decode_mla_k,
                q_absorbed_buf,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                1,
                nq,
                1,
                mla_cache_dim,
                bs as u32,
                inv_sqrt_d,
                nq * mla_cache_dim,
                0,
                stream,
            )
        }
    }

    /// Single-row GLM sparse attention through the tensor-core decode kernel,
    /// 32 heads per launch (attention is independent per head, so a 64-head
    /// MTP body runs two launches). Returns false, leaving the scalar kernel to
    /// the caller, unless every precondition of that kernel holds.
    #[allow(clippy::too_many_arguments)]
    fn try_glm_sparse_tc_decode_heads(
        &self,
        ctx: &ForwardContext,
        query: DevicePtr,
        kv_cache: &PagedKvCache,
        indices: DevicePtr,
        index_width: u32,
        output: DevicePtr,
        block_table: DevicePtr,
        heads: u32,
        head_dim: u32,
        scale: f32,
        stream: u64,
    ) -> Result<bool> {
        const GROUP: u32 = 32;
        // ops::glm_sparse_decode_split's partial O + LSE + merge scratch.
        const SPLIT_SCRATCH_BYTES: usize = 8 * 32 * 512 * 4 + 8 * 32 * 4 + 32 * 4;
        if !ops::glm_sparse_decode_tc_enabled(&ctx.config.model_type)?
            || heads == 0
            || !heads.is_multiple_of(GROUP)
            || head_dim != 512
            || index_width != 2051
            || kv_cache.block_size() != 16
            || scale != 0.0625
            || !matches!(
                self.kv_dtype,
                spark_runtime::kv_cache::KvCacheDtype::Bf16
                    | spark_runtime::kv_cache::KvCacheDtype::Fp8G128
            )
            || ctx.config.qk_rope_head_dim != 0
            || !query.0.is_multiple_of(16)
            || !output.0.is_multiple_of(4)
        {
            return Ok(false);
        }
        let group_bytes = (GROUP * head_dim) as usize * 2;
        for group in 0..(heads / GROUP) as usize {
            let a = ops::GlmSparsePrefillTc {
                config: ctx.config,
                dtype: self.kv_dtype,
                identical_kv_latent: true,
                query: query.offset(group * group_bytes),
                k_cache: kv_cache.k_pool_ptr(self.attn_layer_idx),
                v_cache: kv_cache.v_pool_ptr(self.attn_layer_idx),
                indices,
                output: output.offset(group * group_bytes),
                block_table,
                rows: 1,
                heads: GROUP,
                head_dim,
                index_width,
                block_size: 16,
                scale,
            };
            // The split kernel (8-way K split + merge) beats the single-pass TC
            // kernel at one row; it needs a scratch region disjoint from its
            // inputs, which expert_gate_out is here (query/output live in
            // expert_up_out/attn_output and MoE has not started).
            let scratch = ctx.buffers.expert_gate_out();
            let disjoint = |p: DevicePtr, n: usize| {
                let (s0, s1) = (scratch.0, scratch.0 + SPLIT_SCRATCH_BYTES as u64);
                p.0 + n as u64 <= s0 || s1 <= p.0
            };
            if ops::glm_sparse_decode_split_enabled(&ctx.config.model_type)?
                && ctx.buffers.sizes().expert_gate_out >= SPLIT_SCRATCH_BYTES
                && disjoint(a.query, group_bytes)
                && disjoint(a.output, group_bytes)
                && disjoint(indices, 2051 * 4)
                && disjoint(block_table, 4)
                && ops::try_glm_sparse_decode_split(
                    ctx.gpu,
                    &a,
                    scratch,
                    ctx.buffers.sizes().expert_gate_out,
                    stream,
                )?
            {
                continue;
            }
            if !ops::try_glm_sparse_decode_tc(ctx.gpu, &a, stream)? {
                anyhow::ensure!(
                    group == 0,
                    "GLM sparse TC decode stopped after a partial head group"
                );
                return Ok(false);
            }
        }
        Ok(true)
    }
}
