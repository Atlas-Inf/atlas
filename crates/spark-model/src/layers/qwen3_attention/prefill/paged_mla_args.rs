// SPDX-License-Identifier: AGPL-3.0-only

//! `MlaPrefillArgs` assembly for the paged MLA prefill, and the GLM
//! multi-owner chunk-attention entry that reuses it.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;

impl Qwen3AttentionLayer {
    pub(super) fn mla_prefill_args(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        seq_len_start: usize,
        bs: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> super::paged_mla::MlaPrefillArgs {
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        super::paged_mla::MlaPrefillArgs {
            normed,
            num_tokens,
            n: num_tokens as u32,
            h: ctx.config.hidden_size as u32,
            nq: self
                .num_q_heads_override
                .unwrap_or(ctx.config.num_attention_heads) as u32,
            nkv,
            hd,
            seq_len_start,
            kv_dim: (nkv * hd) as usize,
            eps: ctx.config.rms_norm_eps as f32,
            bf16: 2,
            bs: bs as u32,
            stream,
        }
    }

    /// GLM MLA attention of stacked causal chunks of several sequences
    /// (`owners` tile the `rows` rows at `normed`; `ctx.attn_metadata` covers
    /// every row). Returns the pre-all-reduce output for all rows. Verify and
    /// passenger owners write every row: no KV write floor.
    pub(in crate::layers::qwen3_attention) fn prefill_attention_glm_owners(
        &self,
        owners: &[super::paged_glm::GlmChunkOwner],
        normed: DevicePtr,
        rows: usize,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(
            self.mla.as_ref().is_some_and(|m| m.glm_indexer.is_some()),
            "GLM chunk attention requires a GLM MLA layer"
        );
        let bs = kv_cache.block_size();
        let args = self.mla_prefill_args(normed, rows, 0, bs, ctx, stream);
        self.glm_chunk_attention(owners, kv_cache, ctx, &args, 0)
    }
}
