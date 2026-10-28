// SPDX-License-Identifier: AGPL-3.0-only

//! The canonical form of few-row GLM MLA attention on an unsharded pair
//! (`ops::glm_sparse_canonical`): the owners a token-sharded pair runs in
//! its merge form (verify, decode, prefill tails) compute the same bits here.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::glm_kv_shard::{LATENT, WIDTH};
use crate::layers::ops;

/// One owner's inputs to the canonical form.
#[derive(Clone, Copy)]
pub(in crate::layers::qwen3_attention) struct CanonicalRows {
    /// This rank's heads' absorbed queries `[rows, 32, 512]` BF16.
    pub query: DevicePtr,
    /// Selected IDs `[rows, 2051]`; `None` attends causally, row `r` to
    /// tokens `[0, causal_start + r + 1)`.
    pub selected: Option<DevicePtr>,
    pub causal_start: u32,
    /// The sequence's block table (logical → physical).
    pub block_table: DevicePtr,
    pub rows: u32,
}

/// A few-row prefill or verify owner's canonical inputs: its absorbed
/// `query` rows and `(IDs, width)` selection (`None`: the whole causal
/// history). Taps the selection for `ATLAS_GLM_DET_TRACE` as every owner does.
pub(super) fn owner_rows(
    o: &super::GlmChunkOwner,
    query: DevicePtr,
    selected: Option<(DevicePtr, u32)>,
    ctx: &ForwardContext,
    stream: u64,
) -> CanonicalRows {
    let selected = selected.map(|(ids, width)| {
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        det.tap("sel", ids, (o.row0, o.rows), width as usize * 4);
        ids
    });
    CanonicalRows {
        query,
        selected,
        causal_start: o.seq_len_start as u32,
        block_table: o.meta.block_table,
        rows: o.rows as u32,
    }
}

/// The canonical scratch: the MoE expert scratch, dead until the layer's FFN,
/// from [`ops::CANONICAL_SCRATCH_OFFSET`].
fn scratch(ctx: &ForwardContext) -> (DevicePtr, usize) {
    let bytes = ctx.buffers.sizes().expert_gate_out;
    (
        ctx.buffers
            .expert_gate_out()
            .offset(ops::CANONICAL_SCRATCH_OFFSET),
        bytes.saturating_sub(ops::CANONICAL_SCRATCH_OFFSET),
    )
}

impl Qwen3AttentionLayer {
    /// The rank whose heads an unsharded owner of `rows` rows over `heads`
    /// local heads attends in the canonical form, or `None` for the earlier
    /// kernels: on a sharded cache (its merge form is the canonical form), off
    /// a GLM pair, past [`crate::layers::glm_kv_shard::MERGE_MAX_ROWS`] rows,
    /// under `ATLAS_GLM_KV_CANONICAL=0`, or (logged once) when the arena's
    /// expert scratch cannot hold the owner.
    pub(in crate::layers::qwen3_attention) fn glm_canonical_rank(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        heads: u32,
        rows: usize,
    ) -> Result<Option<u32>> {
        let fits = kv_cache.latent_shard().is_none()
            && self
                .mla
                .as_ref()
                .is_some_and(|m| m.glm_indexer.is_some() && m.kv_lora_rank == LATENT as usize)
            && kv_cache.block_size() == 16;
        let rank = match fits {
            true => ops::glm_kv_canonical_rank(ctx.config, heads, rows)?,
            false => None,
        };
        let need = ops::CanonicalLayout::bytes(rows as u32);
        if rank.is_some() && scratch(ctx).1 < need {
            if ctx.gpu.op_cache().once("glm:canonical_scratch") {
                tracing::warn!(
                    "GLM canonical attention needs {need} bytes of expert scratch for {rows} rows, \
                     the arena has {}: such owners take the earlier kernels, whose bits differ \
                     from a KV-sharded pair's",
                    scratch(ctx).1
                );
            }
            return Ok(None);
        }
        Ok(rank)
    }

    /// Rank `rank`'s heads' attention for one canonical owner into `output`
    /// (`[rows, 32, 512]` BF16), over this layer's whole latent pool.
    pub(in crate::layers::qwen3_attention) fn glm_canonical_attention(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        rank: u32,
        o: CanonicalRows,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        let hd = self.mla.as_ref().map_or(0, |m| m.nope) as u32;
        let pool = kv_cache.k_pool_ptr(self.attn_layer_idx);
        let a = ops::GlmSparsePrefillTc {
            config: ctx.config,
            dtype: self.kv_dtype,
            identical_kv_latent: true,
            query: o.query,
            k_cache: pool,
            v_cache: pool,
            indices: o.selected.unwrap_or(DevicePtr::NULL),
            output,
            block_table: o.block_table,
            rows: o.rows,
            heads: crate::layers::glm_kv_shard::HEADS,
            head_dim: LATENT,
            index_width: WIDTH,
            block_size: 16,
            scale: self.effective_attn_scale(hd),
        };
        ops::glm_sparse_canonical(
            ctx.gpu,
            &a,
            o.selected,
            o.causal_start,
            rank,
            scratch(ctx),
            stream,
        )
    }
}
