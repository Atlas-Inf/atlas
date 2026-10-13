// SPDX-License-Identifier: AGPL-3.0-only

//! GLM chunk owners: the owner/piece types, owner-batched verify
//! projections, and the `fp8_g128` BF16 latent view.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};

use super::super::super::Qwen3AttentionLayer;
use super::projection;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// One owner of a (possibly multi-sequence) GLM chunk attention: `rows`
/// rows at `row0` of the joint inputs, continuing its sequence at
/// `seq_len_start` with its own single-sequence metadata.
#[derive(Clone, Copy)]
pub(in crate::layers::qwen3_attention) struct GlmChunkOwner {
    pub row0: usize,
    pub rows: usize,
    pub seq_len_start: usize,
    pub meta: crate::layer::AttnMetadataDev,
}

impl GlmChunkOwner {
    /// Leading rows of this owner below a KV write floor of `floor` stacked
    /// rows: they keep their cached index tails and pooled keys.
    pub(super) fn write_skip(&self, floor: usize) -> usize {
        floor.saturating_sub(self.row0).min(self.rows)
    }

    /// Dense attention over the owner's whole causal history is exact: its
    /// sequence ends within `index_topk`, so it selects nothing.
    pub(super) fn dense_is_exact(&self, index_topk: usize) -> bool {
        self.seq_len_start
            .checked_add(self.rows)
            .is_some_and(|end| super::dense_selection_is_exact(end, index_topk))
    }
}

/// `ATLAS_GLM_KV_WRITE_FLOOR_LEGACY=1`. Read once, and logged then. Both
/// ranks must run the same value (`model::startup_parity`), or each attends
/// different rows of the cache.
pub(crate) fn write_floor_legacy() -> bool {
    static LEGACY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *LEGACY.get_or_init(|| {
        let on = std::env::var("ATLAS_GLM_KV_WRITE_FLOOR_LEGACY").as_deref() == Ok("1");
        if on {
            tracing::warn!(
                "ATLAS_GLM_KV_WRITE_FLOOR_LEGACY=1: GLM paged prefill ignores the KV write floor"
            );
        }
        on
    })
}

/// The KV write floor the single-sequence paged prefill applies: `floor`,
/// or 0 under `ATLAS_GLM_KV_WRITE_FLOOR_LEGACY=1`, which rewrites every
/// replayed row as the path did before it honoured the floor. Diagnostic,
/// for a same-binary A/B (like `ATLAS_NO_TAIL_SPLIT`); it also switches
/// `ATLAS_GLM_PC_WRITE_FLOOR` off.
pub(super) fn honoured_write_floor(floor: usize) -> usize {
    if write_floor_legacy() { 0 } else { floor }
}

/// `floor` capped to the `rows` stacked rows of `owners`. It must leave every
/// index pool whole: skip nothing, or everything, or end where a pool does.
/// A floor inside a pool would finalize that pool's key from cached and new
/// raw tails mixed. Host-side, from rank-identical inputs.
pub(super) fn checked_write_floor(
    owners: &[GlmChunkOwner],
    floor: usize,
    rows: usize,
    tokens_per_pool: usize,
) -> Result<usize> {
    let floor = floor.min(rows);
    let start = owners[0].seq_len_start;
    anyhow::ensure!(
        floor == 0
            || floor == rows
            || (tokens_per_pool > 0 && (start + floor).is_multiple_of(tokens_per_pool)),
        "GLM KV write floor {floor} of {rows} rows from token {start} splits an index pool"
    );
    Ok(floor)
}

/// Whether any owner runs the semantic-index selection, the only reader of
/// the index queries and weights.
pub(super) fn any_owner_selects(owners: &[GlmChunkOwner], index_topk: usize) -> bool {
    owners.iter().any(|o| !o.dense_is_exact(index_topk))
}

/// Row-wise projections of an owner-batched verify, one row per stacked row:
/// owner rows start at `row0 * <row bytes>`. `index` (queries, weights) is
/// `None` when no owner selects ([`any_owner_selects`]): nothing projected
/// them this layer.
#[derive(Clone, Copy)]
pub(super) struct GlmOwnerProjections {
    pub(super) keys: DevicePtr,
    pub(super) gates: DevicePtr,
    pub(super) index: Option<(DevicePtr, DevicePtr)>,
    pub(super) q_absorbed: DevicePtr,
    pub(super) key_row: usize,
    pub(super) query_row: usize,
    pub(super) weight_row: usize,
}

impl GlmOwnerProjections {
    /// The index queries and weights of the owner at `row0`. Fails closed when
    /// the batch skipped them: the selector's own projection would land in
    /// the scratch that holds this batch's absorbed queries.
    pub(super) fn index_rows(&self, row0: usize) -> Result<(DevicePtr, DevicePtr)> {
        let (query, weights) = self.index.ok_or_else(|| {
            anyhow::anyhow!("GLM owner batch skipped the index queries of a selecting owner")
        })?;
        Ok((
            query.offset(row0 * self.query_row),
            weights.offset(row0 * self.weight_row),
        ))
    }
}

/// One sequence's prefill chunk of `rows` rows from `seq_len_start` as chunk
/// owners at rows `[0, rows)` of `meta`. Several pieces are consecutive
/// same-sequence owners: the joint cache write lands every row first, and
/// piece k's causal extent is entry 1 + k of the chunk's seq_len buffer
/// (Model::upload_chunk_seq_lens).
pub(in crate::layers::qwen3_attention) fn glm_chunk_pieces(
    meta: crate::layer::AttnMetadataDev,
    seq_len_start: usize,
    rows: usize,
    index_topk: usize,
) -> Vec<GlmChunkOwner> {
    let pieces = crate::layer::prefill_attention_pieces(seq_len_start, rows, index_topk);
    if pieces.len() == 1 {
        return vec![GlmChunkOwner {
            row0: 0,
            rows,
            seq_len_start,
            meta,
        }];
    }
    pieces
        .iter()
        .enumerate()
        .map(|(k, &(row0, rows))| GlmChunkOwner {
            row0,
            rows,
            seq_len_start: seq_len_start + row0,
            meta: crate::layer::AttnMetadataDev {
                positions: meta.positions.offset(row0 * 4),
                positions_h: meta.positions_h.offset(row0 * 4),
                positions_w: meta.positions_w.offset(row0 * 4),
                slot: meta.slot.offset(row0 * 8),
                seq_len: meta.seq_len.offset((1 + k) * 4),
                ..meta
            },
        })
        .collect()
}

/// An `fp8_g128` owner's latents dequantized to BF16 in the arena scratch,
/// addressed through an identity block table of `blocks` entries.
#[derive(Clone, Copy)]
pub(super) struct Bf16LatentView {
    pub(super) latents: DevicePtr,
    pub(super) identity_table: DevicePtr,
    pub(super) blocks: usize,
}

impl Qwen3AttentionLayer {
    /// Owner-batched projections for a verify batch (several owners, few
    /// rows), when the scratch holds every row; `None` projects per owner.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_owner_projections(
        &self,
        owners: &[GlmChunkOwner],
        normed: DevicePtr,
        q_latent: DevicePtr,
        rows: usize,
        nq: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<GlmOwnerProjections>> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let on =
            *ON.get_or_init(|| std::env::var("ATLAS_GLM_OWNER_BATCH_PROJ").as_deref() != Ok("0"));
        let mla = self
            .mla
            .as_ref()
            .expect("GLM owner projections without MLA");
        let c = ctx.config;
        let (hd, kv_lora) = (mla.nope, mla.kv_lora_rank);
        let key_row = c.index_head_dim * 2;
        let query_row = c.index_n_heads * c.index_head_dim * 2;
        let weight_row = c.index_n_heads * 2;
        let latent_row = nq * kv_lora * 2;
        let sizes = ctx.buffers.sizes();
        if !on
            || owners.len() < 2
            || rows > 64
            || sizes.ssm_qkvz < 2 * rows * key_row
            || sizes.ssm_deinterleaved < rows * (latent_row + query_row)
            || sizes.qkv_output < rows * nq * hd * 2
            || sizes.ssm_gates < rows * weight_row
        {
            return Ok(None);
        }
        let n = rows as u32;
        let keys = ctx.buffers.ssm_qkvz();
        let gates = keys.offset(rows * key_row);
        self.glm_index_project_keys(normed, n, keys, gates, ctx, stream)?;
        let q_absorbed = ctx.buffers.ssm_deinterleaved();
        // Short-context batches (every owner dense) skip wq_b and weights_proj.
        let index = if any_owner_selects(owners, c.index_topk) {
            let query = q_absorbed.offset(rows * latent_row);
            let weights = ctx.buffers.ssm_gates();
            self.glm_index_project_query(q_latent, normed, n, query, weights, ctx, stream)?;
            Some((query, weights))
        } else {
            None
        };
        let q_full = ctx.buffers.qkv_output();
        self.paged_glm_projection(
            q_latent,
            &mla.wq_b,
            q_full,
            n,
            (nq * hd) as u32,
            mla.q_lora_rank as u32,
            ctx,
            stream,
            projection::enabled(&c.model_type)?,
        )?;
        self.glm_absorb_queries(q_full, q_absorbed, n, nq as u32, ctx, stream)?;
        Ok(Some(GlmOwnerProjections {
            keys,
            gates,
            index,
            q_absorbed,
            key_row,
            query_row,
            weight_row,
        }))
    }

    /// When `wanted` and the cache is `fp8_g128`, dequantize the owner's
    /// tokens `[0, end)` into the BF16 view — unless they outgrow it, in
    /// which case the caller reads the FP8 cache directly.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_owner_bf16_view(
        &self,
        kv_cache: &PagedKvCache,
        block_table: DevicePtr,
        end: usize,
        wanted: bool,
        block_size: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<Bf16LatentView>> {
        if self.kv_dtype != KvCacheDtype::Fp8G128 || !wanted {
            return Ok(None);
        }
        let (latents, identity_table, capacity) = ctx
            .buffers
            .glm_latent_scratch()
            .ok_or_else(|| anyhow::anyhow!("fp8_g128 GLM cache has no BF16 view scratch"))?;
        if end > capacity {
            return Ok(None);
        }
        ops::glm_latent_dequant_fp8g128(
            ctx.gpu,
            self.glm_latent_dequant_k,
            kv_cache.k_pool_ptr(self.attn_layer_idx),
            block_table,
            latents,
            end as u32,
            block_size,
            stream,
        )?;
        Ok(Some(Bf16LatentView {
            latents,
            identity_table,
            blocks: capacity / 16,
        }))
    }
}
