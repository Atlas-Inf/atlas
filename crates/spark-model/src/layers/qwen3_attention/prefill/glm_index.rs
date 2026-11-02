// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 semantic-index cache population shared by first and later chunks.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::{PagedKvCache, SparseIndexCacheDtype};

use super::super::Qwen3AttentionLayer;
use super::glm_index_split::{self, IndexSplit, OwnerRows};
use crate::layer::ForwardContext;
use crate::layers::ops;

fn profile_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("ATLAS_GLM_INDEX_PROFILE").ok().as_deref() == Some("1"))
}

pub(super) fn profile_start(
    ctx: &ForwardContext,
    stream: u64,
) -> Result<Option<std::time::Instant>> {
    if !profile_enabled() {
        return Ok(None);
    }
    ctx.gpu.synchronize(stream)?;
    Ok(Some(std::time::Instant::now()))
}

pub(super) fn profile_lap(
    ctx: &ForwardContext,
    stream: u64,
    timer: &mut Option<std::time::Instant>,
) -> Result<u128> {
    let Some(started) = timer.take() else {
        return Ok(0);
    };
    ctx.gpu.synchronize(stream)?;
    let elapsed = started.elapsed().as_micros();
    *timer = Some(std::time::Instant::now());
    Ok(elapsed)
}

impl Qwen3AttentionLayer {
    /// Semantic-index keys (projected + layernorm) and pool gates of `n`
    /// normed rows into `keys` / `gates` (`[n, index_head_dim]` BF16 each).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_index_project_keys(
        &self,
        normed: spark_runtime::gpu::DevicePtr,
        n: u32,
        keys: spark_runtime::gpu::DevicePtr,
        gates: spark_runtime::gpu::DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let indexer = self
            .mla
            .as_ref()
            .and_then(|m| m.glm_indexer.as_ref())
            .expect("GLM index keys without indexer weights");
        let h = ctx.config.hidden_size as u32;
        let dim = ctx.config.index_head_dim as u32;
        self.mla_prefill_dense(normed, &indexer.wk, keys, n, dim, h, ctx, stream)?;
        ops::glm_index_layernorm(
            ctx.gpu,
            self.glm_index_layernorm_k,
            keys,
            indexer.k_norm_weight.weight,
            indexer.k_norm_bias.weight,
            n,
            dim,
            1e-6,
            stream,
        )?;
        self.mla_prefill_dense(normed, &indexer.kpool_gate, gates, n, dim, h, ctx, stream)
    }

    /// Semantic-index queries (`[n, heads * dim]`) and per-head weights
    /// (`[n, heads]`) of `n` rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_index_project_query(
        &self,
        q_latent: spark_runtime::gpu::DevicePtr,
        normed: spark_runtime::gpu::DevicePtr,
        n: u32,
        index_query: spark_runtime::gpu::DevicePtr,
        weights: spark_runtime::gpu::DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let mla = self.mla.as_ref().expect("GLM index query without MLA");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("GLM index query without indexer weights");
        let index_heads = ctx.config.index_n_heads as u32;
        self.mla_prefill_dense(
            q_latent,
            &indexer.wq_b,
            index_query,
            n,
            index_heads * ctx.config.index_head_dim as u32,
            mla.q_lora_rank as u32,
            ctx,
            stream,
        )?;
        // BF16 is sufficient for the first functional selector. A later
        // measured refinement will retain this projection's FP32 accumulator,
        // matching upstream's near-tie ranking treatment.
        self.mla_prefill_dense(
            normed,
            &indexer.weights_proj,
            weights,
            n,
            index_heads,
            ctx.config.hidden_size as u32,
            ctx,
            stream,
        )
    }

    /// `projected`: this chunk's keys and gates already produced by
    /// `glm_index_project_keys` (owner-batched verify), else projected here.
    ///
    /// The first `write_skip` rows sit below the KV write floor (positions in
    /// shared prefix-cache blocks): their raw tails and pooled keys stay as
    /// cached. The floor ends on a cached block boundary or covers every row,
    /// so no pool finalized here mixes skipped and written rows.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_index_prefill_cache_update(
        &self,
        normed: spark_runtime::gpu::DevicePtr,
        n: u32,
        write_skip: usize,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        projected: Option<(spark_runtime::gpu::DevicePtr, spark_runtime::gpu::DevicePtr)>,
    ) -> Result<()> {
        let mla = self
            .mla
            .as_ref()
            .expect("GLM index cache update called without MLA weights");
        let indexer = mla
            .glm_indexer
            .as_ref()
            .expect("GLM index cache update called without indexer weights");
        let spec = kv_cache
            .sparse_index_config()
            .ok_or_else(|| anyhow::anyhow!("GLM semantic-index cache is not attached"))?;
        ensure!(
            spec.dtype == SparseIndexCacheDtype::Bf16
                && spec.tokens_per_pool == ctx.config.index_kpool
                && spec.head_dim == ctx.config.index_head_dim,
            "GLM semantic-index cache geometry does not match the checkpoint"
        );
        ensure!(
            self.glm_index_layernorm_k.0 != 0
                && self.glm_index_tail_write_k.0 != 0
                && self.glm_index_kpool_finalize_k.0 != 0,
            "GLM semantic-index kernels are unavailable"
        );

        let skip = write_skip.min(n as usize);
        let rows = n - skip as u32;
        if rows == 0 {
            return Ok(());
        }
        let mut profile = profile_start(ctx, stream)?;

        let dim = spec.head_dim as u32;
        let (keys, gates) = match projected {
            Some(p) => p,
            None => {
                let keys = ctx.buffers.ssm_qkvz();
                let gates = keys.offset(n as usize * spec.head_dim * 2);
                self.glm_index_project_keys(normed, n, keys, gates, ctx, stream)?;
                (keys, gates)
            }
        };
        let projection_us = profile_lap(ctx, stream, &mut profile)?;
        // ATLAS_GLM_DET_TRACE opt-in stages: this piece's raw keys and gates.
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        det.tap("x_ikeys", keys, (0, rows as usize), spec.head_dim * 2);
        det.tap("x_igates", gates, (0, rows as usize), spec.head_dim * 2);
        let meta = ctx
            .attn_metadata
            .expect("GLM index cache update requires slot metadata");
        // Keys and gates are projected for every row; only `[skip, n)` lands.
        let skip_bytes = skip * spec.head_dim * 2;
        // slot_mapping entries are int64 (8 bytes each)
        let slots = meta.slot.offset(skip * 8);
        ops::glm_index_tail_write(
            ctx.gpu,
            self.glm_index_tail_write_k,
            keys.offset(skip_bytes),
            gates.offset(skip_bytes),
            kv_cache.sparse_index_tail_pool_ptr(self.attn_layer_idx),
            kv_cache.sparse_index_tail_map_ptr(),
            slots,
            rows,
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
            slots,
            rows,
            kv_cache.block_size() as u32,
            spec.tokens_per_pool as u32,
            dim,
            kv_cache.sparse_index_tail_block_stride_bytes(self.attn_layer_idx) as u64,
            kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
            stream,
        )?;
        let cache_write_us = profile_lap(ctx, stream, &mut profile)?;
        if profile.is_some() {
            tracing::info!(
                "ATLAS_GLM_INDEX_PROFILE phase=cache layer={} rows={} projection_us={} cache_write_us={}",
                self.attn_layer_idx,
                rows,
                projection_us,
                cache_write_us,
            );
        }
        Ok(())
    }

    /// Project semantic queries and select token-granular sparse history for
    /// every row in this prefill chunk. Logits are processed in bounded row
    /// tiles using the existing MoE activation arena, so scratch does not grow
    /// with the configured model context.
    ///
    /// `projected`: the rows' queries and weights already produced by
    /// `glm_index_project_query` (owner-batched verify), else projected here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_index_prefill_select(
        &self,
        q_latent: DevicePtr,
        normed: DevicePtr,
        n: u32,
        seq_len_start: usize,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
        projected: Option<(DevicePtr, DevicePtr)>,
    ) -> Result<(DevicePtr, u32)> {
        ensure!(
            self.mla.as_ref().is_some_and(|m| m.glm_indexer.is_some()),
            "GLM index selection without indexer weights"
        );
        ensure!(
            self.glm_index_logits_k.0 != 0
                && self.glm_index_topk_expand_k.0 != 0
                && self.glm_sparse_attn_k.0 != 0,
            "GLM sparse selection/attention kernels are unavailable"
        );
        let index_heads = ctx.config.index_n_heads as u32;
        let index_dim = ctx.config.index_head_dim as u32;
        let pool_size = ctx.config.index_kpool as u32;
        let topk = ctx.config.index_topk as u32;
        let output_width = topk + pool_size - 1;
        let sequence_end = seq_len_start + n as usize;
        let logits_stride = sequence_end.div_ceil(pool_size as usize) as u32;

        let mut profile = profile_start(ctx, stream)?;
        let (index_query, weights) = match projected {
            Some(p) => p,
            None => {
                let (q, w) = (ctx.buffers.ssm_deinterleaved(), ctx.buffers.ssm_gates());
                self.glm_index_project_query(q_latent, normed, n, q, w, ctx, stream)?;
                (q, w)
            }
        };

        let logits = ctx.buffers.expert_up_out();
        let tile_rows = super::super::glm_index_capacity::tile_rows(
            n as usize,
            logits_stride as usize,
            ctx.buffers.sizes().expert_up_out,
        )?;
        let selected = ctx.buffers.expert_down_out();
        let query_row_bytes = index_heads as usize * index_dim as usize * 2;
        let weights_row_bytes = index_heads as usize * 2;
        let output_row_bytes = output_width as usize * std::mem::size_of::<i32>();
        // `ATLAS_GLM_INDEX_SPLIT=1`: this rank selects half the rows and swaps
        // them with its peer after the tiles.
        let split = IndexSplit::plan(n as usize, seq_len_start, output_row_bytes, ctx)?;
        let scratch = selected.offset(n as usize * output_row_bytes);
        let passes = glm_index_split::passes(split, n as usize, selected, scratch);
        // Projection launches above are intentionally included in this first
        // lap. They produce the semantic query and per-head weights consumed
        // by every history tile.
        let projection_us = profile_lap(ctx, stream, &mut profile)?;
        // ATLAS_GLM_DET_TRACE opt-in stages: queries, head weights, then each
        // tile's pool logits (row keys are absolute sequence positions).
        let det = crate::det_trace::on_stream(ctx.gpu, stream);
        det.tap(
            "x_iq",
            index_query,
            (seq_len_start, n as usize),
            query_row_bytes,
        );
        det.tap(
            "x_iw",
            weights,
            (seq_len_start, n as usize),
            weights_row_bytes,
        );
        let mut logits_us = 0u128;
        let mut topk_us = 0u128;
        let mut tiles = 0usize;
        for (row_start, rows, out) in glm_index_split::tiles(&passes, tile_rows) {
            let rows = rows as u32;
            ops::glm_index_logits(
                ctx.gpu,
                self.glm_index_logits_k,
                index_query.offset(row_start * query_row_bytes),
                weights.offset(row_start * weights_row_bytes),
                kv_cache.sparse_index_pool_ptr(self.attn_layer_idx),
                logits,
                ctx.attn_metadata
                    .expect("GLM index selection requires block table")
                    .block_table,
                rows,
                (seq_len_start + row_start) as u32,
                logits_stride,
                index_heads,
                index_dim,
                pool_size,
                kv_cache.block_size() as u32,
                kv_cache.sparse_index_block_stride_bytes(self.attn_layer_idx) as u64,
                self.glm_index_logits_rows_per_cta,
                self.glm_index_logits_pools_per_cta,
                stream,
            )?;
            logits_us += profile_lap(ctx, stream, &mut profile)?;
            det.tap(
                "x_ilog",
                logits,
                (seq_len_start + row_start, rows as usize),
                logits_stride as usize * std::mem::size_of::<f32>(),
            );
            ops::glm_index_topk_expand(
                ctx.gpu,
                self.glm_index_topk_expand_k,
                logits,
                out.offset(row_start * output_row_bytes),
                rows,
                (seq_len_start + row_start) as u32,
                logits_stride,
                topk,
                pool_size,
                output_width,
                stream,
            )?;
            topk_us += profile_lap(ctx, stream, &mut profile)?;
            tiles += 1;
        }
        if let Some(split) = split {
            let owner = OwnerRows {
                selected,
                row_bytes: output_row_bytes,
                scratch,
                inputs: [
                    (index_query, n as usize * query_row_bytes),
                    (weights, n as usize * weights_row_bytes),
                ],
            };
            split.exchange(&owner, self.attn_layer_idx, ctx, stream)?;
        }
        let exchange_us = profile_lap(ctx, stream, &mut profile)?;
        if profile.is_some() {
            tracing::info!(
                "ATLAS_GLM_INDEX_PROFILE phase=select layer={} rows={} seq_end={} pools={} tile_rows={} tiles={} split={} projection_us={} logits_us={} topk_us={} exchange_us={}",
                self.attn_layer_idx,
                n,
                sequence_end,
                logits_stride,
                tile_rows,
                tiles,
                split.is_some(),
                projection_us,
                logits_us,
                topk_us,
                exchange_us,
            );
        }
        Ok((selected, output_width))
    }
}
