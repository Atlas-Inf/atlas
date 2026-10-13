// SPDX-License-Identifier: AGPL-3.0-only

//! GLM MLA attention over token-sharded latents (`ATLAS_GLM_KV_SHARD=1`, see
//! `layers::glm_kv_shard`): the merge form for few-row owners, the view form
//! for prefill chunks, and the owner-only cache-write slots.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{ExchangeLane, KvCacheDtype, LatentShard, PagedKvCache};

use super::super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::glm_kv_shard::{self as shard, LATENT, ScratchLayout, WIDTH};
use crate::layers::ops;

/// One merge-form owner: `rows` query rows of one sequence.
#[derive(Clone, Copy)]
pub(in crate::layers::qwen3_attention) struct ShardRows {
    /// This rank's heads' absorbed queries `[rows, 32, 512]` BF16.
    pub query: DevicePtr,
    /// Selected IDs `[rows, 2051]`; `None` attends causally, row `r` to
    /// tokens `[0, causal_start + r + 1)`.
    pub selected: Option<DevicePtr>,
    pub causal_start: u32,
    /// The sequence's block table (logical → physical).
    pub block_table: DevicePtr,
    pub rows: u32,
    /// The last row's causal extent when the host knows it (table check).
    pub end: Option<usize>,
    /// The peer's queries are already in the merge scratch
    /// ([`Qwen3AttentionLayer::glm_shard_swap_queries_during`]).
    pub queries_swapped: bool,
}

#[path = "paged_glm_shard_merge.rs"]
mod merge;

fn owner_rank(s: &LatentShard) -> ops::ShardRank {
    ops::ShardRank {
        rank: s.spec.rank as u32,
        world: s.spec.world as u32,
    }
}

/// The shard, the pair communicator of this rank and the scratch layout.
fn pair<'a>(
    kv_cache: &PagedKvCache,
    ctx: &ForwardContext<'a>,
) -> Result<(LatentShard, &'a dyn CommBackend, ScratchLayout)> {
    let s = kv_cache
        .latent_shard()
        .context("GLM KV shard attention on an unsharded cache")?;
    let comm = ctx
        .comm
        .filter(|c| c.world_size() == 2 && c.rank() == s.spec.rank)
        .context("GLM KV shard attention needs this rank's TP-pair communicator")?;
    Ok((s, comm, ScratchLayout::of(&s, kv_cache.config())?))
}

/// Logical blocks `[0, n)` of a device block table, read on `stream`.
fn read_table(gpu: &dyn GpuBackend, table: DevicePtr, n: usize, stream: u64) -> Result<Vec<u32>> {
    let mut raw = vec![0u8; n * 4];
    gpu.copy_d2h_on_stream(table, &mut raw, stream)?;
    Ok(raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn u32_bytes(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

impl Qwen3AttentionLayer {
    fn glm_shard_geometry(&self, kv_cache: &PagedKvCache) -> Result<()> {
        ensure!(
            matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
                && kv_cache.block_size() == 16
                && self.mla.as_ref().is_some_and(|m| {
                    m.kv_lora_rank == LATENT as usize && m.rope == 0 && m.glm_indexer.is_some()
                }),
            "GLM KV shard needs the NoPE-512 semantic-index MLA on a 16-token BF16/fp8_g128 cache"
        );
        Ok(())
    }

    /// This rank's table must place every logical block on the rank its index
    /// names, and both ranks must see the same number of blocks (their
    /// physical ids legitimately differ). The ranks swap verdicts before
    /// either fails: a rank failing alone would leave its peer waiting in the
    /// exchange that follows.
    fn glm_shard_check(
        &self,
        s: &LatentShard,
        comm: &dyn CommBackend,
        gpu: &dyn GpuBackend,
        words: DevicePtr,
        table: &[u32],
        stream: u64,
    ) -> Result<()> {
        const BROKEN: u64 = 1 << 63;
        let verdict = s.check_table(table);
        let ours = table.len() as u64 | if verdict.is_err() { BROKEN } else { 0 };
        gpu.copy_h2d_async(&ours.to_le_bytes(), words, stream)?;
        shard::pair_exchange(comm, words, words.offset(8), 8, stream)?;
        let mut peer = [0u8; 8];
        gpu.copy_d2h_on_stream(words.offset(8), &mut peer, stream)?;
        let peer = u64::from_le_bytes(peer);
        verdict.with_context(|| format!("attention layer {}", self.attn_layer_idx))?;
        ensure!(
            peer == ours,
            "GLM KV shard: rank {} attends {} blocks, its peer {}{}, at attention layer {}",
            s.spec.rank,
            ours,
            peer & !BROKEN,
            if peer & BROKEN == 0 {
                ""
            } else {
                " of a table that breaks the ownership rule"
            },
            self.attn_layer_idx
        );
        Ok(())
    }

    /// Where exchanges of this cache's merge-form owners overlap compute:
    /// the lane of a cache sharded under `ATLAS_GLM_KV_SHARD_OVERLAP=1`,
    /// outside graph capture.
    pub(super) fn glm_shard_lane(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
    ) -> Option<ExchangeLane> {
        kv_cache
            .latent_shard()
            .and_then(|s| s.lane)
            .filter(|_| !ctx.graph_capture)
    }

    /// `[k, v, block table]` an owner's kernels read: the layer's pools and
    /// the sequence's table, or under a shard its assembled history (the view
    /// form). `None` for a sharded owner of few rows, which merges each
    /// rank's attention over its own tokens ([`Self::glm_shard_merge_owner`]).
    pub(super) fn glm_owner_latents(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        o: &super::GlmChunkOwner,
        end: usize,
        stream: u64,
    ) -> Result<Option<[DevicePtr; 3]>> {
        if kv_cache.latent_shard().is_none() {
            return Ok(Some([
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                o.meta.block_table,
            ]));
        }
        if o.rows <= shard::MERGE_MAX_ROWS {
            return Ok(None);
        }
        let (view, identity) =
            self.glm_shard_assemble_view(kv_cache, ctx, o.meta.block_table, end, stream)?;
        Ok(Some([view, view, identity]))
    }

    /// `ATLAS_GLM_KV_SHARD_OVERLAP=1`: swap a merge-form owner's absorbed
    /// `query` rows, when already projected (owner-batched), with the peer
    /// while `during` (its index update and selection) runs on a context whose
    /// communicator refuses the pair; otherwise just run `during` on `ctx`.
    /// Returns whether the peer's queries are in the merge scratch.
    pub(super) fn glm_shard_swap_queries_during<T>(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        rows: usize,
        query: Option<DevicePtr>,
        stream: u64,
        during: impl FnOnce(&ForwardContext) -> Result<T>,
    ) -> Result<(T, bool)> {
        let lane = self.glm_shard_lane(kv_cache, ctx);
        let (Some(lane), Some(query)) = (lane.filter(|_| rows <= shard::MERGE_MAX_ROWS), query)
        else {
            return Ok((during(ctx)?, false));
        };
        let (s, comm, layout) = pair(kv_cache, ctx)?;
        let work = s.scratch.offset(layout.work);
        let owner = (query, rows as u32);
        let out = merge::swap_queries_during(ctx.gpu, comm, lane, work, owner, stream, |fenced| {
            during(&ForwardContext {
                comm: Some(fenced),
                midchunk_capture: None, // as in every owner's context
                ..*ctx
            })
        })?;
        Ok((out, true))
    }

    /// Merge form: this rank's heads' attention over the selected tokens of
    /// BOTH ranks into `output` (`[rows, 32, 512]` BF16). Each rank attends
    /// every head over the tokens it stores; the peer's heads' partial is
    /// merged to one FP32 partial and swapped, then joins this rank's own
    /// partitions in one exact LSE merge.
    pub(in crate::layers::qwen3_attention) fn glm_shard_merge_attention(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        a: ShardRows,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        self.glm_shard_geometry(kv_cache)?;
        let (s, comm, layout) = pair(kv_cache, ctx)?;
        ensure!(
            (1..=shard::MERGE_MAX_ROWS as u32).contains(&a.rows),
            "GLM KV shard merge form takes 1..={} rows, got {}",
            shard::MERGE_MAX_ROWS,
            a.rows
        );
        let gpu = ctx.gpu;
        let tuning = shard::MergeTuning::get()?;
        if tuning.check
            && let Some(end) = a.end
        {
            let table = read_table(gpu, a.block_table, end.div_ceil(16), stream)?;
            let words = s.scratch.offset(layout.check);
            self.glm_shard_check(&s, comm, gpu, words, &table, stream)?;
        }
        let hd = self.mla.as_ref().map_or(0, |m| m.nope) as u32;
        merge::ShardMerge {
            gpu,
            comm,
            config: ctx.config,
            shard: s,
            work: layout.work,
            work_bytes: layout.total - layout.work,
            dtype: self.kv_dtype,
            pool: kv_cache.latent_pool_ptr(self.attn_layer_idx),
            scale: self.effective_attn_scale(hd),
            compact: tuning.compact,
            lane: self.glm_shard_lane(kv_cache, ctx),
        }
        .run(a, output, stream)
    }

    /// [`Self::glm_shard_merge_attention`] for one few-row prefill or verify
    /// owner: `selected` is its `(IDs, width)` selection, `None` when the
    /// whole causal history is selected.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_shard_merge_owner(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        o: &super::GlmChunkOwner,
        (query, queries_swapped): (DevicePtr, bool),
        selected: Option<(DevicePtr, u32)>,
        output: DevicePtr,
        stream: u64,
    ) -> Result<()> {
        if let Some((ids, width)) = selected {
            ensure!(
                width == WIDTH,
                "GLM KV shard expects {WIDTH} selected IDs, got {width}"
            );
            // ATLAS_GLM_DET_TRACE: the piece's selected token ids, as unsharded.
            let det = crate::det_trace::on_stream(ctx.gpu, stream);
            det.tap("sel", ids, (o.row0, o.rows), width as usize * 4);
        }
        let rows = ShardRows {
            query,
            selected: selected.map(|(ids, _)| ids),
            causal_start: o.seq_len_start as u32,
            block_table: o.meta.block_table,
            rows: o.rows as u32,
            end: Some(o.seq_len_start + o.rows),
            queries_swapped,
        };
        self.glm_shard_merge_attention(kv_cache, ctx, rows, output, stream)
    }

    /// View form: assemble the sequence's latents for logical tokens
    /// `[0, end)` — this rank's blocks copied locally, the peer's exchanged
    /// in pieces — and return `(view, identity table)` for the unchanged
    /// kernels, which read the view in this layer's cache dtype.
    pub(in crate::layers::qwen3_attention) fn glm_shard_assemble_view(
        &self,
        kv_cache: &PagedKvCache,
        ctx: &ForwardContext,
        block_table: DevicePtr,
        end: usize,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        self.glm_shard_geometry(kv_cache)?;
        let (s, comm, layout) = pair(kv_cache, ctx)?;
        let blocks = end.div_ceil(kv_cache.block_size());
        ensure!(
            blocks > 0 && blocks <= s.spec.view_blocks,
            "GLM KV shard view holds {} blocks, the owner needs {blocks}",
            s.spec.view_blocks
        );
        let gpu = ctx.gpu;
        let table = read_table(gpu, block_table, blocks, stream)?;
        // A table that breaks the ownership rule would pair the wrong blocks
        // across the ranks: both fail here, before the first piece.
        self.glm_shard_check(
            &s,
            comm,
            gpu,
            s.scratch.offset(layout.check),
            &table,
            stream,
        )?;
        let plan = s.plan(&table)?;
        let at = |offset: usize| s.scratch.offset(offset);
        let (mine_slot, mine_dst, peer_dst) = (
            at(layout.mine_slot),
            at(layout.mine_dst),
            at(layout.peer_dst),
        );
        gpu.copy_h2d_async(&u32_bytes(&plan.mine_slot), mine_slot, stream)?;
        gpu.copy_h2d_async(&u32_bytes(&plan.mine_logical), mine_dst, stream)?;
        gpu.copy_h2d_async(&u32_bytes(&plan.peer_logical), peer_dst, stream)?;
        let pool = kv_cache.latent_pool_ptr(self.attn_layer_idx);
        let bytes = kv_cache
            .config()
            .k_block_bytes_for_layer(self.attn_layer_idx);
        ensure!(bytes <= layout.block_bytes, "GLM KV shard view stride");
        let view = at(layout.view);
        let (mine, peer) = (plan.mine_slot.len(), plan.peer_logical.len());
        ops::glm_kv_shard_copy_blocks(
            gpu,
            pool,
            Some(mine_slot),
            view,
            Some(mine_dst),
            mine,
            bytes,
            stream,
        )?;
        // Both ranks derive the same rounds from the same table.
        let rounds = mine.max(peer);
        let send = at(layout.work);
        let recv = send.offset(layout.piece_bytes.next_multiple_of(256));
        let mut start = 0usize;
        while start < rounds {
            let n = shard::PIECE_BLOCKS.min(rounds - start);
            let out_n = mine.saturating_sub(start).min(n);
            let in_n = peer.saturating_sub(start).min(n);
            let from = Some(mine_slot.offset(start * 4));
            ops::glm_kv_shard_copy_blocks(gpu, pool, from, send, None, out_n, bytes, stream)?;
            shard::pair_exchange(comm, send, recv, n * bytes, stream)?;
            let to = Some(peer_dst.offset(start * 4));
            ops::glm_kv_shard_copy_blocks(gpu, recv, None, view, to, in_n, bytes, stream)?;
            start += n;
        }
        Ok((view, s.identity))
    }

    /// This rank's local write slots for `num_tokens` global `slot`s (`-1`
    /// where the peer stores the block).
    pub(in crate::layers::qwen3_attention) fn glm_shard_local_slots(
        &self,
        kv_cache: &PagedKvCache,
        gpu: &dyn GpuBackend,
        slot: DevicePtr,
        num_tokens: u32,
        stream: u64,
    ) -> Result<DevicePtr> {
        let s = kv_cache
            .latent_shard()
            .context("GLM KV shard write on an unsharded cache")?;
        let layout = ScratchLayout::of(&s, kv_cache.config())?;
        ensure!(
            num_tokens as usize <= s.spec.write_rows,
            "GLM KV shard write of {num_tokens} rows exceeds {}",
            s.spec.write_rows
        );
        let local = s.scratch.offset(layout.slots);
        let bs = kv_cache.block_size() as u32;
        ops::glm_kv_shard_map_slots(gpu, slot, local, num_tokens, bs, owner_rank(&s), stream)?;
        Ok(local)
    }
}
