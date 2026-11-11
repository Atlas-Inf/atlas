// SPDX-License-Identifier: AGPL-3.0-only

//! Token-sharded latents beyond the pair (`ATLAS_GLM_KV_SHARD=1` on the TP4
//! chain): logical block `l` on rank `l % world`, every rank `64 / world`
//! heads. The pair's swaps become the chain's all-gathers
//! (`layers::glm_kv_shard::gather`):
//!
//! - **merge form**: the ranks gather their heads' absorbed queries; each
//!   attends all 64 heads, as the pair's two 32-head groups in one paired
//!   launch, over the tokens it stores, and merges each group's partitions to
//!   one FP32 partial + LSE; the ranks gather those, and each merges its own
//!   heads' partial from every rank in one exact LSE merge. A rank storing
//!   none of a row's tokens contributes `LSE = -inf`, weight 0.
//! - **view form**: the ranks gather their blocks of the history in logical
//!   order; rank `o`'s `k`-th is logical block `o + k * world`.
//!
//! The pair keeps its own exchanges (`paged_glm_shard_merge.rs`) and bits.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, LatentShard};

use super::{ShardRows, owner_rank};
use crate::layers::glm_kv_shard::{
    self as shard, ChainLayout, HEADS, LATENT, ScratchLayout, WIDTH, head_slot, local_heads,
};
use crate::layers::ops;

/// One layer's merge form on this rank of a chain.
pub(super) struct ChainMerge<'a> {
    pub gpu: &'a dyn GpuBackend,
    pub comm: &'a dyn CommBackend,
    pub config: &'a ModelConfig,
    pub shard: LatentShard,
    /// The merge work region of the shard scratch: offset and bytes.
    pub work: usize,
    pub work_bytes: usize,
    pub dtype: KvCacheDtype,
    /// This rank's latent pool of the layer.
    pub pool: DevicePtr,
    pub scale: f32,
}

impl ChainMerge<'_> {
    /// This rank's heads' attention over the selected tokens of every rank
    /// into `output` (`[rows, 64 / world, 512]` BF16).
    pub(super) fn run(&self, a: ShardRows, output: DevicePtr, stream: u64) -> Result<()> {
        let (gpu, comm, s) = (self.gpu, self.comm, &self.shard);
        let (world, rank) = (s.spec.world, s.spec.rank);
        ensure!(
            !a.queries_swapped,
            "GLM KV shard: query overlap is the pair's"
        );
        let splits = shard::MERGE_SPLITS;
        let m = ChainLayout::new(a.rows, world);
        ensure!(
            m.total <= self.work_bytes,
            "GLM KV shard merge scratch overflow"
        );
        let at = |offset: usize| s.scratch.offset(self.work + offset);
        let rows = a.rows as usize;
        let local = local_heads(world) as usize;
        let latent = LATENT as usize;

        // 1. Every rank's queries, in place, then regrouped into the two
        //    32-head groups the paired launch takes (heads in order).
        let q_bytes = rows * local * latent * 2;
        let q_all = at(m.q_all);
        gpu.copy_d2d_async(a.query, q_all.offset(rank * m.q_part), q_bytes, stream)?;
        shard::gather(comm, q_all, m.q_part, q_bytes, stream)?;
        let group_q = rows * (HEADS * LATENT) as usize * 2;
        for origin in 0..world {
            let (g, h0) = head_slot(origin, world);
            gpu.copy_d2d_2d_async(
                q_all.offset(origin * m.q_part),
                local * latent * 2,
                at(m.q_groups + g * group_q + h0 * latent * 2),
                HEADS as usize * latent * 2,
                local * latent * 2,
                rows,
                stream,
            )?;
        }
        // 2. Each row's IDs this rank stores, packed to the front and counted.
        let (ids, counts) = (at(m.ids), at(m.counts));
        if !a.localized {
            ops::glm_kv_shard_localize(
                gpu,
                a.selected,
                ids,
                counts,
                a.block_table,
                a.rows,
                WIDTH,
                16,
                owner_rank(s),
                a.causal_start,
                stream,
            )?;
        }
        // 3. Both groups' partitions over this rank's tokens, one launch.
        let group_o = rows * (HEADS * LATENT) as usize * 4;
        let group_lse = rows * HEADS as usize * 4;
        let parts = |g: usize| {
            [
                at(m.part_o + g * splits as usize * group_o),
                at(m.part_lse + g * splits as usize * group_lse),
            ]
        };
        let group = |g: usize| {
            let [part_o, part_lse] = parts(g);
            ops::PartialGroup {
                query: at(m.q_groups + g * group_q),
                indices: ids,
                counts,
                part_o,
                part_lse,
            }
        };
        let tc = ops::GlmSparsePrefillTc {
            config: self.config,
            dtype: self.dtype,
            identical_kv_latent: true,
            query: at(m.q_groups),
            k_cache: self.pool,
            v_cache: self.pool,
            indices: ids,
            output,
            block_table: s.identity,
            rows: a.rows,
            heads: HEADS,
            head_dim: LATENT,
            index_width: WIDTH,
            block_size: 16,
            scale: self.scale,
        };
        ops::launch_sparse_partial_pair(gpu, &tc, splits, [group(0), group(1)], stream)?;
        // 4. Each group merged to one FP32 partial + LSE in this rank's part,
        //    then every rank's part gathered.
        let gathered = at(m.gathered);
        let mine = gathered.offset(rank * m.part);
        for g in 0..2 {
            let [po, pl] = parts(g);
            let (o, l) = (
                mine.offset(ChainLayout::group_o(a.rows, g)),
                mine.offset(ChainLayout::group_lse(a.rows, g)),
            );
            ops::launch_merge_f32(gpu, po, pl, o, l, a.rows, splits, stream)?;
        }
        shard::gather(comm, gathered, m.part, m.part_bytes, stream)?;
        // 5. This rank's heads' partial from each rank as the partitions of
        //    one exact LSE merge to BF16.
        let (g, h0) = head_slot(rank, world);
        for origin in 0..world {
            let part = gathered.offset(origin * m.part);
            gpu.copy_d2d_2d_async(
                part.offset(ChainLayout::group_o(a.rows, g) + h0 * latent * 4),
                HEADS as usize * latent * 4,
                at(m.mine_o + origin * rows * local * latent * 4),
                local * latent * 4,
                local * latent * 4,
                rows,
                stream,
            )?;
            gpu.copy_d2d_2d_async(
                part.offset(ChainLayout::group_lse(a.rows, g) + h0 * 4),
                HEADS as usize * 4,
                at(m.mine_lse + origin * rows * local * 4),
                local * 4,
                local * 4,
                rows,
                stream,
            )?;
        }
        ops::launch_merge(
            gpu,
            ops::merge_kernel(gpu)?,
            at(m.mine_o),
            at(m.mine_lse),
            output,
            at(m.out_lse),
            a.rows,
            local as u32,
            world as u32,
            stream,
        )
    }
}

/// One view-form assembly beyond the pair: this rank's blocks of logical
/// `[0, blocks)` are already in `view`; the other ranks' arrive by gather.
pub(super) struct ViewParts {
    /// This rank's latent pool of the layer and the view it fills.
    pub pool: DevicePtr,
    pub view: DevicePtr,
    /// Device `u32` local slots of this rank's blocks, in logical order.
    pub mine_slot: DevicePtr,
    pub blocks: usize,
    /// The layer's latent block bytes (the view's stride).
    pub block_bytes: usize,
}

impl ViewParts {
    /// Logical blocks of `[0, blocks)` rank `origin` stores.
    fn count(&self, origin: usize, world: usize) -> usize {
        (self.blocks + world - 1 - origin) / world
    }

    /// Gather every rank's blocks in rounds of `piece_blocks(world)` per rank:
    /// each packs its next blocks in logical order into its part of the work
    /// region, and rank `o`'s `k`-th lands at view block `o + k * world`.
    pub(super) fn gather(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        s: &LatentShard,
        layout: &ScratchLayout,
        stream: u64,
    ) -> Result<()> {
        let (world, rank) = (s.spec.world, s.spec.rank);
        let bytes = self.block_bytes;
        let per = shard::piece_blocks(world);
        let stride = layout.piece_bytes.next_multiple_of(256);
        let work = s.scratch.offset(layout.work);
        // Rank 0 stores the most blocks; every rank runs its rounds.
        let rounds = self.count(0, world);
        let mut start = 0usize;
        while start < rounds {
            let n = per.min(rounds - start);
            let out_n = self.count(rank, world).saturating_sub(start).min(n);
            let from = Some(self.mine_slot.offset(start * 4));
            let own = work.offset(rank * stride);
            ops::glm_kv_shard_copy_blocks(gpu, self.pool, from, own, None, out_n, bytes, stream)?;
            shard::gather(comm, work, stride, n * bytes, stream)?;
            for origin in (0..world).filter(|&o| o != rank) {
                let in_n = self.count(origin, world).saturating_sub(start).min(n);
                if in_n == 0 {
                    continue;
                }
                gpu.copy_d2d_2d_async(
                    work.offset(origin * stride),
                    bytes,
                    self.view.offset((start * world + origin) * bytes),
                    world * bytes,
                    bytes,
                    in_n,
                    stream,
                )?;
            }
            start += n;
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "paged_glm_shard_chain_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "paged_glm_shard_chain_gpu_tests.rs"]
mod gpu_tests;
