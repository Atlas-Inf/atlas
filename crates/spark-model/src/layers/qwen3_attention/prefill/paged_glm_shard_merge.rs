// SPDX-License-Identifier: AGPL-3.0-only

//! The launches and exchanges of one merge-form owner over token-sharded
//! latents (`layers::glm_kv_shard`), free of the layer so their order and
//! arguments are testable.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{ExchangeLane, KvCacheDtype, LatentShard};

use super::{ShardRows, owner_rank};
use crate::layers::glm_kv_shard::{self as shard, HEADS, LATENT, MergeLayout, WIDTH, WindowComm};
use crate::layers::ops;

/// A payload the pair swaps: send from `.0`, land the peer's in `.1`.
type Swap = (DevicePtr, DevicePtr, usize);

/// The query swap of a merge-form owner of `rows` rows: this rank's heads'
/// absorbed `query` out, the peer's into the merge scratch at `work`.
fn query_swap(work: DevicePtr, query: DevicePtr, rows: u32) -> Swap {
    let m = MergeLayout::new(rows, shard::MERGE_SPLITS);
    let bytes = rows as usize * (HEADS * LATENT) as usize * 2;
    (query, work.offset(m.q_peer), bytes)
}

/// Swap an owner's queries on `lane` while `during` runs; its merge
/// ([`ShardMerge::run`]) then takes `queries_swapped`.
pub(super) fn swap_queries_during<T>(
    gpu: &dyn GpuBackend,
    comm: &dyn CommBackend,
    lane: ExchangeLane,
    work: DevicePtr,
    (query, rows): (DevicePtr, u32),
    stream: u64,
    during: impl FnOnce(&WindowComm) -> Result<T>,
) -> Result<T> {
    let swap = query_swap(work, query, rows);
    shard::overlapped_exchange(gpu, comm, lane, swap, stream, during)
}

/// One layer's merge form on this rank.
pub(super) struct ShardMerge<'a> {
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
    /// Where the exchanges overlap compute (never under graph capture).
    pub lane: Option<ExchangeLane>,
}

impl ShardMerge<'_> {
    /// This rank's heads' attention over the selected tokens of both ranks
    /// into `output` (`[rows, 32, 512]` BF16). Every launch is one the
    /// canonical form of an unsharded pair runs for the same heads
    /// (`ops::glm_sparse_canonical`), so the bits match it.
    pub(super) fn run(&self, a: ShardRows, output: DevicePtr, stream: u64) -> Result<()> {
        let (gpu, comm, s, lane) = (self.gpu, self.comm, &self.shard, self.lane);
        let splits = shard::MERGE_SPLITS;
        let m = MergeLayout::new(a.rows, splits);
        ensure!(
            m.total <= self.work_bytes,
            "GLM KV shard merge scratch overflow"
        );
        let at = |offset: usize| s.scratch.offset(self.work + offset);
        let part = a.rows as usize * (HEADS * LATENT) as usize * 4;
        // Each row's IDs this rank stores, packed to the front and counted.
        let counts = at(m.counts(a.rows, splits));
        let localize = || {
            ops::glm_kv_shard_localize(
                gpu,
                a.selected,
                at(m.ids),
                counts,
                a.block_table,
                a.rows,
                WIDTH,
                16,
                owner_rank(s),
                a.causal_start,
                stream,
            )
        };

        // 1. Swap queries: the peer's heads attend over this rank's tokens too.
        // 2. Localize the selection (beside the swap when overlapping).
        let queries = query_swap(at(0), a.query, a.rows);
        match lane {
            _ if a.queries_swapped => localize()?,
            Some(lane) => {
                shard::overlapped_exchange(gpu, comm, lane, queries, stream, |_| localize())?
            }
            None => {
                shard::pair_exchange(comm, queries.0, queries.1, queries.2, stream)?;
                localize()?;
            }
        }
        let tc = |query: DevicePtr| ops::GlmSparsePrefillTc {
            config: self.config,
            dtype: self.dtype,
            identical_kv_latent: true,
            query,
            k_cache: self.pool,
            v_cache: self.pool,
            indices: at(m.ids),
            output,
            block_table: s.identity,
            rows: a.rows,
            heads: HEADS,
            head_dim: LATENT,
            index_width: WIDTH,
            block_size: 16,
            scale: self.scale,
        };
        // 3. Both heads' partitions over this rank's tokens; the peer's heads'
        //    merged to one FP32 partial + LSE.
        let send = at(m.send);
        let own = [at(m.own_o), at(m.own_lse)];
        let peer = [at(m.peer_o), at(m.peer_lse)];
        let group = |query: DevicePtr, [part_o, part_lse]: [DevicePtr; 2]| ops::PartialGroup {
            query,
            indices: at(m.ids),
            counts,
            part_o,
            part_lse,
        };
        let partials = |query: DevicePtr, [o, l]: [DevicePtr; 2]| {
            ops::launch_sparse_partials(gpu, &tc(query), splits, Some(counts), o, l, stream)
        };
        // Without an overlap both heads' partitions go in one launch; with
        // one, this rank's heads' partitions run beside the partial swap.
        if lane.is_none() {
            let groups = [group(a.query, own), group(queries.1, peer)];
            ops::launch_sparse_partial_pair(gpu, &tc(a.query), splits, groups, stream)?;
        } else {
            partials(queries.1, peer)?;
        }
        let [po, pl] = peer;
        ops::launch_merge_f32(gpu, po, pl, send, send.offset(part), a.rows, splits, stream)?;
        // 4. Swap the partials.
        let recv = at(m.recv);
        let swap: Swap = (send, recv, MergeLayout::partial_bytes(a.rows));
        match lane {
            Some(lane) => shard::overlapped_exchange(gpu, comm, lane, swap, stream, |_| {
                partials(a.query, own)
            })?,
            None => shard::pair_exchange(comm, swap.0, swap.1, swap.2, stream)?,
        }
        // 5. One LSE merge to BF16 with the peer's partial, where it landed,
        //    as partition `splits`.
        let [own_o, own_lse] = own;
        ops::launch_merge_extra(
            gpu,
            own_o,
            own_lse,
            output,
            at(m.out_lse),
            a.rows,
            splits,
            recv,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "paged_glm_shard_merge_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "paged_glm_shard_merge_gpu_tests.rs"]
mod gpu_tests;
