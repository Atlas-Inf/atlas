// SPDX-License-Identifier: AGPL-3.0-only

//! The latent shard beyond the pair: the chain's merge layout and gathers.

use super::*;

/// Offsets (from the work region) of one merge-form owner's buffers beyond
/// the pair (`ChainMerge`): every rank attends all [`TOTAL_HEADS`] over the
/// tokens it stores, as two [`HEADS`]-head groups in one paired launch, and
/// the ranks all-gather queries and merged partials instead of swapping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChainLayout {
    /// Gathered queries: rank `o`'s `[rows, local heads, 512]` BF16 at
    /// `q_all + o * q_part`.
    pub q_all: usize,
    pub q_part: usize,
    /// The queries regrouped `[2][rows, 32, 512]` BF16 (heads in order).
    pub q_groups: usize,
    /// Localized IDs `[rows, 2051]` i32 and their per-row counts `u32[rows]`.
    pub ids: usize,
    pub counts: usize,
    /// Both groups' `splits` FP32 partials `[2][splits][rows, 32, 512]` and
    /// LSEs `[2][splits][rows, 32]`.
    pub part_o: usize,
    pub part_lse: usize,
    /// Gathered merged partials: rank `o`'s at `gathered + o * part`, each
    /// group's `[rows, 32, 512]` FP32 then its `[rows, 32]` LSE
    /// ([`Self::group_o`], [`Self::group_lse`]).
    pub gathered: usize,
    pub part: usize,
    /// Bytes of one rank's merged partials (the gathered part's payload).
    pub part_bytes: usize,
    /// This rank's heads' partial from each rank `[world][rows, local, 512]`
    /// FP32 and LSE `[world][rows, local]`, the final merge's partitions.
    pub mine_o: usize,
    pub mine_lse: usize,
    /// The final merge's LSE `[rows, local]`.
    pub out_lse: usize,
    pub total: usize,
}

impl ChainLayout {
    pub fn new(rows: u32, world: usize) -> Self {
        let rows = rows as usize;
        let local = local_heads(world) as usize;
        let latent = LATENT as usize;
        let splits = MERGE_SPLITS as usize;
        let group_o = rows * HEADS as usize * latent * 4;
        let group_lse = rows * HEADS as usize * 4;
        let q_part = align(rows * local * latent * 2);
        let part_bytes = align(group_o) + align(group_lse) + align(group_o) + group_lse;
        let part = align(part_bytes);
        let mut at = 0usize;
        let mut take = |bytes: usize| {
            let offset = at;
            at += align(bytes);
            offset
        };
        let q_all = take(world * q_part);
        let q_groups = take(2 * rows * HEADS as usize * latent * 2);
        let ids = take(rows * WIDTH as usize * 4);
        let counts = take(rows * 4);
        let part_o = take(2 * splits * group_o);
        let part_lse = take(2 * splits * group_lse);
        let gathered = take(world * part);
        let mine_o = take(world * rows * local * latent * 4);
        let mine_lse = take(world * rows * local * 4);
        let out_lse = take(rows * local * 4);
        Self {
            q_all,
            q_part,
            q_groups,
            ids,
            counts,
            part_o,
            part_lse,
            gathered,
            part,
            part_bytes,
            mine_o,
            mine_lse,
            out_lse,
            total: at,
        }
    }

    /// Offset within a rank's gathered part of group `g`'s FP32 partial.
    pub fn group_o(rows: u32, g: usize) -> usize {
        let group_o = rows as usize * (HEADS * LATENT) as usize * 4;
        let group_lse = rows as usize * HEADS as usize * 4;
        g * (align(group_o) + align(group_lse))
    }

    /// Offset within a rank's gathered part of group `g`'s LSEs.
    pub fn group_lse(rows: u32, g: usize) -> usize {
        Self::group_o(rows, g) + align(rows as usize * (HEADS * LATENT) as usize * 4)
    }
}

/// Where rank `rank`'s heads sit among the two [`HEADS`]-head groups:
/// `(group, first head in it)`.
pub fn head_slot(rank: usize, world: usize) -> (usize, usize) {
    let first = rank * local_heads(world) as usize;
    (first / HEADS as usize, first % HEADS as usize)
}

/// The chain's in-place all-gather on `stream`: rank `r`'s `bytes` at
/// `base + r * part` land at the same place on every rank. Every rank calls
/// it in the same order with the same sizes.
pub fn gather(
    comm: &dyn spark_comm::CommBackend,
    base: DevicePtr,
    part: usize,
    bytes: usize,
    stream: u64,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    ensure!(
        comm.gather_async(base.0, part as i64, bytes, stream)?,
        "GLM KV shard: no all-gather for {bytes}-byte parts over {} ranks \
         (the RDMA chain serves it: ATLAS_RDMA_PAIR_CHAIN=1)",
        comm.world_size()
    );
    Ok(())
}
