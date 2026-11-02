// SPDX-License-Identifier: AGPL-3.0-only

//! QSA prefill selection split across the SP pair
//! (`ATLAS_QWEN4EXP_PREFILL_QSA_SPLIT=1`, with `ATLAS_QWEN4EXP_PREFILL_SP=1`).
//!
//! Both ranks attend every selective row (their own heads), so both used to
//! compute every row's block list: the indexer's q projection, q prep, block
//! scores and top-k -- ~19 ms a layer at a 16K chunk on GB10, the same work
//! twice. Here rank r computes the lists of slabs r, r + 2, r + 4, ... (the
//! scorer's cost grows with position, so alternating slabs balances the
//! pair), the ranks swap them slab for slab on the side stream, and each rank
//! attends its own slabs while the peer's arrive.
//!
//! Exact by construction: a slab's list is computed by the same launches on
//! the same rows with the same slab boundaries on either rank (the slabs are
//! the unsplit loop's), so the received lists are the bytes this rank would
//! have computed; attention per slab is unchanged, in any slab order. Each
//! exchange moves one whole list slot both ways (the pair's exchanges need
//! equal sizes), the pair after an odd last slab moving a spare slot.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::cell::Cell;

use super::qsa_select::SlabBufs;
use super::{QsaIndexer, QsaSeqState};
use crate::layer::ForwardContext;
use crate::layers::qwen4exp_sp_pipe::SideExchanges;
use spark_runtime::buffers::QSA_SELECT_SCRATCH_ROWS as QSA_SLAB;

/// `ATLAS_QWEN4EXP_PREFILL_QSA_SPLIT=1`.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_QSA_SPLIT").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

thread_local! {
    /// `(ptr, bytes)` of this thread's list slots.
    static SLOTS: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
}

/// The slabs `rank` computes, and the pair exchanges as (slot this rank
/// sends, slot it receives into) -- slot `ns` is the spare.
pub(super) fn split_plan(ns: usize, rank: usize) -> (Vec<usize>, Vec<(usize, usize)>) {
    let own = (rank..ns).step_by(2).collect();
    let pairs = (0..ns.div_ceil(2))
        .map(|p| {
            let (mine, peer) = (2 * p + rank, 2 * p + 1 - rank);
            (mine.min(ns), peer.min(ns))
        })
        .collect();
    (own, pairs)
}

impl QsaIndexer {
    /// Run the slab loop split across the pair, or `Ok(false)` (nothing
    /// launched) when not requested or not a two-rank SP chunk (`sp_ctx`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_select_split(
        &self,
        st: &QsaSeqState,
        bufs: &SlabBufs,
        slabs: &[(usize, usize)],
        sp_ctx: Option<&ForwardContext>,
        gpu: &dyn GpuBackend,
        stream: u64,
        attend: &dyn Fn(usize, usize, DevicePtr) -> Result<()>,
    ) -> Result<bool> {
        let max_rows = slabs.iter().map(|&(_, rows)| rows).max().unwrap_or(0);
        let slot = max_rows * self.block_topk as usize * 4;
        let Some(ctx) = sp_ctx.filter(|ctx| {
            requested()
                && slabs.len() >= 2
                && ctx.config.model_type == "qwen4_exp"
                && !ctx.graph_capture
                && ctx
                    .comm
                    .is_some_and(|c| c.world_size() == 2 && c.supports_exchange_async(slot))
        }) else {
            return Ok(false);
        };
        let Some(wire) = SideExchanges::new(ctx)? else {
            return Ok(false);
        };
        let rank = ctx.comm.expect("checked").rank();
        let ns = slabs.len();
        let most = crate::layers::qwen4exp_sp_pipe::prealloc_rows(ctx).div_ceil(QSA_SLAB) + 2;
        let base = slots(gpu, (ns + 1).max(most) * slot, stream)?;
        let at = |i: usize| base.offset(i * slot);
        let (own, pairs) = split_plan(ns, rank);
        // Each pair's swap goes out as soon as this rank's half of it is set.
        for &(mine, peer) in &pairs {
            if let Some(&(first_pos, rows)) = slabs.get(mine) {
                self.prefill_slab_lists(st, bufs, first_pos, rows, at(mine), gpu, stream)?;
            }
            wire.send(at(mine), at(peer), slot, stream)?;
        }
        for &i in &own {
            attend(slabs[i].0, slabs[i].1, at(i))?;
        }
        wire.join(stream)?;
        for i in (1 - rank..ns).step_by(2) {
            attend(slabs[i].0, slabs[i].1, at(i))?;
        }
        Ok(true)
    }
}

/// This thread's list slots, at least `bytes`, grown once the stream drains
/// (every swap into them was joined into it).
fn slots(gpu: &dyn GpuBackend, bytes: usize, stream: u64) -> Result<DevicePtr> {
    let (ptr, size) = SLOTS.with(Cell::get);
    if size >= bytes {
        return Ok(DevicePtr(ptr));
    }
    if ptr != 0 {
        gpu.synchronize(stream)?;
        gpu.free(DevicePtr(ptr))?;
    }
    let p = gpu.alloc(bytes)?;
    SLOTS.with(|s| s.set((p.0, bytes)));
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::split_plan;

    /// Every slab computed by exactly one rank; each pair's sends land in
    /// the peer's matching receive; the receiver gets every peer slab.
    #[test]
    fn the_pair_covers_every_slab_once() {
        for ns in 2..12 {
            let (o0, p0) = split_plan(ns, 0);
            let (o1, p1) = split_plan(ns, 1);
            let mut all: Vec<usize> = o0.iter().chain(&o1).copied().collect();
            all.sort();
            assert_eq!(all, (0..ns).collect::<Vec<_>>(), "ns={ns}");
            assert_eq!(p0.len(), p1.len());
            for (&(s0, r0), &(s1, r1)) in p0.iter().zip(&p1) {
                assert_eq!(
                    (s0, r0),
                    (r1, s1),
                    "ns={ns}: sends land in the peer's receive"
                );
            }
            let got0: Vec<usize> = p0.iter().map(|&(_, r)| r).filter(|&r| r < ns).collect();
            assert_eq!(got0, o1, "ns={ns}: rank 0 receives every rank-1 slab");
            let got1: Vec<usize> = p1.iter().map(|&(_, r)| r).filter(|&r| r < ns).collect();
            assert_eq!(got1, o0, "ns={ns}: rank 1 receives every rank-0 slab");
        }
    }
}

#[cfg(test)]
#[path = "qsa_select_sp_gpu_tests.rs"]
mod gpu_tests;
