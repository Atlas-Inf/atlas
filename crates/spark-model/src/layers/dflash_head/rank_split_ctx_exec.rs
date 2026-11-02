// SPDX-License-Identifier: AGPL-3.0-only

//! The rank-split context append at run time (`rank_split_ctx`): the rows
//! the head announces, its turn at each append, and the pieces both ranks
//! run.

use anyhow::{Result, bail};
use spark_comm::CommBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::atomic::{AtomicBool, Ordering};

use super::super::super::{BlockDiffusionDraftHead, DflashProposerState, DflashScratch};
use super::super::ctx::{CTX_MAX_ROWS, CtxRows, ctx_steps};
use super::super::{Frame, Geometry, Graphs, Piece, SplitOps, Step, Swap, walk};
use super::RankSplit;
use crate::layers::ops;
use crate::speculative::ProposerState;

/// One sequence of a propose about to be announced: its state, the capture
/// row its propose appends from, and its position.
pub(crate) type CtxSeq<'a> = (&'a dyn ProposerState, Option<DevicePtr>, usize);

impl RankSplit {
    /// The largest swap of a context append of `n` rows.
    fn ctx_bytes(&self, n: usize) -> usize {
        let g = Geometry {
            ctx: CtxRows::single(n),
            ..self.geometry
        };
        ctx_steps(0)
            .iter()
            .filter_map(|s| match s {
                Step::Swap(swap) => Some(g.bytes(*swap)),
                Step::Run(_) => None,
            })
            .max()
            .unwrap_or(0)
    }

    /// Head, at a context append of `n` rows in a split propose: the index
    /// of the announced append it is, when it splits. An announced append
    /// whose rows differ has its swaps drained here, in their place in the
    /// plan, and appends unsplit.
    pub(crate) fn ctx_turn(
        &self,
        n: usize,
        comm: &dyn CommBackend,
        stream: u64,
    ) -> Result<Option<usize>> {
        let i = self.ctx_next.fetch_add(1, Ordering::Relaxed);
        let planned = self.plan().ctx.get(i);
        if planned == 0 || planned == n {
            return Ok((planned > 0).then_some(i));
        }
        // A mispredicted append costs its swaps, never a value: say so once.
        static SAID: AtomicBool = AtomicBool::new(false);
        let message = "rank-split context append: announced rows differ, swaps drained";
        if SAID.swap(true, Ordering::Relaxed) {
            tracing::debug!("{message} (append {i}: {planned} announced, {n} appended)");
        } else {
            tracing::warn!("{message} (append {i}: {planned} announced, {n} appended)");
        }
        for step in ctx_steps(i) {
            if let Step::Swap(swap) = step {
                self.exchange(swap, self.sink, self.sink, comm, stream)?;
            }
        }
        Ok(None)
    }
}

/// A context append the head splits: the propose's split, its communicator
/// and the announced append's index.
#[derive(Clone, Copy)]
pub(crate) struct CtxTurn<'a> {
    split: &'a RankSplit,
    comm: &'a dyn CommBackend,
    index: usize,
}

impl BlockDiffusionDraftHead {
    /// Head, right before it announces a split propose over `seqs`: the
    /// rows each sequence's context append will split, packed for the
    /// announce (`announce_word`) and kept for the propose's begin. A
    /// sequence's rows are those its propose appends from the state it
    /// starts from (`pending_ctx_rows`); only appends of 1..=`CTX_MAX_ROWS`
    /// rows on the tensor-core tiers within one precompute pass split.
    pub(crate) fn split_ctx_rows(&self, comm: &dyn CommBackend, seqs: &[CtxSeq<'_>]) -> u32 {
        let Some(split) = self.rank_split.as_ref() else {
            return 0;
        };
        let mut ctx = CtxRows::default();
        if split.geometry.ctx_in > 0 {
            for (slot, &(state, stack, position)) in ctx.0.iter_mut().zip(seqs) {
                let Some(dstate) = state.as_any().downcast_ref::<DflashProposerState>() else {
                    continue;
                };
                let n = self.pending_ctx_rows(dstate, stack, position);
                let tier = crate::layers::w4a16_gemv_tiers::tc_kernel(n as u32).0 != 0;
                if (1..=CTX_MAX_ROWS.min(self.ctx_window)).contains(&n)
                    && tier
                    && comm.supports_exchange_async(split.ctx_bytes(n))
                {
                    *slot = n as u8;
                }
            }
        }
        split.announced.store(ctx.pack(), Ordering::Relaxed);
        ctx.pack()
    }

    /// Head, at a context append of `n` rows in a propose handed `comm`
    /// (only a split propose is): the announced append it walks, if it
    /// splits (`RankSplit::ctx_turn`).
    pub(crate) fn ctx_split_turn<'a>(
        &'a self,
        comm: Option<&'a dyn CommBackend>,
        n: usize,
        stream: u64,
    ) -> Result<Option<CtxTurn<'a>>> {
        let (Some(split), Some(comm)) = (self.rank_split.as_ref(), comm) else {
            return Ok(None);
        };
        Ok(split
            .ctx_turn(n, comm, stream)?
            .map(|index| CtxTurn { split, comm, index }))
    }

    /// Head: the `fc`, `hidden_norm` and fused K/V steps of a context append
    /// (`precompute_ctx_kv`) split as `turn`, from the accumulator rows at
    /// `input` into `scratch.fc_proj` and `scratch.fused_kv_out`, where the
    /// unsplit steps leave them.
    pub(crate) fn walk_ctx_split(
        &self,
        turn: CtxTurn<'_>,
        input: DevicePtr,
        gpu: &dyn GpuBackend,
        scratch: &DflashScratch,
        stream: u64,
    ) -> Result<()> {
        let walker = CtxWalk {
            head: self,
            turn,
            gpu,
            frame: Frame {
                ctx_in: input,
                ..Frame::serial(scratch)
            },
            stream,
        };
        walk(&ctx_steps(turn.index), Graphs::Eager, gpu, stream, &walker)
    }

    /// A context piece both ranks run, over `rank`'s half: the GEMVs of the
    /// unsplit `ctx_projection` (its NVFP4 twin, at the append's rows) over a
    /// row range, and the norm over the same joined rows.
    pub(super) fn ctx_piece(
        &self,
        split: &RankSplit,
        piece: Piece,
        rank: usize,
        gpu: &dyn GpuBackend,
        frame: &Frame,
        stream: u64,
    ) -> Result<()> {
        let g = &split.plan();
        let twin = |which: usize| self.twins.ctx_q4.as_ref().map(|q4| &q4[which]);
        match piece {
            Piece::CtxFc(i) => self.split_gemv(
                gpu,
                twin(0),
                frame.ctx_in,
                split.ctx_own,
                [g.ctx.get(i), g.hidden, g.ctx_in],
                rank,
                stream,
            ),
            Piece::CtxNorm(i) => ops::rms_norm(
                gpu,
                self.kernels.rms_norm,
                frame.ctx_fc,
                &self.hidden_norm,
                frame.ctx_fc,
                g.ctx.get(i) as u32,
                g.hidden as u32,
                self.rms_norm_eps,
                stream,
            ),
            Piece::CtxKv(i) => self.split_gemv(
                gpu,
                twin(1),
                frame.ctx_fc,
                split.ctx_own,
                [g.ctx.get(i), g.ctx_kv, g.hidden],
                rank,
                stream,
            ),
            other => bail!("rank-split propose: {other:?} is not a context piece"),
        }
    }
}

/// The head's view of one split context append.
struct CtxWalk<'a> {
    head: &'a BlockDiffusionDraftHead,
    turn: CtxTurn<'a>,
    gpu: &'a dyn GpuBackend,
    frame: Frame,
    stream: u64,
}

impl SplitOps for CtxWalk<'_> {
    fn piece(&self, piece: Piece) -> Result<()> {
        let split = self.turn.split;
        self.head
            .ctx_piece(split, piece, 0, self.gpu, &self.frame, self.stream)
    }

    fn swap(&self, swap: Swap) -> Result<()> {
        let CtxTurn { split, comm, .. } = self.turn;
        split.swap(swap, 0, self.gpu, comm, &self.frame, self.stream)
    }
}
