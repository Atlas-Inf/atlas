// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-split batched propose (`ATLAS_GLM_DRAFT_TP_BATCH`, default off): the
//! staged B×gamma forward walked as the single-sequence split's pieces and
//! swaps (`rank_split`), over the batch buffers.
//!
//! The model announces it with its row count (`model::draft_assist`); the
//! worker walks the same swaps at those rows over its own batch buffers. The
//! halves are the GEMVs the unsplit batched forward launches, over a row
//! range at the same B×gamma rows (`rank_split_batch_with` admits only
//! those), so every value is the one the unsplit launch writes. Eager: the
//! batched propose captures no graphs.

use anyhow::{Result, bail, ensure};
use spark_comm::CommBackend;

use super::BlockDiffusionDraftHead;
use super::batch_forward::BatchLayerArgs;
use super::rank_split::{Frame, Graphs, Piece, RankSplit, SplitOps, Swap, walk};

impl BlockDiffusionDraftHead {
    /// `ATLAS_GLM_DRAFT_TP_BATCH`: the split, communicator and rows a
    /// batched propose of `n` sequences on the head rank swaps with, if it
    /// splits; the one predicate the model announces by and the batched
    /// propose splits by. It splits only the staged native B×gamma forward
    /// whose GEMVs the halves reproduce: the tensor-core NVFP4 tiers at
    /// B×gamma rows, and for the head the NVFP4 twin `project_head` takes.
    pub(crate) fn rank_split_batch_with<'a>(
        &'a self,
        comm: Option<&'a dyn CommBackend>,
        n: usize,
        grammar: bool,
    ) -> Option<(&'a RankSplit, &'a dyn CommBackend, usize)> {
        let (split, comm) = self.rank_split_with(comm, grammar)?;
        let rows = n.checked_mul(self.gamma)?;
        let tier = |m: usize| crate::layers::w4a16_gemv_tiers::tc_kernel(m as u32).0 != 0;
        let tiers = tier(rows.min(32)) && (rows % 32 == 0 || tier(rows % 32));
        let staged = self.startup.generic_batch_authoritative
            && !self.startup.native_batch_authoritative
            && !self.startup.diagnostics.batch_parity;
        let head_twin = !split.geometry.parts.head
            || (self.lm_head_nvfp4.is_none()
                && !(self.lm_head_shared_fp8.is_some()
                    && matches!(self.quant, super::DflashQuantization::Fp8Weights)));
        (n >= 2
            && rows <= split.capacity
            && tiers
            && staged
            && head_twin
            && comm.supports_exchange_async(split.max_bytes_at(rows)))
        .then_some((split, comm, rows))
    }

    /// This head's B×gamma buffers as a split frame.
    pub(crate) fn batch_frame(&self) -> Frame {
        Frame {
            norm: self.batch_norm,
            inter: self.batch_mlp_gate,
            acc: self.batch_mlp_down,
            logits: self.batch_logits,
            ..Frame::serial(&self.scratch)
        }
    }

    /// Head: the staged forward's layers and tail base (`a.batch_rows` rows,
    /// the rows the propose began with), leaving `batch_tokens` as the
    /// unsplit `run_batched_layer_stage` loop and `run_batched_tail_base` do.
    pub(super) fn walk_batch_split(
        &self,
        split: &RankSplit,
        comm: &dyn CommBackend,
        a: BatchLayerArgs<'_>,
    ) -> Result<()> {
        ensure!(
            a.batch_rows as usize == split.plan().gamma,
            "rank-split batched propose of {} rows began at {}",
            a.batch_rows,
            split.plan().gamma
        );
        let walker = BatchWalk {
            head: self,
            split,
            comm,
            frame: self.batch_frame(),
            a,
        };
        walk(
            &split.geometry.layer_steps(0),
            Graphs::Eager,
            a.ctx.gpu,
            a.stream,
            &walker,
        )
    }
}

/// The head's view of one batched propose.
struct BatchWalk<'a> {
    head: &'a BlockDiffusionDraftHead,
    split: &'a RankSplit,
    comm: &'a dyn CommBackend,
    frame: Frame,
    a: BatchLayerArgs<'a>,
}

impl SplitOps for BatchWalk<'_> {
    fn piece(&self, piece: Piece) -> Result<()> {
        let (head, a) = (self.head, &self.a);
        match piece {
            Piece::Attention(l) => head.batched_attention(l, a),
            Piece::Project(l) => head.batched_project(l, a),
            Piece::Residual(l) => head.batched_residual(l, a),
            Piece::Post(l) => {
                head.batched_project(l, a)?;
                head.batched_mlp(l, a)?;
                head.batched_residual(l, a)
            }
            Piece::Norm => head.batched_final_norm(a.batch_rows, a.ctx, a.stream),
            Piece::Tail => head.run_batched_tail_base(a.batch_rows, a.ctx, a.stream),
            Piece::Select => head.batched_argmax(a.batch_rows, a.ctx, a.stream),
            Piece::GateUp(_) | Piece::Down(_) | Piece::Vocab => {
                head.split_piece(self.split, piece, 0, a.ctx.gpu, &self.frame, a.stream)
            }
            Piece::CtxFc(_) | Piece::CtxNorm(_) | Piece::CtxKv(_) => {
                bail!("rank-split propose: {piece:?} walks with its context append")
            }
        }
    }

    fn swap(&self, swap: Swap) -> Result<()> {
        self.split.swap(
            swap,
            0,
            self.a.ctx.gpu,
            self.comm,
            &self.frame,
            self.a.stream,
        )
    }
}

/// A begun batched split propose. Dropped, however the propose ended, it
/// issues every swap the propose did not reach (`RankSplit::finish`), so the
/// worker, which walks them all, is never left waiting in one.
pub(super) struct BatchSplitGuard<'a> {
    split: Option<(&'a RankSplit, &'a dyn CommBackend)>,
    stream: u64,
}

impl<'a> BatchSplitGuard<'a> {
    pub(super) fn begin(
        split: Option<(&'a RankSplit, &'a dyn CommBackend, usize)>,
        stream: u64,
    ) -> Result<Self> {
        if let Some((split, _, rows)) = split {
            split.begin_announced(rows)?;
        }
        Ok(Self {
            split: split.map(|(split, comm, _)| (split, comm)),
            stream,
        })
    }

    /// Issues the swaps the propose did not reach and returns a failed drain,
    /// which a drop can only log.
    pub(super) fn finish(mut self) -> Result<()> {
        match self.split.take() {
            Some((split, comm)) => split.finish(comm, self.stream),
            None => Ok(()),
        }
    }
}

impl Drop for BatchSplitGuard<'_> {
    fn drop(&mut self) {
        if let Some((split, comm)) = self.split
            && let Err(e) = split.finish(comm, self.stream)
        {
            tracing::error!("rank-split batched propose: draining its swaps failed: {e:#}");
        }
    }
}
