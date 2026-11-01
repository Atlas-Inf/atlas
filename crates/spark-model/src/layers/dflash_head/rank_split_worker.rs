// SPDX-License-Identifier: AGPL-3.0-only

//! The worker's side of a split propose (`rank_split`): its whole walk,
//! enqueued at the announce.

use anyhow::{Context, Result, ensure};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

use super::super::super::BlockDiffusionDraftHead;
use super::super::ctx::read_announce;
use super::super::{Frame, Graphs, Piece, SplitOps, Swap, walk};
use super::RankSplit;

impl BlockDiffusionDraftHead {
    /// Worker side of a split propose: enqueue this rank's whole walk on
    /// `stream`. No host sync: the swaps order it against the head. `word`
    /// is the announce's preamble word (`announce_word`): a batched
    /// propose's B×gamma rows (0: a single sequence's gamma) and, with
    /// `ATLAS_GLM_DRAFT_TP_CTX`, its context appends' rows.
    pub fn rank_split_serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        word: u32,
    ) -> Result<()> {
        let split = self
            .rank_split
            .as_ref()
            .context("rank-split propose announced to a head without the split")?;
        ensure!(
            comm.rank() == 1 && comm.world_size() == 2,
            "rank-split propose serves on rank 1 of 2"
        );
        let (rows, ctx) = read_announce(word, split.geometry.ctx_in > 0);
        // The joined `fc` rows land in the scratch window, as on the head.
        ensure!(
            ctx.appends().all(|i| ctx.get(i) <= self.ctx_window),
            "rank-split propose announced context rows {ctx:?} past the window {}",
            self.ctx_window
        );
        let frame = Frame {
            ctx_in: split.ctx_input,
            ..match rows {
                0 => Frame::serial(&self.scratch),
                _ => self.batch_frame(),
            }
        };
        split.begin(if rows == 0 { self.gamma } else { rows }, ctx)?;
        let worker = Worker {
            head: self,
            split,
            frame,
            gpu,
            comm,
            stream,
        };
        let walked = walk(&split.plan().steps(1), Graphs::Eager, gpu, stream, &worker);
        // The head is already walking these swaps: issue them all.
        let drained = split.finish(comm, stream);
        walked.and(drained)
    }
}

struct Worker<'a> {
    head: &'a BlockDiffusionDraftHead,
    split: &'a RankSplit,
    frame: Frame,
    gpu: &'a dyn GpuBackend,
    comm: &'a dyn CommBackend,
    stream: u64,
}

impl SplitOps for Worker<'_> {
    fn piece(&self, piece: Piece) -> Result<()> {
        self.head
            .split_piece(self.split, piece, 1, self.gpu, &self.frame, self.stream)
    }
    fn swap(&self, swap: Swap) -> Result<()> {
        self.split
            .swap(swap, 1, self.gpu, self.comm, &self.frame, self.stream)
    }
}
