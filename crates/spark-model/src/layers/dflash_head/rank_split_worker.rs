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
    /// `ATLAS_GLM_DRAFT_TP_CTX`, its context appends' rows; `extra` the
    /// second announce word, when [`Self::rank_split_extra_follows`].
    pub fn rank_split_serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        word: u32,
        extra: Option<u32>,
    ) -> Result<()> {
        let split = self
            .rank_split
            .as_ref()
            .context("rank-split propose announced to a head without the split")?;
        let rank = comm.rank();
        ensure!(
            rank > 0 && comm.world_size() == split.geometry.ranks,
            "rank-split propose serves on a worker of its {} ranks (rank {rank} of {})",
            split.geometry.ranks,
            comm.world_size()
        );
        let (rows, ctx) = read_announce(word, extra, split.geometry.ctx_in > 0);
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
            rank,
        };
        let walked = walk(
            &split.plan().steps(rank),
            Graphs::Eager,
            gpu,
            stream,
            &worker,
        );
        // The head is already walking these swaps: issue them all.
        let drained = split.finish(comm, stream);
        walked.and(drained)
    }
}

impl BlockDiffusionDraftHead {
    /// Worker: whether a second announce word follows the command of the
    /// split propose announced by `word` (`rank_split::extra_follows`).
    pub fn rank_split_extra_follows(&self, word: u32) -> bool {
        let Some(split) = self.rank_split.as_ref() else {
            return false;
        };
        let ctx_split = split.geometry.ctx_in > 0;
        let (rows, _) = read_announce(word, None, ctx_split);
        super::super::ctx::extra_follows(rows / self.gamma, ctx_split)
    }
}

struct Worker<'a> {
    head: &'a BlockDiffusionDraftHead,
    split: &'a RankSplit,
    frame: Frame,
    gpu: &'a dyn GpuBackend,
    comm: &'a dyn CommBackend,
    stream: u64,
    rank: usize,
}

impl SplitOps for Worker<'_> {
    fn piece(&self, piece: Piece) -> Result<()> {
        self.head.split_piece(
            self.split,
            piece,
            self.rank,
            self.gpu,
            &self.frame,
            self.stream,
        )
    }
    fn swap(&self, swap: Swap) -> Result<()> {
        self.split.swap(
            swap,
            self.rank,
            self.gpu,
            self.comm,
            &self.frame,
            self.stream,
        )
    }
}
