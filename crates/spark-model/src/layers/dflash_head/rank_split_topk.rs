// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split's candidate swap (`ATLAS_GLM_DRAFT_TP_TOPK`, `rank_split`):
//! each rank's top-K lists of its share of the vocabulary, sequence by
//! sequence (one for a single-sequence propose, B for a batched one), and
//! the head's selector walk of each sequence over every rank's lists (the
//! pair's two, the chain's four).

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::super::{BlockDiffusionDraftHead, DflashScratch};
use super::super::{Swap, share};
use super::{RankSplit, Shares};
use crate::layers::ops;

/// A rank's side of the candidate swap (`Swap::Candidates`).
pub(crate) struct TopkSelect {
    /// Every rank's top-K lists of up to the split's capacity in rows, one
    /// `gamma`-row block a sequence: this rank's, and on the head the
    /// workers' once swapped.
    pub(crate) lists: Shares,
    /// Rows of a sequence's block.
    gamma: usize,
    /// The ranks whose lists the walk merges.
    ranks: usize,
    /// Zeroed `dflash2_selector_topk` scratch.
    scratch: DevicePtr,
    topk: KernelHandle,
    /// The walk over every rank's lists (the pair's two-list entry, the
    /// chain's `_ranks` one): the confidence twin when the head's selector
    /// reports confidences.
    pub(crate) walk: KernelHandle,
}

impl TopkSelect {
    pub(crate) fn new(
        gamma: usize,
        capacity: usize,
        draft_conf: bool,
        ranks: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let ranked = if ranks == 2 { "" } else { "_ranks" };
        let (module, walk) = if draft_conf {
            (
                crate::layers::dflash_head::draft_conf::conf_module(),
                format!("dflash2_selector_walk{ranked}_conf"),
            )
        } else {
            (
                "dflash2_candidate_selector",
                format!("dflash2_selector_walk{ranked}"),
            )
        };
        let stride = ops::dflash2_selector_lists_bytes(capacity.max(gamma));
        let scratch_bytes = ops::dflash2_selector_scratch_bytes(gamma);
        let scratch = gpu.alloc(scratch_bytes)?;
        gpu.memset(scratch, 0, scratch_bytes)?;
        Ok(Self {
            lists: Shares {
                base: gpu.alloc(ranks * stride)?,
                stride,
            },
            gamma,
            ranks,
            scratch,
            topk: gpu.kernel("dflash2_candidate_selector", "dflash2_selector_topk")?,
            walk: gpu.kernel(module, &walk)?,
        })
    }
}

impl TopkSelect {
    /// Rank `rank`'s lists of sequence `seq`.
    fn block(&self, rank: usize, seq: usize) -> DevicePtr {
        self.lists
            .of(rank)
            .offset(seq * ops::dflash2_selector_lists_bytes(self.gamma))
    }

    /// Every rank's lists of sequence `seq`, as the walk reads them.
    pub(crate) fn sequence(&self, seq: usize) -> ops::SelectorLists {
        match self.ranks {
            2 => ops::SelectorLists::Pair([self.block(0, seq), self.block(1, seq)]),
            ranks => ops::SelectorLists::Ranks {
                base: self.block(0, seq),
                stride: self.lists.stride,
                count: ranks as u32,
            },
        }
    }
}

impl RankSplit {
    /// `Piece::TopK` on `rank`: every row's exact top-K of this rank's share
    /// of the logits, candidate ids in the whole vocabulary, one block a
    /// sequence of the propose's rows.
    pub(crate) fn topk_piece(&self, rank: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let g = self.plan();
        let t = self
            .topk
            .as_ref()
            .context("rank-split propose: candidate swap without its lists")?;
        let (first, columns) = share(g.vocab, rank, g.ranks);
        let logits = self.own(Swap::Logits, rank);
        for seq in 0..g.gamma / t.gamma {
            ops::dflash2_selector_topk(
                gpu,
                t.topk,
                logits.offset(seq * t.gamma * columns * 2),
                columns as u32,
                columns as u32,
                first as u32,
                t.gamma as u32,
                t.block(rank, seq),
                t.scratch,
                stream,
            )?;
        }
        Ok(())
    }
}

impl BlockDiffusionDraftHead {
    /// Head: the selector's walk over every rank's candidate lists
    /// (`Swap::Candidates`), from the hidden rows `Piece::TopK` projected,
    /// into `scratch.draft_tokens_dev`.
    pub(crate) fn select_split_lists(
        &self,
        split: &RankSplit,
        last_token: u32,
        ban_depth: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
        scratch: &DflashScratch,
    ) -> Result<()> {
        let (selector, t) = self
            .candidate_selector
            .as_ref()
            .zip(split.topk.as_ref())
            .context("rank-split candidate swap without a DFlash2 selector")?;
        selector.walk_lists(
            Self::selector_anchor(last_token, ban_depth, scratch),
            t.sequence(0),
            scratch.dflash2_projected_hidden,
            scratch.draft_tokens_dev,
            self.gamma,
            gpu,
            t.walk,
            stream,
        )
    }
}
