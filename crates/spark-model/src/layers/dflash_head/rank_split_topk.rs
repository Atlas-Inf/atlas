// SPDX-License-Identifier: AGPL-3.0-only

//! The rank split's candidate swap (`ATLAS_GLM_DRAFT_TP_TOPK`, `rank_split`):
//! each rank's top-K lists of its share of the vocabulary, and the head's
//! selector walk over both.

use anyhow::{Context, Result};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::super::super::{BlockDiffusionDraftHead, DflashScratch};
use super::super::half;
use super::RankSplit;
use crate::layers::ops;

/// A rank's side of the candidate swap (`Swap::Candidates`).
pub(crate) struct TopkSelect {
    /// This rank's and (on the head) the worker's top-K lists, `gamma` rows.
    pub(crate) lists: DevicePtr,
    pub(crate) lists_peer: DevicePtr,
    /// Zeroed `dflash2_selector_topk` scratch.
    scratch: DevicePtr,
    topk: KernelHandle,
    /// The walk over both lists: the confidence twin when the head's
    /// selector reports confidences.
    walk: KernelHandle,
}

impl TopkSelect {
    pub(crate) fn new(gamma: usize, draft_conf: bool, gpu: &dyn GpuBackend) -> Result<Self> {
        let (module, walk) = if draft_conf {
            (
                crate::layers::dflash_head::draft_conf::conf_module(),
                "dflash2_selector_walk_conf",
            )
        } else {
            ("dflash2_candidate_selector", "dflash2_selector_walk")
        };
        let lists = ops::dflash2_selector_lists_bytes(gamma);
        let scratch_bytes = ops::dflash2_selector_scratch_bytes(gamma);
        let scratch = gpu.alloc(scratch_bytes)?;
        gpu.memset(scratch, 0, scratch_bytes)?;
        Ok(Self {
            lists: gpu.alloc(lists)?,
            lists_peer: gpu.alloc(lists)?,
            scratch,
            topk: gpu.kernel("dflash2_candidate_selector", "dflash2_selector_topk")?,
            walk: gpu.kernel(module, walk)?,
        })
    }
}

impl RankSplit {
    /// `Piece::TopK` on `rank`: every row's exact top-K of this rank's half
    /// of the logits, candidate ids in the whole vocabulary.
    pub(crate) fn topk_piece(&self, rank: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        let g = self.plan();
        let t = self
            .topk
            .as_ref()
            .context("rank-split propose: candidate swap without its lists")?;
        let (first, columns) = half(g.vocab, rank);
        ops::dflash2_selector_topk(
            gpu,
            t.topk,
            self.logits,
            columns as u32,
            columns as u32,
            first as u32,
            g.gamma as u32,
            t.lists,
            t.scratch,
            stream,
        )
    }
}

impl BlockDiffusionDraftHead {
    /// Head: the selector's walk over this rank's and the worker's candidate
    /// lists (`Swap::Candidates`), from the hidden rows `Piece::TopK`
    /// projected, into `scratch.draft_tokens_dev`.
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
            [t.lists, t.lists_peer],
            scratch.dflash2_projected_hidden,
            scratch.draft_tokens_dev,
            self.gamma,
            gpu,
            t.walk,
            stream,
        )
    }
}
