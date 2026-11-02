// SPDX-License-Identifier: AGPL-3.0-only

//! Rank-split propose (`ATLAS_GLM_DRAFT_TP`, `ATLAS_QWEN4EXP_MTP_DRAFT_TP`),
//! model side.
//!
//! The head announces a propose that will swap work with the worker
//! (`layers::dflash_head::rank_split`, `layers::qwen4exp_mtp::draft_tp`); the
//! worker serves it from its own copy of the drafter ([`DraftAssist`]), which
//! proposes nothing.
//!
//! Wire protocol, v2-addressed:
//!
//! ```text
//! EC  (seq_id 0)     -> the worker enqueues its walk of the propose's swaps
//! EC  (seq_id rows)  -> the same for a batched propose of `rows` = B×gamma
//!                       rows (`ATLAS_GLM_DRAFT_TP_BATCH`, v2 only)
//! EC  (seq_id rows | ctx << 12)
//!                    -> either, with the rows of its context appends
//!                       (`ATLAS_GLM_DRAFT_TP_CTX`, v2 only; `announce_word`)
//! EC  (seq_id plan)  -> a qwen4_exp batched propose's draft head
//!                       (`ATLAS_QWEN4EXP_MTP_DRAFT_TP`, v2 only; `draft_tp::Plan`)
//! ```
//!
//! The command carries no payload: both ranks plan the swaps from the same
//! drafter shapes, the same agreed switches (`startup_parity`) and, for a
//! batched propose or split context appends, the preamble slot word. The head
//! sends it only outside any other command, so the worker is at its command
//! loop; the worker enqueues and returns there, its next command ordered
//! behind the walk on the same stream.

use std::sync::Arc;

use anyhow::{Context, Result};
use spark_comm::CommBackend;
use spark_runtime::gpu::GpuBackend;

use super::types::TransformerModel;
use crate::layers::BlockDiffusionDraftHead;
use crate::layers::dflash_head::rank_split::announce_word;
use crate::layers::qwen4exp_mtp::draft_tp::Qwen4ExpDraftAssist;
use crate::speculative::DraftProposer;

pub(super) const EP_CMD_DRAFT_ASSIST: u32 = 0xFFFF_FFEC;

/// A worker rank's copy of the drafter, serving the head's announced
/// proposes: `word` is the announce's preamble word.
pub trait DraftAssist: Send + Sync {
    /// Enqueue this rank's whole part of the announced propose on `stream`,
    /// with no host sync (the swaps order it against the head).
    fn serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        word: u32,
    ) -> Result<()>;
}

impl DraftAssist for BlockDiffusionDraftHead {
    fn serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        word: u32,
    ) -> Result<()> {
        self.rank_split_serve(gpu, comm, stream, word)
    }
}

impl DraftAssist for Qwen4ExpDraftAssist {
    fn serve(
        &self,
        gpu: &dyn GpuBackend,
        comm: &dyn CommBackend,
        stream: u64,
        word: u32,
    ) -> Result<()> {
        Qwen4ExpDraftAssist::serve(self, gpu, comm, stream, word)
    }
}

impl TransformerModel {
    /// Install the worker rank's copy of the drafter: it serves split
    /// proposes and is never the proposer.
    pub fn set_draft_assist(&mut self, head: Arc<dyn DraftAssist>) {
        self.draft_assist = Some(head);
    }

    /// Head: the communicator a propose swaps over, when it will split.
    pub(super) fn draft_split_comm(
        &self,
        proposer: &dyn DraftProposer,
        grammar: bool,
    ) -> Option<&dyn CommBackend> {
        let comm = self.comm_ref()?;
        (self.multi_rank_protocol_active() && proposer.rank_split_ready(comm, grammar))
            .then_some(comm)
    }

    /// Head: announce the split propose about to run, with its packed
    /// context rows (`DraftProposer::rank_split_ctx`; 0 without v2).
    pub(super) fn announce_draft_split(&self, ctx: u32) -> Result<()> {
        let word = announce_word(0, ctx)?;
        self.ep_broadcast_seq_and_cmd(word, EP_CMD_DRAFT_ASSIST, self.ep_protocol_v2)
    }

    /// Head: the communicator and rows a batched propose of `n` sequences
    /// swaps over, when it will split. The rows ride the v2 preamble.
    pub(super) fn draft_split_batch(
        &self,
        proposer: &dyn DraftProposer,
        n: usize,
        grammar: bool,
    ) -> Option<(&dyn CommBackend, usize)> {
        let comm = self.comm_ref()?;
        let rows = (self.multi_rank_protocol_active() && self.ep_protocol_v2)
            .then(|| proposer.rank_split_batch_rows(comm, n, grammar))??;
        Some((comm, rows))
    }

    /// Head: announce the batched split propose of `rows` rows about to run,
    /// with its packed context rows.
    pub(super) fn announce_draft_split_rows(&self, rows: usize, ctx: u32) -> Result<()> {
        self.ep_broadcast_seq_and_cmd(announce_word(rows, ctx)?, EP_CMD_DRAFT_ASSIST, true)
    }

    /// Head: `ATLAS_QWEN4EXP_MTP_DRAFT_TP`. When a batched propose of `n`
    /// sequences and `num_drafts` positions will split its draft head,
    /// announce it with its plan and return the communicator to hand it.
    pub(super) fn announce_draft_tp(
        &self,
        proposer: &dyn DraftProposer,
        n: usize,
        num_drafts: usize,
        grammar: bool,
        ctx: &crate::layer::ForwardContext<'_>,
    ) -> Result<Option<&dyn CommBackend>> {
        let Some(comm) = self.comm_ref() else {
            return Ok(None);
        };
        if !(self.multi_rank_protocol_active() && self.ep_protocol_v2) {
            return Ok(None);
        }
        let Some(word) = proposer.draft_tp_word(comm, n, num_drafts, grammar, ctx) else {
            return Ok(None);
        };
        self.ep_broadcast_seq_and_cmd(word, EP_CMD_DRAFT_ASSIST, true)?;
        Ok(Some(comm))
    }

    /// Worker side of EC: `seq_id` is the announce word (`announce_word`).
    pub(super) fn draft_assist_serve(&self, seq_id: u32) -> Result<bool> {
        let head = self
            .draft_assist
            .as_ref()
            .context("rank-split propose announced, but this rank holds no drafter")?;
        let comm = self
            .comm_ref()
            .context("rank-split propose without a communicator")?;
        head.serve(self.gpu.as_ref(), comm, self.gpu.default_stream(), seq_id)?;
        Ok(true)
    }
}
