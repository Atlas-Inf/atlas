// SPDX-License-Identifier: AGPL-3.0-only

//! The finish leaf of qwen4_exp (`ATLAS_QWEN4EXP_FINISH_LEAF=1`, default off,
//! with the parent module's preconditions).
//!
//! # What differs from the GLM leaf
//!
//! qwen4_exp keeps per-sequence state outside the SSM pool: the PLE n-gram
//! token history and conv carry, and each QSA indexer's pooled block keys and
//! raw-key tail. A restore needs all of it (`marconi_restorable` declines a
//! slot without aux), so a leaf carries it: `finish_leaf_write`
//! attaches `collect_aux_states_into` to the slot, the bytes a chunk
//! checkpoint attaches, read from the same layer states. Nothing else is
//! per-sequence and outlives a token: the mHC streams are per row, and the
//! MTP drafter attends only to the rows it wrote during this request's
//! decode (its KV starts empty at each request, `Qwen4ExpMtpProposerState`),
//! so it needs nothing from a snapshot; its first draft after a restore
//! reads the suffix prefill's last row, as after any restore.
//!
//! The GLM leaf is saved ON the boundary, mid-step, by folding KDA records
//! in two launches. A qwen4_exp verify commit (the exact lane, its deferred
//! GDN replay, the PLE carry and QSA rewinds) has no such split point, so
//! the leaf is saved at the END of the step that crossed a save boundary,
//! `B <= at < B + k`, where the whole state is canonical: after the commit
//! (its copies on the secondary stream, ordered before the save), or after a
//! plain decode step. Registering a snapshot off the block grid is ordinary
//! (chunk checkpoints at 16388-token chunk ends are; the index keys any
//! length), but a lookup only returns one at or below its whole-block match,
//! and the finish insert caches whole blocks only. So the leaf is reachable
//! once the sequence has run to the end of the block `at` falls in.
//!
//! # Two slots while the leaf sits in a partial block
//!
//! A turn that finishes before the leaf's block is complete could not reach
//! it, and a rolling slot would by then have overwritten the leaf before it,
//! which was reachable. So the leaf before it stays registered (`prev` in
//! the cell) until the current one's block is complete, then goes back to
//! the free list. At finish the deepest reachable one is kept and the other
//! freed. Both are ordinary leaves in the index (evicted first, never taking
//! a frontier), so the second slot is never held at a conversation's
//! expense, and it is held for at most one block of decode per span.
//!
//! # Rank symmetry
//!
//! Every input of a save decision is rank-identical: the hooks run on both
//! ranks from the same verdicts (`commit_accepted_prefix` on the head, and
//! on the worker through `ep_worker_apply_verdict`; `decode` /
//! `decode_batch` and the worker's batched decode), and the finish caching
//! is mirrored by the head's command, as for GLM. A rank that could not get
//! a slot only loses its leaf, and the restore agreement brings both ranks
//! to the deepest depth both hold.
//!
//! # Cost
//!
//! Per save: the SSM state copy (one Marconi slot, about 57 MB at TP2) and
//! the aux readback, about 768 B per token of context for the twelve QSA
//! indexers (59 MB at 77K) plus 0.37 MB of PLE carry, through the pinned
//! stage with `ATLAS_QWEN4EXP_AUX_PINNED=1`. One save per
//! `ATLAS_GLM_PC_FINISH_LEAF_BLOCKS` blocks of decode (default 4: 64 tokens).

use super::super::super::types::TransformerModel;
use super::rolling::{FinishLeaf, boundary_row};
use super::{enabled, qwen4exp_enabled, span_blocks};
use crate::traits::SequenceState;
use anyhow::Result;

/// The position a step that committed `rows` rows and left the sequence at
/// `end` tokens saves its leaf at: `end`, when the step crossed a multiple
/// of `span`.
pub(in crate::model) fn step_end_save(end: usize, rows: usize, span: usize) -> Option<usize> {
    boundary_row(end.checked_sub(rows)?, rows, span).map(|_| end)
}

/// What one step of a sequence does with its leaves ([`plan_step`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) struct StepPlan {
    /// Free the leaf before the current one: the current one is reachable.
    pub retire_prev: bool,
    /// Save a new leaf at this position.
    pub save_at: Option<usize>,
}

/// The step that left a sequence at `end` tokens after committing `rows`
/// rows, with current leaf `cur`, a save span of `span` tokens, blocks of
/// `bs` and the restore floor `min`.
pub(in crate::model) fn plan_step(
    cur: Option<FinishLeaf>,
    end: usize,
    rows: usize,
    (span, bs, min): (usize, usize, usize),
) -> StepPlan {
    let at = step_end_save(end, rows, span);
    StepPlan {
        retire_prev: cur.is_some_and(|l| reachable(l.tokens, end, bs)),
        save_at: at.filter(|&at| at >= min && cur.is_none_or(|l| l.tokens != at)),
    }
}

/// Whether a leaf at `at` tokens is reachable once a sequence of `len`
/// tokens is cached: the finish insert caches its whole blocks only.
pub(in crate::model) fn reachable(at: usize, len: usize, bs: usize) -> bool {
    bs > 0 && at <= len / bs * bs
}

/// Of the current leaf and the one before it, the one a sequence of `len`
/// tokens keeps at finish (the deepest reachable), and the ones it frees.
pub(in crate::model) fn finish_pick(
    cur: Option<FinishLeaf>,
    prev: Option<FinishLeaf>,
    len: usize,
    bs: usize,
) -> (Option<FinishLeaf>, Vec<FinishLeaf>) {
    let mut keep = None;
    let mut free = Vec::new();
    for leaf in [cur, prev].into_iter().flatten() {
        if keep.is_none() && reachable(leaf.tokens, len, bs) {
            keep = Some(leaf);
        } else {
            free.push(leaf);
        }
    }
    (keep, free)
}

impl TransformerModel {
    /// The qwen4_exp leaf is on for this model.
    pub(in crate::model) fn qwen4exp_leaf_on(&self) -> bool {
        qwen4exp_enabled() && self.config.model_type == "qwen4_exp"
    }

    /// A finish leaf of either kind is on for this model: what caches a
    /// finished sequence on every rank and orders the snapshot copies.
    pub(in crate::model) fn leaf_on(&self) -> bool {
        enabled() || self.qwen4exp_leaf_on()
    }

    /// A verify commit of `rows` rows of `seq` was just issued (h/conv on the
    /// secondary stream, the aux rewinds on the default one): save the leaf
    /// when the step crossed a save boundary.
    pub(in crate::model) fn qwen4exp_leaf_after_commit(
        &self,
        seq: &SequenceState,
        rows: usize,
    ) -> Result<()> {
        if !self.qwen4exp_leaf_on() {
            return Ok(());
        }
        let Some(at) = self.qwen4exp_leaf_plan(seq, rows) else {
            return Ok(());
        };
        // Only a save orders the default stream after the commit's copies:
        // every other step keeps the commit's overlap with the next propose.
        self.sync_secondary_dispatch()?;
        self.qwen4exp_leaf_save(seq, at, self.gpu.default_stream());
        Ok(())
    }

    /// A plain decode step of `seq` advanced its state in place on `stream`.
    pub(in crate::model) fn qwen4exp_leaf_after_step(
        &self,
        seq: &SequenceState,
        rows: usize,
        stream: u64,
    ) {
        if let Some(at) = self.qwen4exp_leaf_plan(seq, rows) {
            self.qwen4exp_leaf_save(seq, at, stream);
        }
    }

    /// Retire the leaf before the current one once the current one is
    /// reachable, and the save this step of `rows` rows makes, if any.
    fn qwen4exp_leaf_plan(&self, seq: &SequenceState, rows: usize) -> Option<usize> {
        let span = self.finish_leaf_span(seq)?;
        // Below the restore floor a leaf could never be restored.
        let min = crate::model::mtp_carry::marconi_min_tokens();
        let shape = (span, span / span_blocks(), min);
        let plan = plan_step(seq.finish_leaf.get(), seq.tokens.len(), rows, shape);
        if plan.retire_prev
            && let Some(prev) = seq.finish_leaf.take_prev()
            && let Some(snap) = self.finish_leaf_take_leaf(seq, prev)
        {
            self.ssm_snapshots.free(snap);
        }
        plan.save_at
    }

    /// Save `seq`'s state on `stream` as its leaf at `at` tokens. The
    /// current leaf becomes the one before it; a still-registered older one
    /// gives its slot.
    fn qwen4exp_leaf_save(&self, seq: &SequenceState, at: usize, stream: u64) {
        let prev = seq.finish_leaf.take_prev();
        let Some(snap) = prev
            .and_then(|l| self.finish_leaf_take_leaf(seq, l))
            .or_else(|| self.finish_leaf_reserve(seq.session_hash))
        else {
            return;
        };
        if self.finish_leaf_write(seq, snap, at, None, stream) {
            if let Some(cur) = seq.finish_leaf.take() {
                seq.finish_leaf.set_prev(cur);
            }
            seq.finish_leaf.set(FinishLeaf { snap, tokens: at });
        }
    }

    /// At finish (both ranks, before the leaf is reported): the leaf the
    /// next turn can reach ([`finish_pick`]); the other one is freed. The GLM
    /// leaf is always on a block boundary at or below the end, so for it
    /// this is the current leaf.
    pub(super) fn finish_leaf_settle_at_finish(
        &self,
        seq: &SequenceState,
        bs: usize,
    ) -> Option<FinishLeaf> {
        let (cur, prev) = (seq.finish_leaf.take(), seq.finish_leaf.take_prev());
        let (keep, free) = finish_pick(cur, prev, seq.tokens.len(), bs);
        for leaf in free {
            tracing::info!(
                "finish-leaf: leaf at token {} of {} dropped (not the deepest reachable one)",
                leaf.tokens,
                seq.tokens.len()
            );
            if let Some(snap) = self.finish_leaf_take_leaf(seq, leaf) {
                self.ssm_snapshots.free(snap);
            }
        }
        keep
    }
}

#[cfg(test)]
#[path = "qwen4exp_tests.rs"]
mod tests;
