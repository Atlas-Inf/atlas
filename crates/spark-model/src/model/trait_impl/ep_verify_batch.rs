// SPDX-License-Identifier: AGPL-3.0-only

//! The batched multi-sequence verify (`verify_e`) over a parallel
//! communicator: qwen4_exp at TP=EP=2 under `ATLAS_QWEN4EXP_BATCH_FAST=1` and
//! `ATLAS_QWEN4EXP_EXACT_VERIFY=1` (`model/qwen4exp_batch_fast.rs`).
//!
//! Wire protocol (v2, after the head's scheduler picked the batch):
//!
//! ```text
//! preamble seq_id = 0  (ignored: the payload routes the batch)
//! cmd = EP_CMD_VERIFY_BATCH
//! n (u32)
//! seq_ids[n]   (one bulk broadcast: each sequence's slot)
//! ks[n]        (one bulk broadcast: each sequence's verify rows)
//! tokens[R]    (one bulk broadcast, R = sum ks, seq-major)
//!   ... both ranks run `decode_verify_batched_dispatch`: the same layers,
//!       the same row counts, the same collectives in the same order ...
//! accepted[n]  (one bulk broadcast, after the head's picks: drafts accepted
//!               per sequence, in batch order)
//! ```
//!
//! The worker then commits each sequence exactly as the head's
//! `k4_apply_verdict` does (`ep_worker_apply_verdict`: rewind, trim, commit
//! the accepted SSM prefix, roll the PLE carry and the QSA ingest back; a
//! one-row decode row commits its row and trims nothing). The
//! head sends the verdicts once every row has been read and before any
//! commit or re-propose, so nothing the worker waits on can interleave.

use anyhow::{Result, ensure};

use super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// The batched-verify command word.
pub(in crate::model) const EP_CMD_VERIFY_BATCH: u32 = 0xFFFF_FFE6;

impl TransformerModel {
    /// Whether the batched verify may run under this model's parallel
    /// communicator: the exact-batching lane with exact verify rows on a
    /// qwen4_exp pair speaking the v2 protocol, MTP (not DFlash).
    pub(in crate::model) fn batched_verify_under_comm(&self) -> bool {
        self.levers.qwen4exp_batch_fast
            && self.levers.qwen4exp_exact_verify
            && self.ep_protocol_v2
            && self.multi_rank_protocol_active()
            && self.dflash_hidden_save.is_none()
    }

    /// Head: announce a batched verify of `seqs` (`ks[i]` rows each, rows in
    /// `tokens` seq-major) so the worker runs the same forward. No-op without
    /// a multi-rank protocol.
    pub(in crate::model) fn ep_broadcast_verify_batch(
        &self,
        tokens: &[u32],
        ks: &[usize],
        seqs: &[&mut SequenceState],
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        ensure!(
            self.batched_verify_under_comm(),
            "batched verify under a parallel communicator needs \
             ATLAS_QWEN4EXP_BATCH_FAST=1 + ATLAS_QWEN4EXP_EXACT_VERIFY=1 + ATLAS_EP_PROTOCOL=v2"
        );
        let seq_ids: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
        let ks32: Vec<u32> = ks.iter().map(|&k| k as u32).collect();
        self.ep_broadcast_seq_and_cmd(0, EP_CMD_VERIFY_BATCH, true)?;
        self.ep_broadcast_u32(seqs.len() as u32)?;
        self.ep_broadcast_tokens(&seq_ids)?;
        self.ep_broadcast_tokens(&ks32)?;
        self.ep_broadcast_tokens(tokens)?;
        Ok(())
    }

    /// Head: the drafts each sequence of the last batched verify accepted,
    /// in batch order. No-op without a multi-rank protocol.
    pub(in crate::model) fn ep_broadcast_verify_verdicts_impl(
        &self,
        accepted: &[u32],
    ) -> Result<()> {
        if !self.multi_rank_protocol_active() {
            return Ok(());
        }
        self.ep_broadcast_tokens(accepted)?;
        Ok(())
    }

    /// Worker side of [`EP_CMD_VERIFY_BATCH`].
    pub(in crate::model) fn ep_worker_verify_batch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        let n = self.ep_broadcast_u32(0)? as usize;
        // One sequence: a lone window past its own 4-row verify (`verify_rows.rs`).
        ensure!(
            (1..=slots.len()).contains(&n),
            "EP batched verify: {n} sequences for {} slots",
            slots.len()
        );
        let seq_ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let ks: Vec<usize> = self
            .ep_broadcast_tokens(&vec![0u32; n])?
            .into_iter()
            .map(|k| k as usize)
            .collect();
        let r_total: usize = ks.iter().sum();
        ensure!(
            // 1 = a decode row; `decode_verify_batched_dispatch` holds the
            // batch to this rank's own row-count gate.
            ks.iter().all(|k| (1..=32).contains(k)) && r_total <= super::verify_e2::VERIFY_ROW_CAP,
            "EP batched verify: rows {ks:?}"
        );
        let tokens = self.ep_broadcast_tokens(&vec![0u32; r_total])?;

        let mut refs = self.ep_worker_slot_refs(&seq_ids, slots)?;
        let stream = self.gpu.default_stream();
        self.wait_secondary_dispatch()?;
        self.ssm_pool.require_verify_rollback_supported()?;
        for (seq, &k) in refs.iter_mut().zip(&ks) {
            self.mark_gdn_deferred_commit(seq, k)?;
        }
        self.decode_verify_batched_dispatch(&tokens, &ks, &mut refs, stream)?;

        let accepted = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let mut row = 0usize;
        for ((seq, &k), &na) in refs.iter_mut().zip(&ks).zip(&accepted) {
            let window = &tokens[row..row + k];
            row += k;
            ensure!(
                (na as usize) < k,
                "EP batched verify: {na} drafts accepted of a {k}-row window"
            );
            self.ep_worker_apply_verdict(seq, window, na as usize)?;
        }
        Ok(true)
    }

    /// `&mut` to the addressed slots, in `seq_ids` order (bounds and
    /// duplicates checked before any state is touched).
    pub(in crate::model) fn ep_worker_slot_refs<'a>(
        &self,
        seq_ids: &[u32],
        slots: &'a mut [Option<SequenceState>],
    ) -> Result<Vec<&'a mut SequenceState>> {
        let mut seen = std::collections::HashSet::new();
        for &id in seq_ids {
            ensure!(
                (id as usize) < slots.len() && seen.insert(id),
                "EP batch: seq_id {id} out of range or repeated ({} slots)",
                slots.len()
            );
        }
        let mut by_slot: Vec<(usize, &'a mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.as_mut().map(|s| (i, s)))
            .collect();
        let mut refs = Vec::with_capacity(seq_ids.len());
        for &id in seq_ids {
            let pos = by_slot
                .iter()
                .position(|(i, _)| *i == id as usize)
                .ok_or_else(|| anyhow::anyhow!("EP batch: slot {id} not allocated"))?;
            refs.push(by_slot.swap_remove(pos).1);
        }
        Ok(refs)
    }
}
