// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched long verify E9/EA protocol entry points: head announce +
//! traversal, per-owner tail restore, and the worker receive sides.

use super::*;

/// Flag on the E9 width word: one grammar mask per verify row follows the
/// tokens (`ATLAS_GLM_STRICT_OWNER_BATCH`). Only the head reads that switch;
/// a worker learns a masked batch from this bit, so the ranks cannot
/// disagree. It lies above both half-words `encode_width` uses, so an
/// unmasked batch keeps its wire.
pub(super) const MASKED_OWNERS: u32 = 1 << 31;

/// A received E9 width word: its `encode_width` word, and whether row masks
/// follow the tokens.
pub(super) fn split_masked(raw: u32) -> (u32, bool) {
    (raw & !MASKED_OWNERS, raw & MASKED_OWNERS != 0)
}

impl TransformerModel {
    /// Put owner `owner`'s final `rows` verify rows back at arena rows
    /// [0, rows).
    pub(in crate::model) fn glm_long_restore_owner(&self, owner: usize, rows: usize) -> Result<()> {
        ensure!(
            owner_supported(owner, rows),
            "GLM long owner index {owner} x {rows} rows"
        );
        let stage = self
            .glm_long_stage
            .context("GLM long owner stage missing")?;
        stage.copy(
            self.gpu.as_ref(),
            &self.glm_long_final_spans(&stage),
            0,
            owner * rows,
            rows,
            false,
            self.gpu.default_stream(),
        )
    }

    /// Head: announce and run the batched traversal of `seqs.len()` owners
    /// of `rows` rows each (`tokens` owner-major). With `masks` (one
    /// full-vocabulary bitmask per row, owner-major like the tokens) every
    /// rank picks each row under its mask (`glm_verify_masks`).
    pub(in crate::model) fn decode_verify_glm_long_owners_impl(
        &self,
        rows: usize,
        tokens: &[u32],
        seqs: &mut [&mut SequenceState],
        masks: Option<&[u32]>,
    ) -> Result<Vec<u32>> {
        ensure!(
            self.can_batch_glm_long_verify_impl(seqs.len(), rows)
                && tokens.len() == seqs.len() * rows,
            "GLM long owner verify is not available"
        );
        // Validated and uploaded before any word is sent: a refusal here
        // leaves every rank untouched.
        if let Some(masks) = masks {
            ensure!(
                self.glm_verify_logits_argmax_only(),
                "masked owner verify needs the GLM vocab-split head"
            );
            self.upload_row_masks(tokens.len(), masks)?;
        }
        let slots: Vec<u32> = seqs.iter().map(|s| s.slot_idx as u32).collect();
        let flag = if masks.is_some() { MASKED_OWNERS } else { 0 };
        self.ep_broadcast_seq_and_cmd(0, EP_CMD_GLM_LONG_VERIFY, true)?;
        self.ep_broadcast_u32(encode_width(slots.len(), rows) | flag)?;
        self.ep_broadcast_tokens(&slots)?;
        self.ep_broadcast_tokens(tokens)?;
        if masks.is_some() {
            // Only this broadcast follows the flag, so nothing that can fail
            // locally separates the ranks once the worker expects masks.
            self.send_row_masks(tokens.len())?;
        }
        self.glm_long_owner_compute(rows, tokens, seqs)
    }

    /// Head: announce owner `owner`'s tail and restore its `tokens.len()`
    /// rows. The caller then runs the ordinary verdict tail, beginning with
    /// its verdict word.
    pub(in crate::model) fn begin_glm_long_owner_tail_impl(
        &self,
        slot: u32,
        owner: usize,
        tokens: &[u32],
    ) -> Result<()> {
        let rows = tokens.len();
        ensure!(
            owner_supported(owner, rows),
            "GLM long owner tail {owner} x {rows} rows"
        );
        self.ep_broadcast_seq_and_cmd(slot, EP_CMD_GLM_LONG_TAIL, true)?;
        self.ep_broadcast_u32(encode_width(owner, rows))?;
        self.ep_broadcast_tokens(tokens)?;
        self.glm_long_restore_owner(owner, rows)
    }

    /// Worker side of E9.
    pub(in crate::model) fn glm_long_receive_verify(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        // Nothing staged before anything that can fail (`glm_verify_masks`).
        self.verify_row_masks.reset();
        let (word, masked) = split_masked(self.ep_broadcast_u32(0)?);
        let (n, rows) = decode_width(word);
        ensure!(
            owner::width_supported(n, rows) && n <= slots.len(),
            "GLM long owner verify width {n} x {rows} rows"
        );
        let ids = self.ep_broadcast_tokens(&vec![0u32; n])?;
        let tokens = self.ep_broadcast_tokens(&vec![0u32; n * rows])?;
        if masked {
            // Strict structured output: the head's per-row grammar masks,
            // received before any local check below can fail.
            self.send_row_masks(n * rows)?;
        }
        let mut seen = [false; 64];
        for &id in &ids {
            let id = id as usize;
            ensure!(
                id < slots.len() && id < seen.len() && !seen[id],
                "GLM long owner verify slot {id} invalid or repeated"
            );
            seen[id] = true;
        }
        let mut refs: Vec<(usize, &mut SequenceState)> = slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.as_mut().map(|s| (i, s)))
            .collect();
        let mut seqs = Vec::with_capacity(n);
        for &id in &ids {
            let at = refs
                .iter()
                .position(|(i, _)| *i == id as usize)
                .with_context(|| format!("GLM long owner verify slot {id} unallocated"))?;
            seqs.push(refs.swap_remove(at).1);
        }
        self.sync_secondary()?;
        self.glm_long_owner_compute(rows, &tokens, &mut seqs)?;
        Ok(true)
    }

    /// Worker side of EA: restore this owner's rows, then the F5 verdict tail.
    pub(in crate::model) fn glm_long_receive_tail(&self, seq: &mut SequenceState) -> Result<()> {
        let (owner, rows) = decode_width(self.ep_broadcast_u32(0)?);
        ensure!(
            owner_supported(owner, rows),
            "GLM long owner tail {owner} x {rows} rows"
        );
        let tokens = self.ep_broadcast_tokens(&vec![0u32; rows])?;
        self.glm_long_restore_owner(owner, rows)?;
        let accepted = self.ep_worker_verdict()?;
        ensure!(
            accepted < rows,
            "GLM long owner verdict {accepted} for {rows} rows"
        );
        self.ep_worker_apply_verdict(seq, &tokens, accepted)
    }
}
