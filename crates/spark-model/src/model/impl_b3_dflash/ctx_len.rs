// SPDX-License-Identifier: AGPL-3.0-only
//! The sequence's drafter context after a prefill.

use anyhow::Result;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// After prefill completes (last chunk on the chunked path), set the
    /// seq's DFlash `ctx_len` to the full captured window — carried rows
    /// plus every position this prefill covered — and slide to the newest
    /// half-window if the window exceeds the retention cap.
    pub(in crate::model) fn update_dflash_ctx_len_after_prefill(
        &self,
        seq: &mut crate::traits::SequenceState,
        chunk_start: usize,
        proc_count: usize,
    ) -> Result<()> {
        if self.dflash_capture_layers.is_empty() {
            return Ok(());
        }
        if let Some(ref c) = self.comm
            && c.rank() != 0
        {
            return Ok(());
        }
        if let Some(ps) = seq.proposer_state.as_mut()
            && let Some(dstate) = ps
                .as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        {
            if self.config.model_type == "glm5_next" {
                dstate.seed_prefill_ctx(seq.tokens.len(), chunk_start + proc_count, seq.seq_len);
                return Ok(());
            }
            if dstate.max_ctx_len == 0 {
                return Ok(());
            }
            dstate.first_append_at = Some(seq.seq_len);
            let chunk_end = chunk_start + proc_count;
            // Pure/idempotent: rows [0..base) are the adopted carry (their
            // carried positions stay), rows [base..ctx_len) are this
            // prefill's captures at positions origin..chunk_end — same
            // layout the capture cursor wrote (base + chunk_start - origin).
            // A repeat call recomputes the same result.
            let base = dstate.ctx_prefill_base;
            let origin = dstate.ctx_prefill_origin.unwrap_or(chunk_start);
            dstate.ctx_len = base.saturating_add(chunk_end.saturating_sub(origin));
            dstate.ctx_positions.truncate(base);
            dstate
                .ctx_positions
                .extend((origin..chunk_end).map(|i| i as i32));
            // Retention slide — same keep-NEWEST-half policy as
            // commit_ctx/dflash_serial_ctx_append: drop_n >= keep so the
            // single D2D copy's src/dst ranges can never overlap. Absolute
            // positions survive the drain; committed K/V is re-precomputed
            // chunk-wise on the next propose.
            if dstate.ctx_len > dstate.max_ctx_len {
                let keep = dstate.max_ctx_len / 2;
                // Bound by PHYSICAL rows: the buffer is sized to
                // ctx_acc_rows (== max_ctx_len) and an all-cached-chunks
                // turn can push formula ctx_len one row past it.
                let phys = dstate.ctx_len.min(dstate.ctx_acc_rows);
                let drop_n = phys - keep;
                let slot = dstate.ctx_slot_bytes;
                let src = dstate.ctx_hidden_acc.offset(drop_n * slot);
                let stream = self.gpu.default_stream();
                self.gpu
                    .copy_d2d_async(src, dstate.ctx_hidden_acc, keep * slot, stream)?;
                let pos_drop = drop_n.min(dstate.ctx_positions.len());
                dstate.ctx_positions.drain(..pos_drop);
                dstate.ctx_len = keep;
                dstate.ctx_committed = 0;
                tracing::info!(
                    "DFlash ctx prefill watermark: slid ctx window \
                     (dropped {drop_n} oldest, keep {keep})",
                );
            }
        }
        Ok(())
    }
}
