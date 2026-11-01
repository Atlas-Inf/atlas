// SPDX-License-Identifier: AGPL-3.0-only

//! The GLM DFlash2 arm of the prefill ctx capture — split out of
//! `impl_b3_dflash.rs` purely for the 500-LoC cap (no behaviour change).

use anyhow::Result;

use super::types::TransformerModel;

impl TransformerModel {
    /// The `glm5_next` branch of `try_dflash_prefill_capture_layer`: the
    /// warm-ctx capture plan and its own-row handling, which only exists on
    /// the GLM drafter.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dflash_glm_prefill_capture(
        &self,
        seq: &mut crate::traits::SequenceState,
        slot_idx: usize,
        chunk_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(dstate) = seq.proposer_state.as_mut().and_then(|ps| {
            ps.as_any_mut()
                .downcast_mut::<crate::layers::DflashProposerState>()
        }) else {
            return Ok(());
        };
        if dstate.ctx_hidden_acc.0 == 0
            && let Some(ref proposer) = self.proposer
        {
            proposer.acquire_ctx_acc(self.gpu.as_ref(), dstate)?;
        }
        let h = self.config.hidden_size;
        let bf16 = 2usize;
        let n_capture = self.dflash_capture_layers.len();
        let acc_base = dstate.ctx_hidden_acc;
        let own_row = dstate.own_row;
        // A sequence-parallel chunk keeps only this rank's rows (rank 0: the
        // upper half, compacted at row 0) in the highway.
        let sp_row0 = crate::layers::glm_sp::current().map_or(0, |sp| sp.row0);
        // Enntity/atlas 5573f4d5 + d3953211 (`ATLAS_DFLASH_WARM_CTX`):
        // the accumulator holds the LAST `max_ctx_len` prompt positions
        // (slot 0 = position `window_start`); `capture_plan` decides the
        // anchoring under the switch — legacy reproduces the old math
        // byte for byte.
        let pass = crate::layers::dflash_head::warm_ctx::Pass {
            tokens_len: seq.tokens.len(),
            prompt_len: seq.prompt_len,
            start: chunk_start,
            rows: proc_count,
            sp_row0,
            branch_at: seq.pc_branch_at,
        };
        let plan = crate::layers::dflash_head::warm_ctx::capture_plan(
            dstate.warm_ctx,
            dstate.max_ctx_len,
            dstate.ctx_floor,
            &pass,
        );
        // Non-legacy `ATLAS_DFLASH_FIRST_APPEND`: keep this pass's last row —
        // after the final chunk, the last prompt position — in the sequence's
        // own spare row, whatever the window below keeps.
        if let Some(own_row) = own_row
            && let Some(row) = crate::layers::dflash_head::own_capture_row(proc_count, sp_row0)
        {
            self.dflash_capture_rows(
                row,
                1,
                own_row.offset(slot_idx * h * bf16),
                n_capture * h * bf16,
                stream,
            )?;
        }
        let Some(plan) = plan else {
            return Ok(());
        };
        dstate.note_capture(&plan);
        self.dflash_capture_rows(
            plan.src_row(&pass),
            plan.rows(),
            acc_base.offset(plan.dst_slot() * n_capture * h * bf16 + slot_idx * h * bf16),
            n_capture * h * bf16,
            stream,
        )
    }
    pub(super) fn update_dflash_ctx_len_after_prefill(
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
            dstate.seed_prefill_ctx(seq.tokens.len(), chunk_start + proc_count, seq.seq_len);
        }
        Ok(())
    }
}
