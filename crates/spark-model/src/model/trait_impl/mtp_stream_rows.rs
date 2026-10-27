// SPDX-License-Identifier: AGPL-3.0-only

//! Stream-highway stash rows -> `hc_streams`: the input of a pre-mixer
//! drafter (qwen4_exp), for the per-sequence and the batched propose alike.

use anyhow::Result;

use super::super::types::TransformerModel;
// Ported from Enntity/atlas 166e2b1 (AGPL-3.0-only).
use crate::traits::SequenceState;

impl TransformerModel {
    /// Stream-highway stash slot `slots[j]` -> `hc_streams` row `j`: where a
    /// pre-mixer drafter (qwen4_exp) reads sequence `j`'s input, alone (row 0)
    /// or batched. No-op without a highway or its stash.
    pub(super) fn restore_stream_rows_from_stash(&self, slots: &[usize]) -> Result<()> {
        let Some(row_bytes) = self.mtp_stream_row_bytes() else {
            return Ok(());
        };
        if self.verify_stream_stash.is_null() {
            return Ok(());
        }
        anyhow::ensure!(
            slots.len() * row_bytes <= self.buffers.sizes().hc_streams,
            "restore_stream_rows_from_stash: {} rows exceed hc_streams",
            slots.len()
        );
        let stream = self.gpu.default_stream();
        for (j, &slot) in slots.iter().enumerate() {
            self.gpu.copy_d2d_async(
                self.verify_stream_stash.offset(slot * row_bytes),
                self.buffers.hc_streams().offset(j * row_bytes),
                row_bytes,
                stream,
            )?;
        }
        Ok(())
    }

    /// `Model::stage_mtp_absorb`: highway rows `row ..` of the live forward
    /// to `seq`'s proposer. No-op without a proposer, a highway or a state.
    pub(super) fn stage_mtp_absorb_dispatch(
        &self,
        seq: &mut SequenceState,
        row: usize,
        accepted: &[u32],
    ) -> Result<()> {
        let (Some(proposer), Some(row_bytes)) =
            (self.proposer.as_ref(), self.mtp_stream_row_bytes())
        else {
            return Ok(());
        };
        let Some(state) = seq.proposer_state.as_mut() else {
            return Ok(());
        };
        anyhow::ensure!(
            (row + accepted.len()) * row_bytes <= self.buffers.sizes().hc_streams,
            "stage_mtp_absorb: rows {row}+{} exceed hc_streams",
            accepted.len()
        );
        proposer.stage_absorb(
            state.as_mut(),
            self.buffers.hc_streams().offset(row * row_bytes),
            accepted,
            self.gpu.as_ref(),
            self.gpu.default_stream(),
        )
    }
}
