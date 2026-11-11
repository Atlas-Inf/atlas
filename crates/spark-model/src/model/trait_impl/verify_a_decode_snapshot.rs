// SPDX-License-Identifier: AGPL-3.0-only

//! Phase-C decode-time SSM boundary snapshot save/restore (the
//! decode-rollback ring), split out of `verify_a.rs`.

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Phase-C decode-time boundary snapshot save.
    ///
    /// Copies the sequence's live SSM state (the active `SsmStatePool`
    /// slot `seq.slot_idx`) into the decode-rollback ring slot
    /// `(seq.slot_idx, ring_slot)` of [`SsmSnapshotPool`]. Reuses the
    /// same D2D copy primitive Marconi and MTP verify use (SSOT). The
    /// copies run on the default stream so they are ordered after the
    /// decode that produced this boundary token and before any later
    /// decode that would overwrite the pool slot.
    pub(in crate::model::trait_impl) fn save_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("save_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.save_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }

    /// Phase-C decode-time boundary snapshot restore.
    ///
    /// Inverse of [`Self::save_decode_ssm_snapshot_dispatch`]: copies the
    /// ring snapshot `(seq.slot_idx, ring_slot)` back into the live
    /// `SsmStatePool` slot, undoing every recurrent update the dropped
    /// degenerate tail applied.
    pub(in crate::model::trait_impl) fn restore_decode_ssm_snapshot_dispatch(
        &self,
        seq: &SequenceState,
        ring_slot: usize,
    ) -> Result<()> {
        if !self.ssm_snapshots.decode_rollback_enabled() {
            anyhow::bail!("restore_decode_ssm_snapshot: decode-rollback region not allocated");
        }
        let stream = self.gpu.default_stream();
        self.ssm_snapshots.restore_decode(
            seq.slot_idx,
            ring_slot,
            &self.ssm_pool,
            self.gpu.as_ref(),
            stream,
        )
    }
}
