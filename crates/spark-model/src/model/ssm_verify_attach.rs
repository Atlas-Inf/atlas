// SPDX-License-Identifier: AGPL-3.0-only
//! Verify-rollback buffers for a freshly allocated SSM slot.

use super::ssm_pool::SsmStatePool;
use crate::layer::SsmLayerState;

impl SsmStatePool {
    /// Point `state` at `slot`'s checkpoint, intermediate and KDA-record
    /// buffers for SSM layer `ssm_layer_idx`, when this pool allocated them.
    ///
    /// Keyed on the pool, not on whether this rank owns a proposer: under
    /// EP/TP the drafter lives on rank 0 only, but every rank runs the
    /// distributed target verify and needs the rollback buffers its pool
    /// holds. Pool-based fixed addresses stay stable across sequence
    /// lifetimes, so CUDA graphs replay without stale pointers.
    pub(super) fn attach_verify_rollback(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        state: &mut SsmLayerState,
    ) {
        if !self.has_mtp {
            return;
        }
        state.h_state_checkpoint = Some(self.h_checkpoint(ssm_layer_idx, slot));
        state.conv_state_checkpoint = Some(self.conv_checkpoint(ssm_layer_idx, slot));
        // Tiered pools: H count is per-SLOT (h_inter_count), conv count is
        // uniform. The vec lengths are the capacity gates every verify arm
        // checks before writing. KDA records replace the H snapshots when
        // allocated.
        state.kda_records = self.kda_records(ssm_layer_idx, slot);
        state.gdn_commit_qkv = self.commit_qkv(ssm_layer_idx, slot);
        state.gdn_commit_gb = self.commit_gb(ssm_layer_idx, slot);
        state.gdn_fuse_n = self.fuse_word(slot);
        state.h_state_intermediates = (0..self.h_snapshot_count(slot))
            .map(|t| self.h_intermediate(ssm_layer_idx, slot, t))
            .collect();
        state.conv_state_intermediates = (0..self.num_intermediates)
            .map(|t| self.conv_intermediate(ssm_layer_idx, slot, t))
            .collect();
    }
}

#[cfg(test)]
#[path = "ssm_verify_attach_tests.rs"]
mod tests;
