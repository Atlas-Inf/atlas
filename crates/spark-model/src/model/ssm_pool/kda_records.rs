// SPDX-License-Identifier: AGPL-3.0-only

//! GLM KDA fold-record accessors for [`SsmStatePool`].
//!
//! GLM KDA fold records (`--ssm-rollback-mode records`), one region
//! per SSM layer of `(mtp_slots + 1) × num_intermediates` rows of
//! `kda_record_row_bytes`; replaces the H snapshot pools, which are then
//! not allocated. Empty otherwise.
//!
//! Prior art: fold-record rollback is vLLM's RecoverSSM
//! (<https://github.com/vllm-project/vllm/pull/51855>, ZJY0516 with
//! benchislett), which grew out of ReplaySSM
//! (<https://github.com/vllm-project/vllm/pull/48018>, Johnny-Liou, Dao AI
//! Lab, NVIDIA), as RiNGSiDE ships it for GLM-5.3 (othexmr, `--use-replayssm`);
//! Apache-2.0. Our record layout and CUDA kernels (`kda.cu`,
//! `kda_recurrent_bf16_verify_rec_owners`, `kda_commit_records`) extend the
//! snapshot-verify kernel; see docs/glm-prior-art.md.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;

use super::SsmStatePool;

impl SsmStatePool {
    /// H snapshots actually held for `slot`: its tiered count, or 0 under
    /// KDA records (per-step scratch with nothing to reset or migrate).
    pub(in crate::model) fn h_snapshot_count(&self, slot: usize) -> usize {
        if self.h_intermediate_pools.is_empty() {
            0
        } else {
            self.h_inter_count(slot)
        }
    }

    /// Slot `slot`'s KDA fold records in SSM layer `ssm_layer_idx`
    /// (`num_intermediates` rows), NULL unless KDA records are on.
    pub(in crate::model) fn kda_records(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.kda_record_pools
            .get(ssm_layer_idx)
            .map_or(DevicePtr::NULL, |pool| {
                pool.offset(
                    self.mtp_slot(slot) * self.num_intermediates * self.kda_record_row_bytes,
                )
            })
    }
}

/// Validate the KDA-records rollback mode for `SsmStatePool::new` and size it:
/// returns `(kda_record_row_bytes, h_inter_held)`.
pub(super) fn plan(
    records: bool,
    config: &ModelConfig,
    h_stored_bytes: usize,
    h_bytes: usize,
    h_inter_total: usize,
) -> Result<(usize, usize)> {
    anyhow::ensure!(
        !records || (config.model_type == "glm5_next" && h_stored_bytes == h_bytes),
        "--ssm-rollback-mode records serves GLM-5 KDA with an FP32 h state only"
    );
    let kda_record_row_bytes = if records {
        crate::ssm_reserve::kda_record_row_bytes(h_bytes)
    } else {
        0
    };
    // H snapshot units the pools really hold (0 under KDA records).
    let h_inter_held = if records { 0 } else { h_inter_total };
    Ok((kda_record_row_bytes, h_inter_held))
}
