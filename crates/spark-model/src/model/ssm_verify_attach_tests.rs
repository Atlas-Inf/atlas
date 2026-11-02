// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;

fn pool(has_mtp: bool) -> SsmStatePool {
    SsmStatePool::new(
        &ModelConfig::qwen3_next_80b_nvfp4(),
        2,
        has_mtp,
        if has_mtp { 4 } else { 0 },
        3,
        false,
        crate::ssm_reserve::SsmRollbackMode::Snapshot,
        false,
        &MockGpuBackend::new(),
    )
    .unwrap()
}

fn fresh() -> SsmLayerState {
    SsmLayerState {
        h_state: DevicePtr::NULL,
        conv_state: DevicePtr::NULL,
        h_state_checkpoint: None,
        conv_state_checkpoint: None,
        h_state_intermediates: Vec::new(),
        conv_state_intermediates: Vec::new(),
        kda_records: DevicePtr::NULL,
        gdn_commit_qkv: DevicePtr(0),
        gdn_commit_gb: DevicePtr(0),
        gdn_commit_pending: false,
        gdn_fuse_n: DevicePtr::NULL,
        h_is_f16: false,
        h_prefill_stage: None,
        ple: None,
    }
}

// An EP worker owns no proposer but runs the distributed verify: whatever
// the pool allocated must reach its layer states (GLM-5.3 EP2 regression:
// "KDA verify needs K-1 h and K conv intermediates (h=0, conv=0, K=8)").
#[test]
fn a_rank_without_a_proposer_attaches_the_pools_rollback_buffers() {
    let pool = pool(true);
    let mut state = fresh();
    pool.attach_verify_rollback(1, 0, &mut state);
    assert!(state.h_state_checkpoint.is_some());
    assert!(state.conv_state_checkpoint.is_some());
    assert_eq!(state.conv_state_intermediates.len(), 4);
    assert_eq!(state.h_state_intermediates.len(), pool.h_snapshot_count(0));
    assert!(!state.h_state_intermediates.is_empty());
}

#[test]
fn a_pool_without_rollback_buffers_attaches_nothing() {
    let pool = pool(false);
    let mut state = fresh();
    pool.attach_verify_rollback(1, 0, &mut state);
    assert!(state.h_state_checkpoint.is_none());
    assert!(state.conv_state_checkpoint.is_none());
    assert!(state.h_state_intermediates.is_empty());
    assert!(state.conv_state_intermediates.is_empty());
    assert_eq!(state.kda_records, DevicePtr::NULL);
}
