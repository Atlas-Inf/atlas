// SPDX-License-Identifier: AGPL-3.0-only

//! The REAL `DraftProposer::free_state` cases on `BlockDiffusionDraftHead`, on the
//! zeroed-head fixtures of `free_state_tests` (see that module's doc).

use super::CTX_ACC_POOL;
use super::free_state_tests::{hold_two_blocks, live_state, lock_and_drain_pool, owner, zero_head};
use super::lifecycle::*;
use crate::speculative::DraftProposer;
use spark_runtime::gpu::mock::MockGpuBackend;

#[test]
fn real_free_state_success_path_reclaims_all_resources() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);

    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 6);
    let allocs_before = gpu.alloc_count();

    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("same-owner free succeeds");

    // KV blocks returned, buffers freed, watermarks reset, descriptor retired.
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 8);
    assert!(boxed.block_table.is_empty());
    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert!(boxed.block_table_dev.is_none());
    assert_eq!(
        (boxed.seq_len, boxed.ctx_len, boxed.ctx_committed),
        (0, 0, 0)
    );
    assert_eq!(
        boxed.lifecycle.as_ref().unwrap().status(),
        CaptureStatus::Retired
    );
    // Block table dev freed to the backend; the accumulator moved to the
    // reuse pool (still a live allocation).
    assert_eq!(gpu.alloc_count(), allocs_before - 1);
    assert_eq!(CTX_ACC_POOL.lock().len(), 1);
}

#[test]
fn real_free_state_second_call_is_idempotent_success() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);

    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);

    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("first same-owner free succeeds");
    // The B3 idempotence contract: a same-owner SECOND cleanup is a success
    // (ownership-only terminal validation), not a Retired error.
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("second same-owner free is idempotent success");
}

#[test]
fn real_free_state_owner_mismatch_reclaims_then_propagates_error() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);
    let stale = owner(3, 76);

    // A pooled graph for THIS owner must remain in the pool: the error arm
    // reclaims state resources but must NOT destroy owner-keyed graphs.
    head.propose_graphs.lock().insert(
        DflashGraphIdentity::new(own, 0x10, 0x20, 0x30, 0).unwrap(),
        vec![spark_runtime::gpu::GraphHandle(0xAA)],
    );

    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 6);
    let allocs_before = gpu.alloc_count();

    let err = head
        .free_state(&gpu, Some(stale), boxed.as_mut())
        .expect_err("stale owner must be rejected");

    // Original validation error propagates…
    assert!(err.to_string().contains("stale owner"));
    // …AND every state-owned resource was reclaimed first (no leak):
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 8);
    assert!(boxed.block_table.is_empty());
    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert!(boxed.block_table_dev.is_none());
    assert_eq!(gpu.alloc_count(), allocs_before - 2);
    assert_eq!(
        boxed.lifecycle.as_ref().unwrap().status(),
        CaptureStatus::Retired
    );
    // Graph pool untouched in the error arm: entry still present, zero
    // destroy_graph calls.
    assert_eq!(head.propose_graphs.lock().len(), 1);
    assert_eq!(gpu.destroy_graph_count(), 0);
}

#[test]
fn real_free_state_different_live_owner_is_rejected_and_reclaimed() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);
    let other = owner(5, 77);

    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    let allocs_before = gpu.alloc_count();

    let err = head
        .free_state(&gpu, Some(other), boxed.as_mut())
        .expect_err("a different live owner must be rejected");
    assert!(err.to_string().contains("stale owner"));
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 8);
    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert!(boxed.block_table_dev.is_none());
    assert_eq!(gpu.alloc_count(), allocs_before - 2);
    assert_eq!(
        boxed.lifecycle.as_ref().unwrap().status(),
        CaptureStatus::Retired
    );
    assert_eq!(gpu.destroy_graph_count(), 0);
}

#[test]
fn real_free_state_missing_owner_is_rejected_and_reclaimed() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);

    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    let allocs_before = gpu.alloc_count();

    let err = head
        .free_state(&gpu, None, boxed.as_mut())
        .expect_err("missing expected owner must be rejected");
    assert!(err.to_string().contains("expected owner"));
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 8);
    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert!(boxed.block_table_dev.is_none());
    assert_eq!(gpu.alloc_count(), allocs_before - 2);
    assert_eq!(
        boxed.lifecycle.as_ref().unwrap().status(),
        CaptureStatus::Retired
    );
    assert_eq!(gpu.destroy_graph_count(), 0);
}
