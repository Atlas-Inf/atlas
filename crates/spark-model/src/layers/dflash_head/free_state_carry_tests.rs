// SPDX-License-Identifier: AGPL-3.0-only

//! Integration tests for the ctx-carry half of the real
//! `DraftProposer::free_state`/`carry_dflash_ctx`/`adopt_dflash_ctx` paths —
//! split out of `free_state_tests.rs` (the file crossed the 500-LoC CI cap
//! when the carry legs grew these cases). Helpers are `pub(super)` in the
//! sibling module; `use` surface is identical.

use super::DflashGraphIdentity;
use super::free_state_tests::*;
use crate::speculative::DraftProposer;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

#[test]
fn reclaim_seam_free_failure_retains_pointers() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let own = owner(3, 77);
    let mut dstate = live_state(&gpu, own);
    let kv_cache = parking_lot::Mutex::new(
        PagedKvCache::new(
            KvCacheConfig {
                block_size: 16,
                num_kv_heads: 2,
                head_dim: 64,
                num_layers: 8,
                dtype: KvCacheDtype::Bf16,
                layer_dtypes: vec![],
                layer_dims: vec![],
                cache_blocks_per_seq: None,
            },
            8,
            &gpu,
        )
        .unwrap(),
    );
    for _ in 0..2 {
        let block = kv_cache.lock().try_alloc_block().expect("block available");
        dstate.block_table.push(block);
    }

    gpu.fail_next_free();
    gpu.fail_next_free();
    dstate.reclaim_on_owner_failure(&gpu, &kv_cache);

    // Both failed frees keep their handles (retryable)…
    assert_ne!(dstate.ctx_hidden_acc.0, 0);
    assert!(dstate.block_table_dev.is_some());
    // …blocks are still returned, and a plain second reclaim retries both
    // frees (injections are one-shot).
    assert!(dstate.block_table.is_empty());
    dstate.reclaim_on_owner_failure(&gpu, &kv_cache);
    assert_eq!(dstate.ctx_hidden_acc.0, 0);
    assert!(dstate.block_table_dev.is_none());
}

#[test]
fn real_free_state_block_table_free_failure_restores_handle_for_retry() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);

    let mut boxed = live_state(&gpu, own);

    // Fail the success-path block-table free: the accumulator pools without
    // a free call; the failed dev-table free must restore its handle.
    gpu.fail_next_free();
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("free_state succeeds despite the backend free failure");
    assert!(
        boxed.block_table_dev.is_some(),
        "handle restored, retryable"
    );
    assert_eq!(boxed.ctx_hidden_acc.0, 0);

    // Retry with no further injections releases it.
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("second free retries the failed free");
    assert!(boxed.block_table_dev.is_none());
}

/// Carry → adopt round trip: the finished state's accumulator, paged
/// blocks, and watermarks move into the carry slot, then a later sequence
/// whose prompt extends the carried tokens adopts the prefix portion.
#[test]
fn ctx_carry_round_trip_adopts_prefix() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);
    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    let carried_acc = boxed.ctx_hidden_acc;
    let carried_bt_dev = boxed.block_table_dev;
    // 300-token sequence, ctx fully committed.
    let tokens: Vec<u32> = (0..300).collect();
    assert!(head.carry_dflash_ctx(&gpu, boxed.as_mut(), &tokens));
    // Resources moved out: free_state's guards see an emptied state.
    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert!(boxed.block_table.is_empty());
    assert!(boxed.block_table_dev.is_none());
    // Pool blocks are NOT freed — the carry owns them now.
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 6);

    // New turn: prompt = carried tokens + 50 appended.
    let mut prompt = tokens.clone();
    prompt.extend(300..350);
    let mut fresh = fresh_state(&gpu);
    assert!(head.adopt_dflash_ctx(&gpu, fresh.as_mut(), &prompt, 300));
    // Carried accumulator + device block table installed; watermarks at
    // the common prefix (all 300 match; carried ctx_len=300 adopted).
    assert_eq!(fresh.ctx_hidden_acc.0, carried_acc.0);
    assert_eq!(
        fresh.block_table_dev.map(|p| p.0),
        carried_bt_dev.map(|p| p.0)
    );
    assert_eq!(fresh.block_table.len(), 2);
    assert_eq!(fresh.ctx_committed, 300);
    assert_eq!(fresh.ctx_len, 300);
    assert!(fresh.prefill_done);
}

/// Divergent prefix: the adopt refuses, and every carried resource is
/// released (blocks back to the pool, buffers to the backend).
#[test]
fn ctx_carry_rejects_divergent_prefix_without_leaking() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);
    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    let bt_ptr = boxed.block_table_dev.unwrap().0;
    let acc_ptr = boxed.ctx_hidden_acc.0;
    // A captured graph set under the carried generation — on reject it
    // must be destroyed, not left orphaned in the pool.
    head.propose_graphs.lock().insert(
        DflashGraphIdentity::new(own, bt_ptr, acc_ptr, 0x30, 0).unwrap(),
        vec![spark_runtime::gpu::GraphHandle(0xAA)],
    );
    let tokens: Vec<u32> = (0..300).collect();
    assert!(head.carry_dflash_ctx(&gpu, boxed.as_mut(), &tokens));

    let mut fresh = fresh_state(&gpu);
    let fresh_acc = fresh.ctx_hidden_acc;
    let allocs_before = gpu.alloc_count();
    // Prompt diverges at token 0 — common=0 < MIN_CARRY_TOKENS.
    let prompt: Vec<u32> = (1000..1400).collect();
    assert!(!head.adopt_dflash_ctx(&gpu, fresh.as_mut(), &prompt, 300));
    // Carried blocks returned to the pool; fresh state untouched.
    assert_eq!(head.kv_cache.lock().num_free_blocks(), 8);
    assert_eq!(fresh.ctx_len, 0);
    assert_eq!(fresh.ctx_committed, 0);
    assert!(fresh.block_table.is_empty());
    assert!(!fresh.prefill_done);
    assert_eq!(fresh.ctx_hidden_acc.0, fresh_acc.0);
    // Carried block table dev freed; the carried accumulator moved to the
    // reuse pool (still live). Fresh acc survives untouched.
    assert_eq!(gpu.alloc_count(), allocs_before - 1);
    // Lifted graph destroyed on the reject path.
    assert_eq!(gpu.destroy_graph_count(), 1);
    assert!(head.propose_graphs.lock().is_empty());
}

/// Partial divergence: common prefix 280 of 300 — the adopt truncates
/// watermarks to the shared span; the tail re-precomputes.
#[test]
fn ctx_carry_truncates_at_divergence() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);
    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    let tokens: Vec<u32> = (0..300).collect();
    assert!(head.carry_dflash_ctx(&gpu, boxed.as_mut(), &tokens));

    let mut fresh = fresh_state(&gpu);
    let mut prompt = tokens.clone();
    prompt[280] = 9999; // divergence at index 280
    assert!(head.adopt_dflash_ctx(&gpu, fresh.as_mut(), &prompt, 300));
    // Carried ctx_len=300 > common=280 → truncated at the divergence.
    assert_eq!(fresh.ctx_committed, 280);
    assert_eq!(fresh.ctx_len, 280);
    assert_eq!(fresh.ctx_positions.len(), 280);
}

/// Graphs captured under the old generation ride the carry: lifted before
/// free_state's retire sweep, re-keyed to the adopting generation with
/// identical pointer fields, lane pinned.
#[test]
fn ctx_carry_lifts_and_rekeys_propose_graphs() {
    let _pool_guard = lock_and_drain_pool();
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let old_own = owner(3, 77);
    let new_own = owner(4, 78);
    let mut boxed = live_state(&gpu, old_own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);
    boxed.lane_id = 2;
    let acc_ptr = boxed.ctx_hidden_acc.0;
    let bt_ptr = boxed.block_table_dev.unwrap().0;
    // A captured graph set under the OLD generation, keyed by the very
    // buffers the carry will move (bt dev table + ctx accumulator).
    head.propose_graphs.lock().insert(
        DflashGraphIdentity::new(old_own, bt_ptr, acc_ptr, 0x30, 2).unwrap(),
        vec![
            spark_runtime::gpu::GraphHandle(0xAA),
            spark_runtime::gpu::GraphHandle(0xBB),
        ],
    );

    let tokens: Vec<u32> = (0..300).collect();
    assert!(head.carry_dflash_ctx(&gpu, boxed.as_mut(), &tokens));
    // Lifted out of the pool — free_state's retire sweep finds nothing.
    assert!(head.propose_graphs.lock().is_empty());
    assert_eq!(gpu.destroy_graph_count(), 0);

    let mut prompt = tokens.clone();
    prompt.extend(300..350);
    let mut fresh = fresh_state_with_owner(&gpu, Some(new_own));
    assert!(head.adopt_dflash_ctx(&gpu, fresh.as_mut(), &prompt, 300));

    // Re-keyed under the NEW owner, same pointer tuple.
    let gmap = head.propose_graphs.lock();
    let expected = DflashGraphIdentity::new(new_own, bt_ptr, acc_ptr, 0x30, 2).unwrap();
    assert_eq!(gmap.len(), 1);
    assert_eq!(gmap.get(&expected).unwrap().len(), 2);
    drop(gmap);
    assert_eq!(fresh.lane_id, 2);
    assert_eq!(gpu.destroy_graph_count(), 0);
}
