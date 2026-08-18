// SPDX-License-Identifier: AGPL-3.0-only

//! Integration tests for the REAL `DraftProposer::free_state` path on
//! `BlockDiffusionDraftHead` (B3 review blockers: the reclaim seam must be
//! reachable through production teardown, and a backend-free failure must
//! stay observable/retryable instead of silently leaking).
//!
//! `BlockDiffusionDraftHead::from_weights` resolves real kernel handles and
//! cannot run on a host test runner, so these tests construct the head via
//! a zeroed literal (all kernel handles `KernelHandle(0)`, no lanes, empty
//! graph pool). `free_state` touches only: `kv_cache`, `propose_graphs`,
//! and the state's own resources — none of which depend on kernel handles —
//! so the zeroed head exercises the production cleanup path faithfully.

use std::collections::HashMap;

use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};

use super::lifecycle::*;
use super::{BlockDiffusionDraftHead, DflashKernels, DflashProposerState, DflashScratch};
use crate::speculative::DraftProposer;
use crate::weight_map::DenseWeight;

fn owner(slot: usize, generation: u64) -> SequenceGeneration {
    SequenceGeneration::new(slot, generation).unwrap()
}

/// Zeroed `DflashScratch`: every device pointer null, no pinned host
/// buffers. `free_state` never touches scratch.
fn zero_scratch() -> DflashScratch {
    DflashScratch {
        stream_buf: DevicePtr(0),
        norm_buf: DevicePtr(0),
        q_buf: DevicePtr(0),
        k_buf: DevicePtr(0),
        v_buf: DevicePtr(0),
        attn_out: DevicePtr(0),
        mlp_intermediate: DevicePtr(0),
        mlp_up: DevicePtr(0),
        stream_acc: DevicePtr(0),
        fc_proj: DevicePtr(0),
        fused_kv_out: DevicePtr(0),
        slot_mapping_dev: DevicePtr(0),
        option_b_indirect_args_dev: DevicePtr(0),
        draft_tokens_host_pinned: Default::default(),
        draft_tokens_event: 0,
        logits: DevicePtr(0),
        draft_tokens_dev: DevicePtr(0),
        markov_prev_dev: DevicePtr(0),
        markov_prev_host_pinned: Default::default(),
        position_ids: DevicePtr(0),
    }
}

fn zero_head() -> BlockDiffusionDraftHead {
    BlockDiffusionDraftHead {
        num_layers: 0,
        hidden_size: 0,
        intermediate_size: 0,
        num_q_heads: 0,
        num_kv_heads: 0,
        head_dim: 0,
        vocab_size: 0,
        draft_vocab_size: 0,
        gamma: 0,
        mask_token_id: 0,
        window_size: None,
        query_causal: true,
        target_layer_ids: Vec::new(),
        target_hidden_size: 0,
        embed_tokens_shared: DevicePtr(0),
        lm_head_shared: DevicePtr(0),
        lm_head_nvfp4: None,
        lm_head_shared_fp8: None,
        hidden_norm: DenseWeight {
            weight: DevicePtr(0),
        },
        norm: DenseWeight {
            weight: DevicePtr(0),
        },
        fc: DenseWeight {
            weight: DevicePtr(0),
        },
        markov_w1: None,
        markov_w2: None,
        markov_rank: 0,
        draft_id_to_target_id: None,
        layers: Vec::new(),
        fused_kv_weight: None,
        kv_cache: parking_lot::Mutex::new(zero_kv_cache()),
        scratch: zero_scratch(),
        batch_capacity: 1,
        batch_query_ids_dev: DevicePtr(0),
        batch_query_embed: DevicePtr(0),
        extra_lanes: Vec::new(),
        lane0_markov_embed: DevicePtr(0),
        lane0_markov_bias: DevicePtr(0),
        kernels: zero_kernels(),
        max_seq_len: 0,
        yarn_inv_freq: DevicePtr(0),
        rope_theta: 0.0,
        rotary_dim: 0,
        rms_norm_eps: 0.0,
        ctx_window: 0,
        propose_graphs: parking_lot::Mutex::new(HashMap::new()),
        next_lane: std::sync::atomic::AtomicUsize::new(0),
        lanes_start_event: 0,
        suppress_graphs: std::sync::atomic::AtomicBool::new(false),
        propose_warmup_count: std::sync::atomic::AtomicUsize::new(0),
        quant: super::DflashQuantization::Bf16,
        startup: super::DsparkStartupExecution::from_env_lenient(),
    }
}

fn zero_kernels() -> DflashKernels {
    let zero = spark_runtime::gpu::KernelHandle(0);
    DflashKernels {
        rms_norm: zero,
        residual_rms_norm: zero,
        dense_gemv: zero,
        dense_gemm: zero,
        w4a16_gemm: zero,
        dense_gemm_pipelined: zero,
        rope_qwen3: zero,
        reshape_cache_fp8: zero,
        reshape_cache_bf16: zero,
        prefill_attn_dflash_fp8: zero,
        prefill_attn_dflash_bf16: zero,
        prefill_attn_dflash_bf16_indirect: zero,
        silu_mul: zero,
        residual_add: zero,
        argmax: zero,
        batched_embed: zero,
        fill_slots: zero,
        prefill_attn: zero,
        quantize_bf16_to_fp8: zero,
        fp8_gemm_n128_row_scaled: zero,
        dense_gemv_fp8w: zero,
        fp8_gemm_n128_row_scaled_m16: zero,
        w4a16_gemv_batch4: zero,
    }
}

fn zero_kv_cache() -> PagedKvCache {
    let gpu = MockGpuBackend::new();
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
    .unwrap()
}

/// A live per-sequence state mirroring `owner_failure_reclaim_frees_state_
/// resources_without_leaking`'s setup but driven through the REAL trait
/// method (`<BlockDiffusionDraftHead as DraftProposer>::free_state`).
fn live_state(gpu: &MockGpuBackend, own: SequenceGeneration) -> Box<DflashProposerState> {
    Box::new(DflashProposerState {
        block_table: Vec::new(),
        seq_len: 12,
        last_num_drafted: 3,
        prefill_done: true,
        ctx_hidden_acc: gpu.alloc(4096).unwrap(),
        ctx_len: 12,
        last_num_accepted: 1,
        skip_next_decode_append: false,
        max_ctx_len: 1024,
        ctx_slot_bytes: 64,
        block_table_dev: Some(gpu.alloc(256).unwrap()),
        ctx_count_drafter: 12,
        max_ctx_count_drafter: 1024,
        ctx_committed: 12,
        ctx_positions: vec![1, 2, 3],
        lane_id: 0,
        lifecycle: Some(CaptureDescriptor::bind(own, 40, 4, 4, 16).unwrap()),
    })
}

/// Grab two real KV pool blocks into the state's block table.
fn hold_two_blocks(state: &mut DflashProposerState, kv: &parking_lot::Mutex<PagedKvCache>) {
    for _ in 0..2 {
        let block = kv.lock().try_alloc_block().expect("block available");
        state.block_table.push(block);
    }
}

#[test]
fn real_free_state_success_path_reclaims_all_resources() {
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
    // Both mock allocations (accumulator + block table) removed.
    assert_eq!(gpu.alloc_count(), allocs_before - 2);
}

#[test]
fn real_free_state_second_call_is_idempotent_success() {
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

#[test]
fn real_free_state_backend_free_failure_retains_pointer_for_retry() {
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);

    let mut boxed = live_state(&gpu, own);
    hold_two_blocks(boxed.as_mut(), &head.kv_cache);

    // Inject failure on the FIRST free call (the ctx accumulator in the
    // success path). The pointer must be RETAINED so a retry can release it.
    gpu.fail_next_free();
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("free_state succeeds despite the backend free failure");

    // Accumulator pointer retained (observable, retryable)…
    assert_ne!(boxed.ctx_hidden_acc.0, 0);
    // …and the retry DOES release it (flag is one-shot).
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("second free retries the accumulator free");
    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn real_free_state_block_table_free_failure_restores_handle_for_retry() {
    let gpu = MockGpuBackend::new();
    let head = zero_head();
    let own = owner(3, 77);

    let mut boxed = live_state(&gpu, own);

    // Fail BOTH success-path frees: the ctx accumulator first, then the
    // device block table. Each failed free must retain/restore its handle.
    gpu.fail_next_free();
    gpu.fail_next_free();
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("free_state succeeds despite the backend free failures");
    assert!(
        boxed.block_table_dev.is_some(),
        "handle restored, retryable"
    );
    assert_ne!(boxed.ctx_hidden_acc.0, 0);

    // Retry with no further injections releases both.
    head.free_state(&gpu, Some(own), boxed.as_mut())
        .expect("second free retries both failed frees");
    assert_eq!(boxed.ctx_hidden_acc.0, 0);
    assert!(boxed.block_table_dev.is_none());
    assert_eq!(gpu.alloc_count(), 0);
}

#[test]
fn reclaim_seam_free_failure_retains_pointers() {
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
