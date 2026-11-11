// SPDX-License-Identifier: AGPL-3.0-only

//! KV-budget tail tests (extracted from `kv_budget_tests.rs` under the
//! 500-LoC cap; no logic edits).

use super::*;

/// qwen4_exp at TP2: one KV head of 256 per rank on 12 layers (12 KiB per
/// token) plus the 12 QSA pooled-key carries (hd 128, ratio 4: 64 B each). The pool
/// is sized for both; without aux state the arithmetic is unchanged.
#[test]
fn aux_state_is_charged_per_pool_token() {
    use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, PagedKvCache};
    let kv = KvCacheConfig {
        block_size: 16,
        num_kv_heads: 1,
        head_dim: 256,
        num_layers: 12,
        dtype: KvCacheDtype::Bf16,
        layer_dtypes: vec![],
        layer_dims: vec![],
        cache_blocks_per_seq: None,
    };
    let budget = gib(12.5);
    let plain = PagedKvCache::compute_num_blocks(&kv, budget).unwrap();
    assert_eq!(blocks_with_aux(&kv, budget, 0).unwrap(), plain);
    let with_qsa = blocks_with_aux(&kv, budget, 12 * 64).unwrap();
    assert_eq!(with_qsa, budget / (16 * (12 * 1024 + 12 * 64)));
    assert!(
        with_qsa * 16 >= 1_000_000,
        "12.5 GiB holds 1M tokens and their QSA keys"
    );
}

/// DevLab Strix Halo (2026-10-29, runs 40835733703 / 40836997818): hipMemGetInfo
/// reports the 62.5 GB GTT cap as total while free tracks 124 GB of system RAM,
/// so free (79.4) exceeds total. The budget used to charge 0 GB pre-KV and size a
/// pool that could not fit beside the 27B's weights ("Failed to build model").
#[test]
fn free_above_total_charges_the_self_relative_footprint() {
    let i = Inputs {
        total: gib(62.5),
        free_now: gib(79.4),
        utilization: 0.88,
        reserve: gib(4.2),
        baseline_free: Some(gib(118.7)),
        manual_external: None,
        tracked: None,
    };
    let b = size(&i);
    assert_eq!(b.basis, Basis::Delta);
    // The pre-KV footprint is the 39.3 GB the load took, not 0.
    assert!(
        (b.own as f64 / GIB_F - 39.3).abs() < 0.05,
        "own {}",
        b.own as f64 / GIB_F
    );
    // The pool fits in what is left of the device after the footprint and reserve.
    assert!(b.bytes > 0);
    assert!(
        b.own + i.reserve + b.bytes <= i.total,
        "pool {} does not fit",
        b.bytes as f64 / GIB_F
    );
}

/// A device whose free reading fits inside its total is sized exactly as before.
#[test]
fn free_within_total_is_unchanged() {
    let i = start(118.0, 15.0, driver(100.0));
    let b = size(&i);
    assert_eq!(b.used_raw, i.total - i.free_now);
}
