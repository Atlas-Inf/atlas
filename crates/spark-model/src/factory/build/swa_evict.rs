// SPDX-License-Identifier: AGPL-3.0-only

//! Sliding-window KV eviction (`ATLAS_SWA_EVICT=1`).
//!
//! MiMo-V2.6-Flash's 39 sliding-window layers read only their last 128
//! tokens, yet a block-indexed pool keeps their whole history (~234 KiB per
//! token against ~27 KiB for the 9 full-attention layers). With eviction on,
//! those layers' pools hold a per-sequence ring of slots
//! (`PagedKvCache::new_with_swa_slots`) and the budget buys blocks for the
//! full-attention layers only.

use anyhow::{Result, ensure};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::kv_cache::{KvCacheConfig, SwaSlotPlan};

/// Rows past one prefill chunk a sequence may write before its lagging slots
/// are reclaimed: a speculative verify block plus rollback.
const SPEC_ROWS: usize = 64;

pub(super) fn plan(
    config: &ModelConfig,
    kv: &KvCacheConfig,
    max_batch_tokens: usize,
    max_batch_size: usize,
    prefix_cache_active: bool,
    hss_active: bool,
) -> Result<Option<SwaSlotPlan>> {
    if std::env::var("ATLAS_SWA_EVICT").ok().as_deref() != Some("1") {
        return Ok(None);
    }
    ensure!(
        config.model_type == "mimo_v2",
        "ATLAS_SWA_EVICT=1: only the mimo_v2 attention kernels read slotted \
         sliding-window pools (model_type {})",
        config.model_type
    );
    ensure!(
        !prefix_cache_active,
        "ATLAS_SWA_EVICT=1 drops sliding-window history that prefix reuse needs; \
         run without --enable-prefix-caching"
    );
    ensure!(
        !hss_active,
        "ATLAS_SWA_EVICT=1 is incompatible with --high-speed-swap"
    );
    let layers: Vec<usize> = config
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| matches!(t, LayerType::SlidingAttention))
        .map(|(i, _)| i)
        .collect();
    ensure!(
        !layers.is_empty() && config.sliding_window > 0,
        "ATLAS_SWA_EVICT=1: the model has no sliding-window layers"
    );
    let rows = config.sliding_window as usize + max_batch_tokens + SPEC_ROWS;
    Ok(Some(SwaSlotPlan {
        layers,
        lag_blocks: rows.div_ceil(kv.block_size) + 1,
        sequences: max_batch_size,
    }))
}

/// Blocks `budget` buys once the slot pools are paid for: each block carries
/// only the block-indexed layers (plus `aux_per_token` of outside state).
pub(super) fn blocks(
    plan: &SwaSlotPlan,
    kv: &KvCacheConfig,
    budget: usize,
    aux_per_token: usize,
) -> Result<usize> {
    let pools = plan.pool_bytes(kv);
    ensure!(
        budget > pools,
        "KV budget {budget} B cannot hold the {pools} B sliding-window slot pools"
    );
    let per_block = plan.unslotted_block_bytes(kv) + aux_per_token * kv.block_size;
    ensure!(per_block > 0, "KV cache block size is zero");
    let blocks = (budget - pools) / per_block;
    tracing::info!(
        "SWA eviction: {:.2} GB slot pools ({} slots/layer), {} B/token in the block pool",
        pools as f64 / (1u64 << 30) as f64,
        plan.capacity(),
        per_block / kv.block_size
    );
    Ok(blocks)
}
