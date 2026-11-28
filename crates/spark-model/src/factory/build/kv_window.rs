// SPDX-License-Identifier: AGPL-3.0-only

//! Window-bounded KV for the sliding-window layers
//! (`spark_runtime::kv_cache::window`): which models take it, the pool it
//! needs, and the primary blocks the rest of the budget buys.
//!
//! Inkling 2 is the model whose attention reads window layers through the
//! window map (`layers::inkling2`), so it is the one that takes it. The pool
//! holds every live sequence's window and one step's rows, one forward's
//! batch of new rows, and pinned snapshot windows: up to an eighth of the KV
//! budget (at least [`MIN_PINS`]). A long prompt snapshots every few thousand
//! tokens, and a resume needs its snapshot's window, so pins are what keep
//! resume points across interleaved sessions.

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::kv_cache::{KvCacheConfig, WindowPlan};

/// Fewest snapshot windows kept, whatever the budget.
const MIN_PINS: usize = 8;
/// Pinned windows take at most this fraction (1/n) of the KV budget.
const PIN_SHARE: usize = 8;

/// What else the cache was asked for, any of which keeps full-length pools.
#[derive(Clone, Copy, Debug)]
pub(super) struct Exclusions {
    pub(super) high_speed_swap: bool,
    pub(super) disk_tier: bool,
    pub(super) multi_rank: bool,
    pub(super) glm_plan: bool,
}

/// The window plan for `config`, or `None` (full-length pools, as before).
/// Its pins are left to [`primary_blocks`], which knows the budget.
pub(super) fn plan(
    config: &ModelConfig,
    kv: &KvCacheConfig,
    excluded: Exclusions,
    max_batch_size: usize,
    max_batch_tokens: usize,
) -> Option<WindowPlan> {
    if config.model_type != "inkling2_loop"
        || config.sliding_window == 0
        || config.sliding_attention_layers.is_empty()
    {
        return None;
    }
    let Exclusions {
        high_speed_swap,
        disk_tier,
        multi_rank,
        glm_plan,
    } = excluded;
    if high_speed_swap || disk_tier || multi_rank || glm_plan {
        tracing::warn!(
            "KV window off ({excluded:?}): the {} sliding-window layers keep a row for \
             every token of context",
            config.sliding_attention_layers.len()
        );
        return None;
    }
    let window = config.sliding_window as usize;
    let blocks = WindowPlan::size_blocks(
        window,
        kv.block_size,
        max_batch_size,
        crate::layers::inkling2::layer::MAX_VERIFY_ROWS,
        max_batch_tokens,
        0,
    );
    Some(WindowPlan {
        layers: config.sliding_attention_layers.clone(),
        window,
        blocks,
        pins: 0,
    })
}

/// Pin `plan`'s share of `budget` bytes, then the primary blocks the rest
/// buys beside its window pools, with `aux_per_token` bytes riding on each
/// primary token.
pub(super) fn primary_blocks(
    plan: &mut WindowPlan,
    kv: &KvCacheConfig,
    budget: usize,
    aux_per_token: usize,
) -> Result<usize> {
    let pin_blocks = WindowPlan::span_blocks(plan.window, kv.block_size, 0);
    let pin_bytes = pin_blocks * plan.local_block_bytes(kv);
    plan.blocks -= plan.pins * pin_blocks;
    plan.pins = (budget / PIN_SHARE / pin_bytes).max(MIN_PINS);
    plan.blocks += plan.pins * pin_blocks;
    let window_bytes = plan.blocks * plan.local_block_bytes(kv);
    ensure!(
        window_bytes < budget,
        "KV window pools need {:.1} GiB of a {:.1} GiB KV budget",
        window_bytes as f64 / f64::from(1u32 << 30),
        budget as f64 / f64::from(1u32 << 30)
    );
    // each primary block also carries its map entry
    let per_block = plan.primary_block_bytes(kv) + 4 + aux_per_token * kv.block_size;
    let blocks = (budget - window_bytes) / per_block;
    tracing::info!(
        "KV window: {} of {} layers keep the last {} tokens in {} blocks ({:.2} GiB, {} \
         snapshot pins); the other layers' {:.0} KiB a token buy {} blocks",
        plan.layers.len(),
        kv.num_layers,
        plan.window,
        plan.blocks,
        window_bytes as f64 / f64::from(1u32 << 30),
        plan.pins,
        plan.primary_block_bytes(kv) as f64 / kv.block_size as f64 / 1024.0,
        blocks
    );
    Ok(blocks)
}

#[cfg(test)]
#[path = "kv_window_tests.rs"]
mod tests;
