// SPDX-License-Identifier: AGPL-3.0-only

//! Target-only, once-at-load shared FP8 cache (existing K32 arithmetic).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::GpuBackend;
use std::sync::atomic::AtomicU8;

#[path = "shared_fp8_prefill_rows.rs"]
pub(super) mod prefill_rows;

pub(super) fn prefill_2048_enabled() -> Result<bool> {
    match std::env::var("ATLAS_GLM_PREFILL_2048").ok().as_deref() {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("ATLAS_GLM_PREFILL_2048 must be 0 or 1"),
    }
}

pub(super) fn prefill_slab_budget() -> Result<Option<usize>> {
    let two = prefill_2048_enabled()?;
    let four = match std::env::var("ATLAS_GLM_PREFILL_4096").ok().as_deref() {
        None | Some("0") => false,
        Some("1") => true,
        _ => anyhow::bail!("ATLAS_GLM_PREFILL_4096 must be 0 or 1"),
    };
    ensure!(
        !(two && four),
        "ATLAS_GLM_PREFILL_2048 and ATLAS_GLM_PREFILL_4096 cannot both be enabled"
    );
    Ok(if four {
        Some(4096)
    } else if two {
        Some(2048)
    } else {
        None
    })
}

pub(super) const WEIGHT_BYTES: usize = 2048 * 4096;
const LAYER_BYTES: usize = 3 * WEIGHT_BYTES;

/// Factory-owned future arena/inference obligations throughout the cache pass.
#[derive(Clone, Copy, Debug)]
pub(crate) struct SharedFp8Reserve {
    protected: usize,
}
impl SharedFp8Reserve {
    pub(crate) fn new(arena: usize, inference: usize) -> Result<Self> {
        let future = arena
            .checked_add(inference)
            .ok_or_else(|| anyhow::anyhow!("shared FP8 future reserve overflow"))?;
        Ok(Self {
            protected: future.max(4 * 1024 * 1024 * 1024),
        })
    }
    pub(crate) fn remaining_bytes(self, layers: usize) -> Result<usize> {
        layers
            .checked_mul(LAYER_BYTES)
            .ok_or_else(|| anyhow::anyhow!("shared FP8 cache byte count overflow"))
    }
    pub(crate) fn check(self, free: usize, remaining: usize) -> Result<()> {
        let required = remaining
            .checked_add(self.protected)
            .ok_or_else(|| anyhow::anyhow!("shared FP8 deferred reserve overflow"))?;
        ensure!(
            free >= required,
            "shared FP8 deferred reserve needs {required} bytes, have {free}"
        );
        tracing::info!(
            free,
            remaining,
            protected = self.protected,
            required,
            "GLM shared FP8 deferred reserve checked before allocation"
        );
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct SharedFp8CacheState {
    pub layer: Option<usize>,
    pub verify: bool,
    pub checked: AtomicU8,
}

pub(super) fn flags() -> Result<(bool, bool)> {
    let read = |key| -> Result<bool> {
        match std::env::var(key).ok().as_deref() {
            None | Some("0") => Ok(false),
            Some("1") => Ok(true),
            _ => anyhow::bail!("{key} must be 0 or 1"),
        }
    };
    let enabled = read("ATLAS_GLM_TARGET_SHARED_FP8")?;
    let verify = read("ATLAS_GLM_TARGET_SHARED_FP8_VERIFY")?;
    ensure!(
        !verify || enabled,
        "shared FP8 VERIFY requires cache enabled"
    );
    Ok((enabled, verify))
}

pub fn validate_shared_fp8_cache_graphs(model_type: &str, use_graphs: bool) -> Result<()> {
    let (_, verify) = flags()?;
    ensure!(
        !(model_type == "glm5_next" && verify && use_graphs),
        "shared FP8 VERIFY requires eager execution"
    );
    Ok(())
}

pub fn validate_shared_fp8_cache_profile(
    config: &ModelConfig,
    prefill_budget: usize,
    has_adapters: bool,
) -> Result<()> {
    let (enabled, verify) = flags()?;
    let extended = prefill_slab_budget()?;
    ensure!(
        extended.is_none() || enabled,
        "ATLAS_GLM_PREFILL_2048/4096 requires shared FP8 cache"
    );
    if !enabled {
        return Ok(());
    }
    validate_config(config)?;
    ensure!(!has_adapters, "shared FP8 cache excludes adapters");
    prefill_rows::validate_budget(prefill_budget, extended.unwrap_or(1024))?;
    for key in ["ATLAS_GLM_K5_BATCHED_SHARED", "ATLAS_NVFP4_MMQ_MOE"] {
        let value = std::env::var(key).unwrap_or_default();
        ensure!(
            value != "1" && !value.eq_ignore_ascii_case("true"),
            "shared FP8 cache excludes {key}"
        );
    }
    validate_verify_overlap(
        verify,
        std::env::var("ATLAS_MOE_SHARED_REDUCE_OVERLAP")
            .ok()
            .as_deref(),
    )?;
    Ok(())
}

fn validate_verify_overlap(verify: bool, value: Option<&str>) -> Result<()> {
    ensure!(
        !verify || value == Some("0"),
        "shared FP8 VERIFY requires explicit ATLAS_MOE_SHARED_REDUCE_OVERLAP=0"
    );
    Ok(())
}

fn validate_config(c: &ModelConfig) -> Result<()> {
    ensure!(
        c.model_type == "glm5_next"
            && c.hidden_size == 4096
            && c.moe_intermediate_size == 2048
            && c.shared_expert_intermediate_size == 2048
            && c.num_hidden_layers == 45
            && c.num_experts == 288
            && c.num_experts_per_tok == 8
            && c.tp_world_size == 2
            && c.ep_world_size == 2
            && c.adapter_max_rank == 0,
        "shared FP8 cache requires exact target GLM TP2/EP2 native geometry without adapters"
    );
    ensure!(
        c.mlp_only_layers == [0, 1, 2],
        "shared FP8 cache requires target dense layers 0,1,2"
    );
    Ok(())
}

/// A logical check only: allocations subsequently enter actual-free KV
/// accounting, so the cache is never subtracted a second time there.
#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_shared_fp8_cache_factory_reserve(
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    max_batch_tokens: usize,
    max_seq_len: usize,
    block_size: usize,
    max_batch_size: usize,
    inference_reserve: usize,
    before_layers: bool,
) -> Result<Option<SharedFp8Reserve>> {
    if !flags()?.0 {
        return Ok(None);
    }
    validate_config(config)?;
    let arena = spark_runtime::buffers::BufferSizes::from_config(
        config,
        max_batch_tokens,
        max_seq_len,
        block_size,
        max_batch_size,
    )
    .total_bytes();
    // Cache allocation is deferred until checkpoint replacement, MTP and
    // LM-head setup finish. At either outer boundary only A+I is unallocated;
    // the deferred pass separately protects its full remaining cache budget.
    let cache = 0;
    let required = factory_required(arena, inference_reserve, cache)?;
    let free = gpu.free_memory()?;
    ensure!(
        free >= required,
        "shared FP8 factory reserve needs {required} bytes, have {free}; before_layers={before_layers}"
    );
    tracing::info!(
        before_layers,
        required,
        free,
        arena,
        inference_reserve,
        cache,
        "GLM shared FP8 factory reserve passed"
    );
    Ok(Some(SharedFp8Reserve::new(arena, inference_reserve)?))
}

fn factory_required(arena: usize, inference: usize, cache: usize) -> Result<usize> {
    arena
        .checked_add(inference)
        .and_then(|n| n.checked_add(cache))
        .ok_or_else(|| anyhow::anyhow!("shared FP8 factory reserve overflow"))
}

#[derive(Debug)]
pub(super) struct CachePlan {
    pub total_bytes: usize,
    pub remaining_bytes: usize,
}

pub(super) fn target_plan(
    config: &ModelConfig,
    layer: usize,
    target: bool,
    enabled: bool,
) -> Result<Option<CachePlan>> {
    if !target || !enabled {
        return Ok(None);
    }
    validate_config(config)?;
    ensure!(
        (3..45).contains(&layer),
        "shared FP8 cache requires target MoE layer ordinal"
    );
    Ok(Some(CachePlan {
        total_bytes: 42 * LAYER_BYTES,
        remaining_bytes: (45 - layer) * LAYER_BYTES,
    }))
}

/// Publish only the returned complete triple. Every allocated temporary is
/// released on failure, including when another cleanup itself fails.
pub(super) fn transaction(
    mut alloc: impl FnMut() -> Result<DevicePtr>,
    mut free: impl FnMut(DevicePtr) -> Result<()>,
    mut convert: impl FnMut(usize, DevicePtr) -> Result<()>,
) -> Result<[DevicePtr; 3]> {
    let mut owned = Vec::with_capacity(3);
    let result = (|| {
        for i in 0..3 {
            let ptr = alloc()?;
            owned.push(ptr);
            convert(i, ptr)?;
        }
        Ok([owned[0], owned[1], owned[2]])
    })();
    if let Err(error) = result {
        let mut failures = Vec::new();
        for ptr in owned {
            if let Err(e) = free(ptr) {
                failures.push(e.to_string());
            }
        }
        anyhow::bail!(
            "shared FP8 cache transaction failed: {error:#}; cleanup failures: {failures:?}"
        );
    }
    result
}

#[cfg(test)]
#[path = "shared_fp8_cache_tests.rs"]
mod tests;
