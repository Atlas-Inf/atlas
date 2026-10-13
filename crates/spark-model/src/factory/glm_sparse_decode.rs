// SPDX-License-Identifier: AGPL-3.0-only
//! Startup admission for the GLM sparse-decode tensor-core / split kernels on
//! the DFlash long-context lane.
use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{gpu::GpuBackend, kv_cache::KvCacheDtype};

pub(super) struct BuildPolicy<'a> {
    pub config: &'a ModelConfig,
    pub self_speculative: bool,
    pub context: usize,
    pub block_size: usize,
    pub kv_dtype: KvCacheDtype,
    pub layer_dtypes: &'a [KvCacheDtype],
    /// `--dflash` on the GLM DFlash verify lane (`ATLAS_GLM_DFLASH=1`).
    pub dflash: bool,
}

/// The sparse verifier has a bounded indexed domain. The serving context may
/// be larger because requests beyond that domain are admitted to the native
/// lane with speculation disabled; startup must still validate the verifier
/// against its own bounded domain.
pub(super) fn repair_context(context: usize) -> usize {
    context.min(crate::speculative::glm_repair_policy::MAX_LONG_CONTEXT)
}

impl BuildPolicy<'_> {
    fn validate(&self, dflash_lane_enabled: bool, long_context: bool) -> Result<()> {
        let dflash_lane = self.dflash && !self.self_speculative && dflash_lane_enabled;
        ensure!(
            self.config.model_type == "glm5_next"
                && self.config.tp_world_size == 2
                && self.config.ep_world_size == 2
                // Serving topology has already converted this to local heads.
                && self.config.num_attention_heads == 32
                && (2049..=crate::speculative::glm_repair_policy::max_long_context())
                    .contains(&self.context)
                && self.block_size == 16
                && matches!(self.kv_dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
                && self.layer_dtypes.iter().all(|d| *d == self.kv_dtype)
                && dflash_lane
                && long_context,
            "GLM sparse decode TC requires the GLM DFlash lane, TP2/EP2 local32 heads and BF16 or fp8_g128 block16 caches"
        );
        Ok(())
    }
}

pub(super) fn initialize(gpu: &dyn GpuBackend, policy: BuildPolicy<'_>) -> Result<()> {
    let split = crate::layers::ops::glm_sparse_decode_split_enabled(&policy.config.model_type)?;
    let tc = crate::layers::ops::glm_sparse_decode_tc_enabled(&policy.config.model_type)?;
    ensure!(
        !split || tc,
        "GLM split requires ATLAS_GLM_SPARSE_DECODE_TC=1"
    );
    if !tc {
        return Ok(());
    }
    policy.validate(
        crate::speculative::glm_repair_policy::dflash_enabled(),
        crate::speculative::glm_repair_policy::long_context_enabled(),
    )?;
    crate::layers::ops::initialize_glm_sparse_decode_tc(gpu, policy.config)?;
    tracing::info!(
        "GLM sparse decode TC initialized before KV sizing; repaired K3 rows1, BF16 identical K/V"
    );
    if split {
        crate::layers::ops::initialize_glm_sparse_decode_split(gpu, policy.config)?;
        tracing::info!(
            "GLM sparse decode split initialized before KV sizing; fixed S8, repaired K3 rows1"
        );
    }
    Ok(())
}

#[cfg(test)]
#[path = "glm_sparse_decode_tests.rs"]
mod tests;
