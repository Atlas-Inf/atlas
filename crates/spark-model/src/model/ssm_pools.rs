// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-construction bundle for the SSM state/snapshot pools and the flags
//! that size them.
//!
//! `TransformerModel::new` used to allocate these pools *after* the KV
//! cache had been sized from `gpu_memory_utilization × total − used_so_far`.
//! Everything allocated inside `new()` is invisible to that budget, so at
//! long `--max-seq-len` the pools (Marconi snapshot slots, verify
//! intermediates, decode-rollback ring) overflowed the physical remainder
//! and boot died inside `cuMemAlloc` (Atlas-Inf/atlas#61). `build_model`
//! now constructs this bundle BEFORE the `gpu.free_memory()` snapshot so
//! the pools land in `used_so_far` and the KV pool shrinks to fit — the
//! same positional-budgeting rule the LoRA adapter load documents.

use anyhow::{Context, Result};
use std::sync::Arc;

use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::ssm_tier::{SnapshotBlobStore, SpillHome};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;

/// SSM pools plus the sizing flags `TransformerModel::new` still needs
/// downstream (verify-stash row counts, capture-buffer capacity).
pub struct SsmPools {
    pub(crate) pool: Arc<SsmStatePool>,
    pub(crate) snapshots: SsmSnapshotPool,
    pub(crate) tier_store: Option<Arc<dyn SnapshotBlobStore>>,
    /// Where `tier_store` keeps its spills (`None` without a tier).
    pub(crate) tier_home: Option<SpillHome>,
    pub(crate) has_mtp: bool,
    /// Native GLM MTP was validated from rank-symmetric marker/head weights.
    /// This remains distinct from the generic MTP and DFlash flags: native
    /// GLM uses the canonical target verifier without setting DFlash env.
    pub(crate) native_glm_mtp: bool,
    pub(crate) num_intermediates: usize,
    pub(crate) dflash_kgamma: usize,
}

impl SsmPools {
    /// Host bytes the snapshot spill tier takes after KV sizing (0 without a
    /// tier) — see `SpillHome::lazy_host_bytes`.
    pub(crate) fn tier_lazy_host_bytes(&self) -> usize {
        self.tier_home.map_or(0, |home| {
            home.lazy_host_bytes(self.snapshots.spill_blob_bytes())
        })
    }

    /// Compatibility constructor for callers that have no external native
    /// MTP capability to report.
    pub(crate) fn new(
        config: &ModelConfig,
        max_batch_size: usize,
        self_speculative: bool,
        use_speculative: bool,
        mtp_weights_empty: bool,
        draft_lm_head_nvfp4_present: bool,
        num_drafts: usize,
        ssm_cache_slots: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        Self::new_with_native(
            config,
            max_batch_size,
            self_speculative,
            use_speculative,
            false,
            mtp_weights_empty,
            draft_lm_head_nvfp4_present,
            num_drafts,
            ssm_cache_slots,
            gpu,
        )
    }

    /// Build the pools. Must run before the KV-cache budget snapshot in
    /// `factory::build_model` — see the module doc. `native_mtp` is a
    /// rank-symmetric capability bit for architectures whose proposer body is
    /// loaded outside the generic `MtpWeights` bundle (currently native GLM).
    pub(crate) fn new_with_native(
        config: &ModelConfig,
        max_batch_size: usize,
        self_speculative: bool,
        use_speculative: bool,
        native_mtp: bool,
        mtp_weights_empty: bool,
        draft_lm_head_nvfp4_present: bool,
        num_drafts: usize,
        ssm_cache_slots: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        // num_intermediates = K, the verify-width ceiling. The CONV pools
        // allocate K snapshots per slot; the H pools allocate K-1 (index
        // K-1 is never written or read — see ssm_reserve) and tier by slot.
        // For MTP K=2/3/4 verify: K = num_drafts + 1.
        // For DFlash/DSpark, verify rows are `[last_token, draft_1, ..., draft_K]`:
        // exactly `num_drafts + 1`. The anchor bonus is not a draft row.
        let dflash_kgamma = super::dspark_pool::dflash_verify_rows(
            !config.dflash_capture_layers.is_empty(),
            num_drafts,
        )?;
        // DFlash needs the SSM verify pools regardless of MTP weight presence
        // or lm_head quantization — its K=γ verify path checkpoints SSM state
        // for partial-accept rollback. Force `has_mtp` on whenever DFlash is
        // active so the checkpoint pools exist.
        // qwen4_exp installs its MTP proposer AFTER construction (its module is
        // a reused trunk layer, not the Qwen-shaped `MtpWeights`), so
        // `mtp_weights` is empty here even though drafting WILL run. Its 36 GDN
        // layers still need the verify/checkpoint pools: without them the very
        // first draft indexes `conv_intermediate_pools[0]` on a zero-length
        // Vec and panics. DeepSeek-V4 reaches the proposer the same way but has
        // no SSM layers, so it never exercised this.
        let external_mtp_module = use_speculative && config.model_type == "qwen4_exp";
        let has_mtp = self_speculative
            || (use_speculative && !mtp_weights_empty && draft_lm_head_nvfp4_present)
            || external_mtp_module
            || native_mtp
            || dflash_kgamma > 0;
        let num_intermediates = if has_mtp {
            (num_drafts + 1).max(dflash_kgamma)
        } else {
            0
        };
        let pool = Arc::new(SsmStatePool::new(
            config,
            max_batch_size,
            has_mtp,
            num_intermediates,
            num_drafts,
            // Stage-3 f16-SIZED h pools. No CLI surface publishes this and
            // preflight refuses it until prefill narrowing lands, so it is
            // false on every serveable config today.
            crate::layers::qwen3_ssm::ssm_h_f16_pool_enabled(),
            // `--ssm-rollback-mode` (EXPERIMENTAL replay scaffold; default
            // snapshot, published by spark-server's serve_flags).
            crate::ssm_reserve::ssm_rollback_mode(),
            // ATLAS_GDN_DEFERRED_COMMIT — build-time lever, same class as
            // the rollback-mode global (pool geometry is fixed at boot).
            crate::layers::ops::ModelLevers::from_env().gdn_deferred_commit,
            gpu,
        )?);

        // Fail fast if an SSM tier was requested (`ATLAS_SSM_TIER`) on a model
        // with no recurrent state — a tier request there was previously a
        // silent no-op. No-op when the tier is unset (default path).
        super::ssm_tier::ensure_ssm_tier_capability(config)?;

        // SSM snapshot pool: Marconi prefix-cache slots + Phase-C
        // decode-rollback ring. The ring's ONLY writer (scheduler
        // snapshot_boundary_if_ssm) and reader (content-loop
        // rollback_to_boundary) live on the PLAIN decode path — the
        // speculative path does its rejection rollback through the verify
        // snapshot, never this ring. The ring-depth decision (env overrides +
        // speculative/watchdog skip) is SSOT'd in
        // `crate::ssm_reserve::decode_rollback_ring_slots` — spark-server's
        // `preflight_reserve` calls the SAME helper, so the GPU reservation
        // and this allocation cannot drift.
        let ring =
            crate::ssm_reserve::decode_rollback_ring_slots(pool.num_ssm_layers, use_speculative);
        if let Some(reason) = ring.skip_reason {
            let per_seq = (pool.h_bytes + pool.conv_bytes)
                * pool.num_ssm_layers
                * atlas_kernels::DECODE_ROLLBACK_RING_SLOTS;
            tracing::info!(
                "SSM decode-rollback ring: SKIPPED ({}) — the ring's save/rollback \
                 path only runs on plain decode with watchdogs enabled. Saves {:.1} GB \
                 ({} seqs x {} slots x full SSM blob). If plain-decode loop re-steer is \
                 ever reached it fail-opens to decline; ATLAS_SSM_DECODE_RING=1 \
                 force-restores the ring.",
                reason,
                (per_seq * max_batch_size) as f64 / 1e9,
                max_batch_size,
                atlas_kernels::DECODE_ROLLBACK_RING_SLOTS,
            );
        }
        // The pools are deliberately out of `preflight_reserve` (see the
        // module doc): they allocate BEFORE the KV budget's free_memory()
        // snapshot, so nothing checks that they fit before `cuMemAlloc`
        // dies with NV_ERR_NO_MEMORY and the process is killed with no error
        // line (plain decode at 64 seqs: ~81 GB of decode-rollback ring). Size the
        // same inputs the pool is about to use and fail fast with remedies.
        let hidden_bytes = config.hidden_size * 2;
        let marconi_bytes = ssm_cache_slots
            .checked_mul(pool.num_ssm_layers)
            .and_then(|s| s.checked_mul(pool.h_bytes + pool.conv_bytes))
            .and_then(|s| s.checked_add(ssm_cache_slots.checked_mul(hidden_bytes)?))
            .context("SSM snapshot pool size overflow")?;
        let ring_bytes = ring
            .slots
            .checked_mul(max_batch_size)
            .and_then(|s| s.checked_mul(pool.num_ssm_layers))
            .and_then(|s| s.checked_mul(pool.h_bytes + pool.conv_bytes))
            .context("SSM decode-rollback ring size overflow")?;
        snapshot_pool_fits(
            marconi_bytes,
            ring_bytes,
            gpu.free_memory()?,
            SNAPSHOT_POOL_OOM_GUARD,
        )?;
        let snapshots = SsmSnapshotPool::new(
            ssm_cache_slots,
            pool.h_bytes,
            pool.conv_bytes,
            pool.num_ssm_layers,
            ring.slots,
            max_batch_size,
            // Last-token hidden snapshot: post-final-norm `norm_output` is
            // BF16 (`hidden_size` elements). Used to emit exact-hit logits
            // without re-running the last token through the SSM layers.
            hidden_bytes,
            gpu,
        )?;
        // Optional SSM snapshot spill tier. `None` (default) keeps the reclaim
        // drop path byte-identical; blob sizing tracks the pool's spill layout.
        let (tier_store, tier_home) = super::impl_a1_init::build_ssm_tier_store(
            config,
            snapshots.spill_blob_bytes(),
            pool.num_ssm_layers,
        )?
        .unzip();
        Ok(Self {
            pool,
            snapshots,
            tier_store,
            tier_home,
            has_mtp,
            native_glm_mtp: native_mtp && config.model_type == "glm5_next",
            num_intermediates,
            dflash_kgamma,
        })
    }
}

/// Slack the snapshot-pool fit check keeps between the pools and the OOM
/// kill: the KV budget sizes itself off the free memory measured after
/// these allocations, so a pool that leaves ~0 free would boot-fail one
/// step later anyway.
const SNAPSHOT_POOL_OOM_GUARD: usize = 2 << 30;

/// Does the SSM snapshot pool fit the currently-free device memory?
/// `bail!`s with the split that overflowed and the operator remedies —
/// the decode-rollback ring exists only for plain decode's watchdog
/// re-steer, so speculation sidesteps it entirely.
fn snapshot_pool_fits(
    marconi_bytes: usize,
    ring_bytes: usize,
    free: usize,
    guard: usize,
) -> Result<()> {
    let need = marconi_bytes
        .checked_add(ring_bytes)
        .context("SSM snapshot pool size overflow")?;
    let budget = free.saturating_sub(guard);
    if need <= budget {
        return Ok(());
    }
    anyhow::bail!(
        "SSM snapshot pools do not fit: need {need_gb:.1} GB (Marconi prefix slots {marconi_gb:.1} GB + decode-rollback ring {ring_gb:.1} GB) against {free_gb:.1} GB free (guard {guard_gb:.1} GB). The decode ring serves only PLAIN decode's watchdog rollback — run with --speculative or --dflash, lower --max-batch-size, or shrink the ring with ATLAS_SSM_DECODE_RING=0 / ATLAS_DISABLE_WATCHDOGS=1.",
        need_gb = need as f64 / 1e9,
        marconi_gb = marconi_bytes as f64 / 1e9,
        ring_gb = ring_bytes as f64 / 1e9,
        free_gb = free as f64 / 1e9,
        guard_gb = guard as f64 / 1e9,
    )
}

#[cfg(test)]
#[path = "ssm_pools_tests.rs"]
mod tests;
