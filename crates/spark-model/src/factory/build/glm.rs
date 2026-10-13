// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5 build steps for `build_model`, split out of `build.rs` for the
//! file-size cap: expert-TP gate, weight caches, arena consumers, the
//! MLA/semantic-index cache plan and the cross-rank KV block agreement.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheConfig, KvCacheDtype, SparseIndexCacheConfig, TailSlotPlan};
use spark_runtime::weights::WeightStore;

use crate::layer::TransformerLayer;
use crate::layers::moe::SharedFp8Reserve;
use crate::model::glm_cache_plan::{GlmCachePlan, GlmMlaShape};

pub(super) fn ensure_expert_tp_drafter(
    config: &ModelConfig,
    use_speculative: bool,
    no_dflash: bool,
) -> Result<()> {
    anyhow::ensure!(
        !(config.expert_tp && use_speculative && no_dflash),
        "ATLAS_GLM_EXPERT_TP=1 serves with the DFlash drafter only: sliced routed \
         experts have no whole-expert MTP path"
    );
    Ok(())
}

/// Shared-FP8 and dense GLM weight caches after layer load, then the
/// post-load shared-cache reserve check.
#[allow(clippy::too_many_arguments)]
pub(super) fn init_weight_caches(
    config: &ModelConfig,
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
    shared_cache_reserve: Option<SharedFp8Reserve>,
    max_batch_tokens: usize,
    max_seq_len: usize,
    kv_block_size: usize,
    max_batch_size: usize,
    inference_reserve: usize,
) -> Result<()> {
    super::super::glm_shared_cache::initialize(config, store, gpu, layers, shared_cache_reserve)?;
    super::super::glm_dense_cache::initialize(
        config,
        gpu,
        layers,
        max_batch_tokens,
        max_seq_len,
        kv_block_size,
        max_batch_size,
        inference_reserve,
    )?;
    let _ = crate::layers::moe::validate_shared_fp8_cache_factory_reserve(
        config,
        gpu,
        max_batch_tokens,
        max_seq_len,
        kv_block_size,
        max_batch_size,
        inference_reserve,
        false,
    )?;
    Ok(())
}

/// GLM consumers of the fresh buffer arena.
#[allow(clippy::too_many_arguments)]
pub(super) fn init_arena_consumers(
    config: &ModelConfig,
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    layers: &mut [Box<dyn TransformerLayer>],
    buffers: &mut BufferArena,
    kv_dtype: KvCacheDtype,
    max_seq_len: usize,
    max_batch_tokens: usize,
    lm_head: DevicePtr,
    bf16_head: bool,
) -> Result<()> {
    // The BF16 dense (<=2048) and native (<=32768) GLM prefill kernels read an
    // `fp8_g128` owner's latents through a dequantized BF16 view; allocated
    // before KV sizing so the budget sees it.
    if kv_dtype == KvCacheDtype::Fp8G128 && config.model_type == "glm5_next" {
        buffers.attach_glm_latent_scratch(
            crate::layers::ops::GLM_LATENT_BF16_VIEW_TOKENS.min(max_seq_len.next_multiple_of(16)),
            gpu,
        )?;
    }
    crate::layers::ops::validate_glm_sparse_decode_split_scratch(
        &config.model_type,
        buffers.expert_gate_out(),
        buffers.sizes().expert_gate_out,
        &[],
    )?;
    // Both ranks initialize HC's TF32 library path before state construction
    // and the actual-free KV snapshot; the optional helper reuses dead scratch.
    super::super::glm_hc_prewarm::initialize(config, gpu, buffers, max_batch_tokens)?;
    crate::model::prepare_glm_head_mxfp8(config, gpu, lm_head, bf16_head)?;
    crate::model::prepare_glm_verify_masks(config, gpu, buffers)?;
    crate::layers::moe::bind_resident_btile_arenas(config, store, gpu, layers, buffers)?;
    Ok(())
}

pub(super) fn cache_shape(
    config: &ModelConfig,
    kv_dtype: KvCacheDtype,
    layer_dtypes: &[KvCacheDtype],
) -> Result<Option<GlmMlaShape>> {
    let glm_cache_shape = GlmMlaShape::for_model(
        &config.model_type,
        config.kv_lora_rank,
        config.qk_rope_head_dim,
    )?;
    anyhow::ensure!(
        glm_cache_shape.is_some()
            || (kv_dtype != KvCacheDtype::Fp8G128
                && !layer_dtypes.contains(&KvCacheDtype::Fp8G128)),
        "--kv-cache-dtype fp8_g128 stores the GLM NoPE-512 latent; {} has none",
        config.model_type
    );
    Ok(glm_cache_shape)
}

/// Returns `(sparse_index, tail_slots, glm_cache_plan)`. `lag_bounded` is
/// speculative decode without HSS whose prefix-cache resumes, if any, land on
/// whole blocks.
#[allow(clippy::type_complexity)]
pub(super) fn cache_plan(
    config: &ModelConfig,
    kv_config: &KvCacheConfig,
    glm_cache_shape: Option<GlmMlaShape>,
    lag_bounded: bool,
    max_batch_tokens: usize,
    kv_block_size: usize,
    max_batch_size: usize,
) -> Result<(
    Option<SparseIndexCacheConfig>,
    Option<TailSlotPlan>,
    Option<GlmCachePlan>,
)> {
    // GLM-5.3 selects one semantic-index key for each four-token pool. Keep
    // those keys under the same physical block lifecycle as MLA K/V so prefix
    // sharing and recycling cannot leave the two histories out of sync.
    // BF16 is the correctness baseline; the cache API also models scaled FP8
    // for the production-memory follow-up.
    let sparse_index = glm_cache_shape
        .map(|shape| shape.bf16_index(config.index_kpool, config.index_head_dim))
        .transpose()?;
    // Raw index tails are only read to finalize a sequence's newest pools, so
    // lend them to trailing blocks instead of carrying one in every block.
    // Kept per block where a resume is not lag-bounded: prefix sharing, HSS,
    // and non-speculative decode (whose rollback ring rewinds arbitrarily).
    // ATLAS_GLM_INDEX_TAIL_SLOTS=0 restores per-block tails.
    let tail_slots = (sparse_index.is_some()
        && lag_bounded
        && std::env::var("ATLAS_GLM_INDEX_TAIL_SLOTS").as_deref() != Ok("0"))
    .then(|| TailSlotPlan {
        lag_blocks: max_batch_tokens.div_ceil(kv_block_size) + 2,
        sequences: max_batch_size + 1,
    });
    // GLM-5 writes one NoPE latent to both cache sides, so V aliases K.
    let glm_cache_plan = glm_cache_shape
        .map(|shape| {
            let plan = GlmCachePlan::new(shape, kv_config, sparse_index)?.aliased_v(kv_config);
            anyhow::Ok(match (sparse_index, tail_slots) {
                (Some(index), Some(slots)) => plan.slotted_tails(kv_config, index, slots),
                _ => plan,
            })
        })
        .transpose()?;
    Ok((sparse_index, tail_slots, glm_cache_plan))
}

pub(super) fn agree_kv_blocks(
    comm: Option<&dyn spark_comm::CommBackend>,
    gpu: &dyn GpuBackend,
    mut num_kv_blocks: usize,
    glm_cache_plan: Option<GlmCachePlan>,
    spill_tier: u32,
) -> Result<usize> {
    // Test knob: cap the pool (e.g. to force prefix-cache eviction and the
    // optional cache tiers without starving the rest of the budget).
    if let Some(cap) = std::env::var("ATLAS_KV_MAX_BLOCKS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&cap| cap > 0 && cap < num_kv_blocks)
    {
        tracing::info!("KV cache: ATLAS_KV_MAX_BLOCKS caps {num_kv_blocks} blocks at {cap}");
        num_kv_blocks = cap;
    }
    // Every rank mirrors the head's sequences into its own pool, so a rank
    // with less headroom (e.g. no drafter, different co-tenants) must not
    // size a pool the others cannot back: all ranks take the minimum.
    if let Some(comm) = comm.filter(|c| c.world_size() > 1) {
        let agreed = min_across_ranks(comm, gpu, num_kv_blocks, spill_tier)?;
        if agreed < num_kv_blocks {
            tracing::info!(
                "KV cache: rank {} fits {num_kv_blocks} blocks; all ranks agree on {agreed}",
                comm.rank()
            );
        }
        num_kv_blocks = agreed;
    }
    if let Some(plan) = glm_cache_plan {
        plan.bytes_for_blocks(num_kv_blocks)?;
    }
    Ok(num_kv_blocks)
}

/// The minimum block count over all ranks. The upper half of the gathered
/// word carries the SSM backing mode so ranks agree on restore collectives.
fn min_across_ranks(
    comm: &dyn spark_comm::CommBackend,
    gpu: &dyn GpuBackend,
    value: usize,
    spill_tier: u32,
) -> Result<usize> {
    let blocks = u32::try_from(value)?;
    let word = u64::from(blocks) | u64::from(spill_tier) << 32;
    let (tiers, blocks): (Vec<u32>, Vec<u32>) =
        crate::model::startup_parity::gather_words(comm, gpu, &[word])?
            .into_iter()
            .map(|word| ((word >> 32) as u32, word as u32))
            .unzip();
    super::ssm_spill::verify_ranks(spill_tier, &tiers)?;
    blocks
        .into_iter()
        .min()
        .map(|blocks| blocks as usize)
        .ok_or_else(|| anyhow::anyhow!("empty rank gather"))
}

#[cfg(test)]
#[path = "glm_tests.rs"]
mod tests;
