// SPDX-License-Identifier: AGPL-3.0-only

//! Speculative-decode buffer allocations hoisted from `TransformerModel::new`:
//! the GLM owner-batched long-context verify stage and the MTP prompt-hidden
//! capture. Each helper mirrors its former inline block in `new()` 1:1.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::layers::ops;

pub(super) fn alloc_glm_long_stage(
    config: &ModelConfig,
    has_mtp: bool,
    gpu: &dyn GpuBackend,
) -> Result<Option<crate::layer::glm_long_owner::GlmLongStage>> {
    // Owner-batched long-context verify stage (4 owners x 8 rows): ~13 MB,
    // allocated before KV sizing so the pool accounts for it.
    let glm_long_stage = if (has_mtp || crate::speculative::glm_repair_policy::dflash_enabled())
        && config.model_type == "glm5_next"
        && crate::layer::glm_long_owner::enabled()?
    {
        Some(crate::layer::glm_long_owner::GlmLongStage::alloc(
            gpu,
            crate::layer::glm_long_owner::RowBytes::new(
                config.hidden_size,
                config.hc_mult,
                config.vocab_size,
            ),
        )?)
    } else {
        None
    };
    Ok(glm_long_stage)
}

/// Returns `(mtp_arena_context, mtp_prefill_hidden)`.
pub(super) fn alloc_mtp_prefill_hidden(
    config: &ModelConfig,
    max_seq_len: usize,
    has_mtp: bool,
    dflash_kgamma: usize,
    mtp_quant: crate::layers::MtpQuantization,
    levers: &ops::ModelLevers,
    gpu: &dyn GpuBackend,
) -> Result<(usize, DevicePtr)> {
    // Prompt hidden capture buffer, [mtp_arena_context, hidden_size] BF16 —
    // 335 MB at 32k/h=5120. Backs BOTH halves of the drafter-context
    // feature (see `crate::model::drafter_context`); NULL here disables
    // prefill AND carry, since the carry path reads this buffer.
    //
    // Three conditions, all necessary: MTP must be active, the feature must
    // not be killed, and the head must be a precision the batched prefill
    // can actually run at — an NVFP4/FP8 MTP head would allocate this and
    // never write it.
    // The repaired GLM long-context lane indexes a bounded 32K arena even
    // when the native lane serves a larger context; the capture below and
    // every stored capacity must quote that same arena. `arena_context`
    // is the SSOT for the bound, so the private-cache quote
    // (`PrivateStoragePlan`) cannot disagree with this allocation.
    let mtp_arena_context = crate::speculative::glm_repair_policy::arena_context(
        &config.model_type,
        crate::speculative::glm_repair_policy::enabled()
            && crate::speculative::glm_repair_policy::long_context_enabled(),
        max_seq_len,
    );
    // A DFlash head never prefills from this buffer (its context comes
    // from the multi-layer capture below), so it is not allocated.
    let mtp_prefill_hidden = if has_mtp
        && dflash_kgamma == 0
        && mtp_quant.supports_drafter_prefill()
        && crate::layers::mtp_drafter_prefill_enabled(levers)
    {
        let bytes = mtp_arena_context
            .checked_mul(config.hidden_size)
            .and_then(|n| n.checked_mul(2))
            .context("MTP drafter context capture reserve overflow")?;
        tracing::info!(
            "MTP drafter context: allocating {:.0} MB prompt-hidden capture \
             ({} x {} BF16)",
            bytes as f64 / 1e6,
            mtp_arena_context,
            config.hidden_size,
        );
        gpu.alloc(bytes)?
    } else {
        if has_mtp
            && !mtp_quant.supports_drafter_prefill()
            && crate::layers::mtp_drafter_prefill_enabled(levers)
        {
            tracing::info!(
                "MTP drafter context: INACTIVE — the batched drafter prefill \
                 needs a BF16 MTP head (--mtp-quantization bf16); this head is \
                 {mtp_quant:?}. No prompt-hidden capture allocated.",
            );
        }
        DevicePtr::NULL
    };
    Ok((mtp_arena_context, mtp_prefill_hidden))
}
