// SPDX-License-Identifier: AGPL-3.0-only

//! Load-time launch guards: the shared-FP8-cache prefill profile and the GLM
//! native tool-call boundary.

use anyhow::Result;
use atlas_core::config::ModelConfig;

use crate::cli;
use crate::main_modules::serve_phases;

/// The prefill budget (resolved here unless preflight already did), validated
/// against the bounded shared-FP8-cache lane.
pub(super) fn resolve_validated_prefill(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    resolved_prefill: Option<serve_phases::PrefillBudget>,
    ssm_prefill_chunk: usize,
) -> Result<serve_phases::PrefillBudget> {
    // Resolve once before weight loading, so the bounded shared-cache lane
    // rejects unsupported chunks/adapters before allocating model weights.
    let resolved_prefill = resolved_prefill
        .unwrap_or_else(|| serve_phases::resolve_prefill_budget(args, ssm_prefill_chunk));
    spark_model::layers::moe::validate_shared_fp8_cache_profile(
        config,
        resolved_prefill.prefill_budget,
        !args.lora_adapter.is_empty()
            || !args.lora_stageable.is_empty()
            || !args.lora_stageable_disk.is_empty(),
    )?;
    Ok(resolved_prefill)
}

/// A GLM checkpoint must resolve its native `<tool_call>` boundary token, and
/// it must be the tool-call start token.
pub(super) fn ensure_glm_tool_boundary(
    config: &ModelConfig,
    glm_tool_boundary: Option<u32>,
    tool_call_start_token: Option<u32>,
) -> Result<()> {
    anyhow::ensure!(
        config.model_type != "glm5_next"
            || (glm_tool_boundary.is_some() && glm_tool_boundary == tool_call_start_token),
        "GLM tokenizer/tool format is missing or mismatches its native <tool_call> token"
    );
    Ok(())
}
