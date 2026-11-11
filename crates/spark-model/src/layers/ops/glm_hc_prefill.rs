// SPDX-License-Identifier: AGPL-3.0-only
//! Exact HC4 finalizer selection at the two GLM raw-mix prefill call sites.
use anyhow::{Result, bail, ensure};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

fn parse(model: &str, value: Option<&str>) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        Some(other) => bail!("ATLAS_GLM_HC_PREFILL_VEC must be 0 or 1, got {other:?}"),
    }
}

/// Keep the established raw-mix ABI and its Sinkhorn parameters. This selector
/// is called only by GLM prefill; small-row verification retains its kernel.
pub fn glm_hc_prefill_finalize_kernel(
    gpu: &dyn GpuBackend,
    model: &str,
    fallback: KernelHandle,
    tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
) -> Result<KernelHandle> {
    let value = std::env::var("ATLAS_GLM_HC_PREFILL_VEC");
    let enabled = match value {
        Ok(value) => parse(model, Some(&value))?,
        Err(std::env::VarError::NotPresent) => parse(model, None)?,
        Err(error) => return Err(error.into()),
    };
    select(gpu, model, fallback, tokens, hidden_size, hc_mult, enabled)
}

fn select(
    gpu: &dyn GpuBackend,
    model: &str,
    fallback: KernelHandle,
    tokens: u32,
    hidden_size: u32,
    hc_mult: u32,
    enabled: bool,
) -> Result<KernelHandle> {
    if !enabled || model != "glm5_next" || tokens <= 8 {
        return Ok(fallback);
    }
    ensure!(
        hidden_size == 4096 && hc_mult == 4,
        "GLM vector HC prefill requires hidden4096 and HC4"
    );
    let kernel = gpu.kernel(
        "glm_hc_prefill_vec",
        &super::hc_kernel_name(model, "glm_hc_pre_from_raw_mix_vec"),
    )?;
    ensure!(kernel.0 != 0, "GLM vector HC prefill kernel is unavailable");
    Ok(kernel)
}

#[cfg(test)]
#[path = "glm_hc_prefill_tests.rs"]
mod tests;
