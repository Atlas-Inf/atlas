// SPDX-License-Identifier: AGPL-3.0-only
//! Opt-in BF16 GLM K3 vocabulary projection. Weights stay resident and unchanged;
//! the batch-GEMV reduction is bit-identical to three scalar GEMVs, but can differ
//! from the default GEMM in BF16 rounding. This is a separately qualified route.
use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    weights::{WeightDtype, WeightTensor},
};
use std::sync::OnceLock;

use crate::{layers::ops, weight_map::DenseWeight};

const FLAG: &str = "ATLAS_GLM_K3_HEAD_BATCHM";
fn parse_flag(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => bail!("{FLAG} must be 0 or 1"),
    }
}
fn enabled() -> Result<bool> {
    static VALUE: OnceLock<std::result::Result<bool, String>> = OnceLock::new();
    VALUE
        .get_or_init(|| {
            let value = std::env::var(FLAG);
            match value {
                Ok(value) => parse_flag(Some(&value)).map_err(|e| e.to_string()),
                Err(std::env::VarError::NotPresent) => Ok(false),
                Err(_) => Err(format!("{FLAG} must be 0 or 1")),
            }
        })
        .as_ref()
        .copied()
        .map_err(|e| anyhow::anyhow!(e.clone()))
}
fn validate_config(config: &ModelConfig) -> Result<()> {
    ensure!(
        config.model_type == "glm5_next"
            && config.hidden_size == 4096
            && config.vocab_size == 154856
            && config.skip_lm_head_quantization(),
        "{FLAG} requires GLM BF16 head with hidden=4096 and local vocab=154856"
    );
    Ok(())
}
/// Called by the GLM weight loader while actual tensor metadata is still owned.
/// A config-only check cannot establish the resident allocation's geometry.
/// Resolve the existing kernel before KV sizing, without launching a fake row:
/// the registry's cuModuleGetFunction path is also used by model construction.
pub(crate) fn initialize_from_weight(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    weight: &WeightTensor,
) -> Result<()> {
    initialize(gpu, config, weight, enabled()?)
}
fn initialize(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    weight: &WeightTensor,
    enabled: bool,
) -> Result<()> {
    if !enabled {
        return Ok(());
    }
    validate_config(config)?;
    ensure!(
        weight.dtype == WeightDtype::BF16
            && (weight.shape == [154856, 4096] || weight.shape == [154880, 4096])
            && weight.ptr.0 != 0
            && weight.ptr.0.is_multiple_of(16),
        "{FLAG} requires 16-byte-aligned BF16 lm_head.weight [154856,4096] or [154880,4096], got {:?} {:?}",
        weight.dtype,
        weight.shape
    );
    let kernel = gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?;
    ensure!(kernel.0 != 0, "{FLAG}: batchm kernel unavailable");
    // NVIDIA pads the resident vocabulary to 154880 rows. Only the active
    // 154856 rows are projected; padding changes neither K nor output stride.
    tracing::info!(
        resident_rows = weight.shape[0],
        active_rows = config.vocab_size,
        hidden = config.hidden_size,
        "GLM K3 BF16 head batchm initialized (ATLAS_GLM_K3_HEAD_BATCHM=1)"
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn project(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows: u32,
    batchm: KernelHandle,
    gemm: KernelHandle,
    stream: u64,
    k5_enabled: bool,
) -> Result<()> {
    let k3_enabled = config.model_type == "glm5_next" && rows == 3 && enabled()?;
    dispatch(
        gpu, config, input, weight, output, rows, batchm, gemm, stream, k3_enabled, k5_enabled,
    )
}

#[allow(clippy::too_many_arguments)]
fn dispatch(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    input: DevicePtr,
    weight: &DenseWeight,
    output: DevicePtr,
    rows: u32,
    batchm: KernelHandle,
    gemm: KernelHandle,
    stream: u64,
    k3_enabled: bool,
    k5_enabled: bool,
) -> Result<()> {
    let k3 = rows == 3 && k3_enabled;
    if k3 {
        validate_config(config)?;
        ensure!(batchm.0 != 0, "{FLAG}: batchm kernel unavailable");
        ensure!(
            input.0 != 0 && weight.weight.0 != 0 && output.0 != 0,
            "{FLAG}: null head operand"
        );
        // A/W are read as uint4; output stores are scalar BF16. K=4096
        // preserves 16-byte alignment across every input and weight row.
        ensure!(
            input.0.is_multiple_of(16)
                && weight.weight.0.is_multiple_of(16)
                && output.0.is_multiple_of(2),
            "{FLAG}: input/weight require 16-byte alignment and output 2-byte alignment"
        );
    }
    let k5 = rows == 5 && config.model_type == "glm5_next" && batchm.0 != 0 && k5_enabled;
    let n = config.vocab_size as u32;
    let k = config.hidden_size as u32;
    if k3 || k5 {
        ops::dense_gemv_batchm(gpu, batchm, input, weight, output, rows, n, k, n, stream)
    } else {
        ops::dense_gemm(gpu, gemm, input, weight, output, rows, n, k, stream)
    }
}

#[cfg(test)]
#[path = "glm_k3_head_tests.rs"]
mod tests;
