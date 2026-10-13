// SPDX-License-Identifier: AGPL-3.0-only
//! Exact-weight batched O projection after 2..=MAX_ROWS causal, serial MLA
//! rows (the repaired K3 verify or a DFlash block); M equals the row count.
use crate::{layers::ops, weight_map::DenseWeight};
use anyhow::{Result, bail, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::{
    gpu::{DevicePtr, GpuBackend, KernelHandle},
    kv_cache::KvCacheDtype,
};
#[path = "glm_k3_mla_o_compare.rs"]
pub(super) mod compare;
const FLAG: &str = "ATLAS_GLM_K3_MLA_O_BATCHM";
const PREFIX: usize = 2 * 128 * 2; // index keys and gates, both BF16
const ROW: usize = 8192 * 2;
const OUTPUT_ROW: usize = 4096 * 2;
/// Widest verify block the staged MLA projections admit. One batched GEMV
/// covers every row, so the block must fit the kernel's row limit.
pub(in crate::layers::qwen3_attention) const MAX_ROWS: usize =
    crate::speculative::glm_repair_policy::MAX_DFLASH_VERIFY_ROWS;
const _: () = assert!(MAX_ROWS <= ops::DENSE_GEMV_BATCHM_MAX_M as usize);
pub(in crate::layers::qwen3_attention) fn rows_supported(rows: usize) -> bool {
    (2..=MAX_ROWS).contains(&rows)
}
/// Index key/gate prefix plus every retained V row.
pub(in crate::layers::qwen3_attention) const fn scratch_bytes(rows: usize) -> usize {
    PREFIX + rows * ROW
}
const fn output_bytes(rows: usize) -> usize {
    rows * OUTPUT_ROW
}
fn parse_flag(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => bail!("{FLAG} must be 0 or 1"),
    }
}
pub(crate) fn enabled(model: &str) -> Result<bool> {
    if model != "glm5_next" {
        return Ok(false);
    }
    match std::env::var(FLAG) {
        Ok(v) => parse_flag(Some(&v)),
        Err(std::env::VarError::NotPresent) => Ok(false),
        Err(_) => bail!("{FLAG} must be 0 or 1"),
    }
}
fn validate_weight(ptr: DevicePtr, n: usize, k: usize) -> Result<()> {
    ensure!(
        n == 4096 && k == 8192 && ptr.0 != 0 && ptr.0.is_multiple_of(16),
        "{FLAG} requires aligned resident BF16 O[4096,8192]"
    );
    Ok(())
}
/// Target-only loader hook, after BF16 conversion/sharding, before KV sizing.
/// The TP1 appended predictor is deliberately excluded by the caller.
pub(crate) fn initialize(
    gpu: &dyn GpuBackend,
    config: &ModelConfig,
    dtype: KvCacheDtype,
    ptr: DevicePtr,
    n: usize,
    k: usize,
) -> Result<()> {
    let compare = compare::enabled(&config.model_type)?;
    if !enabled(&config.model_type)? {
        return Ok(());
    }
    ensure!(
        config.hidden_size == 4096
            && config.num_attention_heads == 32
            && config.v_head_dim == 256
            && config.qk_rope_head_dim == 0
            && config.tp_world_size == 2
            && config.ep_world_size == 2
            && matches!(dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
            && super::glm_long_context::enabled(&config.model_type)
            && crate::speculative::glm_repair_policy::long_lane_enabled(),
        "{FLAG} requires repaired long-context GLM TP2/EP2 BF16 MLA"
    );
    validate_weight(ptr, n, k)?;
    if compare {
        ensure!(
            gpu.kernel("gemv", "dense_gemv_bf16")?.0 != 0,
            "ATLAS_GLM_K3_MLA_O_COMPARE: scalar kernel unavailable"
        );
        tracing::warn!(
            "GLM K3 MLA O comparison enabled: diagnostic synchronization; not for performance"
        );
    }
    ensure!(
        gpu.kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")?
            .0
            != 0,
        "{FLAG}: batchm kernel unavailable"
    );
    tracing::info!(
        "GLM K3 MLA O batchm initialized: local BF16 [4096,8192], causal serial attention"
    );
    Ok(())
}
fn span(ptr: DevicePtr, bytes: usize) -> Result<(u64, u64)> {
    ensure!(ptr.0 != 0, "{FLAG}: null staging/liveness pointer");
    Ok((
        ptr.0,
        ptr.0
            .checked_add(bytes as u64)
            .ok_or_else(|| anyhow::anyhow!("{FLAG}: address overflow"))?,
    ))
}
fn disjoint(a: (u64, u64), b: (u64, u64)) -> bool {
    a.1 <= b.0 || b.1 <= a.0
}
#[derive(Clone, Copy)]
pub(in crate::layers::qwen3_attention) struct StagePlan {
    rows: usize,
    input: DevicePtr,
    output: DevicePtr,
    comparison: Option<DevicePtr>,
}
impl StagePlan {
    pub(in crate::layers::qwen3_attention) fn new(
        rows: usize,
        scratch: DevicePtr,
        capacity: usize,
        output: DevicePtr,
        output_capacity: usize,
        live: &[(DevicePtr, usize)],
    ) -> Result<Self> {
        ensure!(
            rows_supported(rows),
            "{FLAG}: {rows} staged rows outside 2..={MAX_ROWS}"
        );
        let (scratch_bytes, output_bytes) = (scratch_bytes(rows), output_bytes(rows));
        ensure!(
            capacity >= scratch_bytes && output_capacity >= output_bytes,
            "{FLAG}: {rows} rows need {scratch_bytes} scratch bytes and {output_bytes} output bytes"
        );
        ensure!(
            scratch.0.is_multiple_of(16) && output.0.is_multiple_of(2),
            "{FLAG}: invalid operand alignment"
        );
        let whole = span(scratch, scratch_bytes)?;
        let retained = (whole.0 + PREFIX as u64, whole.1);
        let out = span(output, output_bytes)?;
        ensure!(disjoint(whole, out), "{FLAG}: staging/output alias");
        for &(ptr, bytes) in live {
            let other = span(ptr, bytes)?;
            ensure!(
                disjoint(retained, other),
                "{FLAG}: retained MLA rows alias live scratch/input"
            );
        }
        Ok(Self {
            rows,
            input: DevicePtr(retained.0),
            output,
            comparison: None,
        })
    }
    pub(in crate::layers::qwen3_attention) fn row(self, i: usize) -> Result<DevicePtr> {
        ensure!(
            i < self.rows,
            "{FLAG}: staging row {i} outside {}",
            self.rows
        );
        Ok(self.input.offset(i * ROW))
    }
    pub(in crate::layers::qwen3_attention) fn project(
        self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        weight: &DenseWeight,
        stream: u64,
    ) -> Result<()> {
        validate_weight(weight.weight, 4096, 8192)?;
        ensure!(kernel.0 != 0, "{FLAG}: batchm kernel unavailable");
        ops::dense_gemv_batchm(
            gpu,
            kernel,
            self.input,
            weight,
            self.comparison.unwrap_or(self.output),
            self.rows as u32,
            4096,
            8192,
            4096,
            stream,
        )
    }
}
#[cfg(test)]
#[path = "glm_k3_mla_o_tests.rs"]
mod tests;
