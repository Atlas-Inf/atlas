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
/// A staged V row of a TP2 rank (32 heads x 256), the widest; a TP4 rank's
/// 16 heads stage half as wide.
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
/// Index key/gate prefix plus every retained V row, at the widest rows.
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
/// The O projection's local input width: 16 or 32 heads x 256.
fn width_supported(k: usize) -> bool {
    matches!(k, 4096 | 8192)
}
fn validate_weight(ptr: DevicePtr, n: usize, k: usize) -> Result<()> {
    ensure!(
        n == 4096 && width_supported(k) && ptr.0 != 0 && ptr.0.is_multiple_of(16),
        "{FLAG} requires aligned resident BF16 O[4096,4096|8192]"
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
            && ops::glm_sparse_tc_rank(config)
            && config.v_head_dim == 256
            && config.qk_rope_head_dim == 0
            && matches!(dtype, KvCacheDtype::Bf16 | KvCacheDtype::Fp8G128)
            && super::glm_long_context::enabled(&config.model_type)
            && crate::speculative::glm_repair_policy::long_lane_enabled(),
        "{FLAG} requires repaired long-context GLM TP2/EP2 or TP4/EP4 BF16 MLA"
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
        "GLM K3 MLA O batchm initialized: local BF16 [4096,{k}], causal serial attention"
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
    /// Elements of a staged row (the O projection's K).
    width: usize,
    input: DevicePtr,
    output: DevicePtr,
    comparison: Option<DevicePtr>,
}
impl StagePlan {
    /// `rows` staged rows of `width` elements (16 or 32 heads x 256).
    pub(in crate::layers::qwen3_attention) fn new(
        rows: usize,
        width: usize,
        scratch: DevicePtr,
        capacity: usize,
        output: DevicePtr,
        output_capacity: usize,
        live: &[(DevicePtr, usize)],
    ) -> Result<Self> {
        ensure!(
            rows_supported(rows) && width_supported(width),
            "{FLAG}: {rows} staged rows outside 2..={MAX_ROWS}, or width {width}"
        );
        let scratch_bytes = PREFIX + rows * width * 2;
        let output_bytes = output_bytes(rows);
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
            width,
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
        Ok(self.input.offset(i * self.width * 2))
    }
    /// Elements of a staged row (the O projection's K).
    pub(in crate::layers::qwen3_attention) fn width(&self) -> usize {
        self.width
    }
    /// Whether the scalar comparison is on (it reads the BF16 weight).
    pub(in crate::layers::qwen3_attention) fn comparing(&self) -> bool {
        self.comparison.is_some()
    }

    /// The batched O projection of the staged rows: `native(input, output,
    /// rows)` first (the layer's FP8 block twin, `false` without one), else
    /// the BF16 batch-M GEMV.
    pub(in crate::layers::qwen3_attention) fn project(
        self,
        gpu: &dyn GpuBackend,
        kernel: KernelHandle,
        weight: &DenseWeight,
        native: impl FnOnce(DevicePtr, DevicePtr, u32) -> Result<bool>,
        stream: u64,
    ) -> Result<()> {
        if native(
            self.input,
            self.comparison.unwrap_or(self.output),
            self.rows as u32,
        )? {
            return Ok(());
        }
        validate_weight(weight.weight, 4096, self.width)?;
        ensure!(kernel.0 != 0, "{FLAG}: batchm kernel unavailable");
        ops::dense_gemv_batchm(
            gpu,
            kernel,
            self.input,
            weight,
            self.comparison.unwrap_or(self.output),
            self.rows as u32,
            4096,
            self.width as u32,
            4096,
            stream,
        )
    }
}
#[cfg(test)]
#[path = "glm_k3_mla_o_tests.rs"]
mod tests;
