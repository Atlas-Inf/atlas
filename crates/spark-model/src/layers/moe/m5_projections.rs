// SPDX-License-Identifier: AGPL-3.0-only

//! Independent, bounded GLM M5 projection policies.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use std::sync::atomic::{AtomicU32, Ordering};

#[derive(Clone, Copy)]
struct Toggle {
    enabled: bool,
    verify: bool,
}
impl Toggle {
    fn parse(main: Option<&str>, verify: Option<&str>) -> Result<Self> {
        let parse = |value| match value {
            None | Some("0") => Ok(false),
            Some("1") => Ok(true),
            _ => anyhow::bail!("GLM M5 projection toggles require 0 or 1"),
        };
        let (enabled, verify) = (parse(main)?, parse(verify)?);
        anyhow::ensure!(
            !verify || enabled,
            "GLM M5 projection VERIFY requires its main feature"
        );
        Ok(Self { enabled, verify })
    }
    fn env(name: &str) -> Result<Self> {
        Self::parse(
            std::env::var(name).ok().as_deref(),
            std::env::var(format!("{name}_VERIFY")).ok().as_deref(),
        )
    }
}

fn reject_graphs(model: &str, rows: usize, graphs: bool, router: bool, shared: bool) -> Result<()> {
    anyhow::ensure!(
        model != "glm5_next" || rows != 5 || !graphs || !(router || shared),
        "GLM M5 projection VERIFY requires actual M5 graphs disabled before capture"
    );
    Ok(())
}

/// Checked at each actual M5 graph decision, before lookup, warmup or capture.
pub(crate) fn validate_m5_projection_graphs(
    model_type: &str,
    rows: usize,
    use_graphs: bool,
) -> Result<()> {
    reject_graphs(
        model_type,
        rows,
        use_graphs,
        Toggle::env("ATLAS_GLM_M5_ROUTER_BN4")?.verify,
        Toggle::env("ATLAS_GLM_M5_SHARED_M16")?.verify,
    )
}

pub(super) fn span(ptr: DevicePtr, bytes: usize, alignment: u64) -> Result<std::ops::Range<u64>> {
    anyhow::ensure!(
        bytes > 0
            && alignment.is_power_of_two()
            && !ptr.is_null()
            && ptr.0.is_multiple_of(alignment),
        "GLM M5 projection null, empty or misaligned span"
    );
    Ok(ptr.0
        ..ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| anyhow::anyhow!("GLM M5 projection address overflow"))?)
}
pub(super) fn disjoint(a: &std::ops::Range<u64>, b: &std::ops::Range<u64>) -> Result<()> {
    anyhow::ensure!(
        a.end <= b.start || b.end <= a.start,
        "GLM M5 projection live span alias"
    );
    Ok(())
}

pub(super) struct ProjectionFeature {
    pub kernel: KernelHandle,
    verify: bool,
    checked: AtomicU32,
    selected: AtomicU32,
    label: &'static str,
}
impl ProjectionFeature {
    fn new(
        gpu: &dyn GpuBackend,
        config: &ModelConfig,
        flag: &'static str,
        module: &str,
        symbol: &str,
    ) -> Result<Self> {
        let toggle = Toggle::env(flag)?;
        let enabled = toggle.enabled && super::prequant_fp4::glm_grouped_shape(config);
        Ok(Self {
            kernel: if enabled {
                gpu.kernel(module, symbol)?
            } else {
                KernelHandle(0)
            },
            verify: enabled && toggle.verify,
            checked: AtomicU32::new(0),
            selected: AtomicU32::new(0),
            label: flag,
        })
    }
    pub(super) fn enabled(&self) -> bool {
        self.kernel.0 != 0
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run(
        &self,
        bit: u32,
        output: DevicePtr,
        bytes: usize,
        old: KernelHandle,
        gpu: &dyn GpuBackend,
        capturing: bool,
        stream: u64,
        overlap: bool,
        launch: impl Fn(KernelHandle) -> Result<()>,
    ) -> Result<()> {
        // Refuse diagnostics even after this projection was checked: a later
        // graph capture must never depend on a warmup's host bookkeeping.
        if self.verify {
            anyhow::ensure!(
                !capturing && !overlap && !gpu.stream_is_capturing(stream),
                "GLM M5 projection VERIFY requires eager, non-overlapped execution"
            );
        }
        if self.verify && self.checked.load(Ordering::Relaxed) & bit == 0 {
            super::m5_projection_oracle::verify_output(
                gpu,
                stream,
                output,
                bytes,
                capturing,
                overlap,
                old,
                self.kernel,
                launch,
            )?;
            self.checked.fetch_or(bit, Ordering::Relaxed);
            tracing::info!(
                feature = self.label,
                projection = bit,
                output = output.0,
                "GLM M5 projection full-output oracle PASS"
            );
        } else {
            launch(self.kernel)?;
        }
        if self.selected.fetch_or(bit, Ordering::Relaxed) & bit == 0 {
            tracing::info!(
                feature = self.label,
                projection = bit,
                "GLM M5 projection selected"
            );
        }
        Ok(())
    }
}

pub(super) struct M5Projections {
    pub router: ProjectionFeature,
    pub shared: ProjectionFeature,
}
impl M5Projections {
    pub(super) fn new(gpu: &dyn GpuBackend, config: &ModelConfig) -> Result<Self> {
        Ok(Self {
            router: ProjectionFeature::new(
                gpu,
                config,
                "ATLAS_GLM_M5_ROUTER_BN4",
                "glm_router_bn4",
                "glm_router_m5_bn4",
            )?,
            shared: ProjectionFeature::new(
                gpu,
                config,
                "ATLAS_GLM_M5_SHARED_M16",
                "w4a16",
                "glm_shared_w4a16_m16",
            )?,
        })
    }
}

#[derive(Clone, Copy)]
pub(super) struct RouterResources {
    pub comm: bool,
    pub lora: bool,
    pub bf16: bool,
    pub pre_norm: bool,
    pub hash: bool,
    pub bias: bool,
    pub old_m5: bool,
}

#[derive(Clone, Copy)]
pub(super) struct SharedResources {
    pub comm: bool,
    pub lora: bool,
    pub nvfp4: bool,
    pub transposed: bool,
    pub fp8_cache: bool,
    pub bf16_override: bool,
    pub exact_gemv: bool,
}

pub(super) fn router_eligible(config: &ModelConfig, rows: u32, r: RouterResources) -> bool {
    rows == 5
        && super::prequant_fp4::glm_grouped_shape(config)
        && r.comm
        && !r.lora
        && r.bf16
        && !r.pre_norm
        && !r.hash
        && r.bias
        && r.old_m5
}

pub(super) fn shared_eligible(config: &ModelConfig, rows: u32, r: SharedResources) -> bool {
    rows == 5
        && super::prequant_fp4::glm_grouped_shape(config)
        && r.comm
        && !r.lora
        && r.nvfp4
        && r.transposed
        && !r.fp8_cache
        && !r.bf16_override
        && !r.exact_gemv
}

#[cfg(test)]
#[path = "m5_projections_tests.rs"]
mod tests;
