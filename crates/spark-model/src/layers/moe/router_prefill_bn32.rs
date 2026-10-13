// SPDX-License-Identifier: AGPL-3.0-only
//! GLM prefill-only order-preserving router with 32-column output tiles.
use super::*;
use anyhow::ensure;
use atlas_core::config::ModelConfig;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};
fn parse(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("ATLAS_GLM_ROUTER_PREFILL_BN32 must be 0 or 1"),
    }
}
pub(super) fn resolve(config: &ModelConfig, gpu: &dyn GpuBackend) -> Result<KernelHandle> {
    if config.model_type != "glm5_next" {
        return Ok(KernelHandle(0));
    }
    let on = match std::env::var("ATLAS_GLM_ROUTER_PREFILL_BN32") {
        Ok(value) => parse(Some(&value))?,
        Err(std::env::VarError::NotPresent) => false,
        Err(error) => return Err(error.into()),
    };
    if !on {
        return Ok(KernelHandle(0));
    }
    ensure!(
        config.hidden_size == 4096 && config.num_experts == 288,
        "GLM BN32 router requires N288/K4096"
    );
    let kernel = gpu.kernel("gemm", "dense_gemm_bf16_router_bn32")?;
    ensure!(kernel.0 != 0, "GLM BN32 router kernel missing");
    Ok(kernel)
}
fn span(ptr: DevicePtr, bytes: usize) -> Result<std::ops::Range<u64>> {
    ensure!(
        !ptr.is_null() && ptr.0.is_multiple_of(16),
        "GLM BN32 router pointer alignment"
    );
    Ok(ptr.0
        ..ptr
            .0
            .checked_add(u64::try_from(bytes)?)
            .ok_or_else(|| anyhow::anyhow!("GLM BN32 router address overflow"))?)
}
impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_router_prefill_bn32(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if self.router_prefill_bn32.0 == 0 || m <= 8 || ctx.config.model_type != "glm5_next" {
            return Ok(false);
        }
        ensure!(
            n == 288
                && k == 4096
                && ctx.config.num_experts == 288
                && ctx.config.hidden_size == 4096,
            "GLM BN32 router geometry mismatch"
        );
        let output_bytes = (m as usize)
            .checked_mul(288 * 2)
            .ok_or_else(|| anyhow::anyhow!("GLM BN32 router output overflow"))?;
        ensure!(
            output == ctx.buffers.gate_logits() && output_bytes <= ctx.buffers.sizes().gate_logits,
            "GLM BN32 router output owner/capacity"
        );
        let input_bytes = (m as usize)
            .checked_mul(4096 * 2)
            .ok_or_else(|| anyhow::anyhow!("GLM BN32 router input overflow"))?;
        let a = span(input, input_bytes)?;
        let b = span(self.weights.gate.weight, 288 * 4096 * 2)?;
        let c = span(output, output_bytes)?;
        ensure!(
            (a.end <= c.start || c.end <= a.start) && (b.end <= c.start || c.end <= b.start),
            "GLM BN32 router output aliases operands"
        );
        KernelLaunch::new(ctx.gpu, self.router_prefill_bn32)
            .grid([9, div_ceil(m, 16), 1])
            .block([8, 16, 1])
            .arg_ptr(input)
            .arg_ptr(self.weights.gate.weight)
            .arg_ptr(output)
            .arg_u32(m)
            .arg_u32(n)
            .arg_u32(k)
            .launch(stream)?;
        Ok(true)
    }
}
#[cfg(test)]
#[path = "router_prefill_bn32_tests.rs"]
mod tests;
