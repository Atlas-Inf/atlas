// SPDX-License-Identifier: AGPL-3.0-only
//! Typed substitutions within the existing transposed shared GEMM branches.
use super::m5_projections::{SharedResources, disjoint, shared_eligible, span};
use super::*;

#[derive(Clone, Copy)]
pub(super) enum SharedProjection {
    Gate,
    Up,
    Down,
}
impl SharedProjection {
    fn geometry(self) -> (u32, u32, u32) {
        match self {
            Self::Gate => (2048, 4096, 1),
            Self::Up => (2048, 4096, 2),
            Self::Down => (4096, 2048, 4),
        }
    }
    fn owner(self, ctx: &ForwardContext) -> (DevicePtr, usize) {
        match self {
            Self::Gate => (
                ctx.buffers.ssm_deinterleaved(),
                ctx.buffers.sizes().ssm_deinterleaved,
            ),
            Self::Up => (ctx.buffers.ssm_qkvz(), ctx.buffers.sizes().ssm_qkvz),
            Self::Down => (ctx.buffers.attn_output(), ctx.buffers.sizes().attn_output),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn checked_output(
    projection: SharedProjection,
    n: u32,
    k: u32,
    input: DevicePtr,
    weight: &QuantizedWeight,
    output: DevicePtr,
    owner: DevicePtr,
    capacity: usize,
) -> Result<usize> {
    let (pn, pk, _) = projection.geometry();
    anyhow::ensure!((n, k) == (pn, pk), "GLM shared M16 projection geometry");
    let bytes = 5 * n as usize * 2;
    anyhow::ensure!(
        output == owner && capacity >= bytes,
        "GLM shared M16 output owner/capacity"
    );
    anyhow::ensure!(
        weight.weight_scale_2.is_finite() && !weight.has_per_row_scale2(),
        "GLM shared M16 scale2 contract"
    );
    let a = span(input, 5 * k as usize * 2, 16)?;
    let b = span(weight.weight, n as usize * k as usize / 2, 16)?;
    let s = span(weight.weight_scale, n as usize * k as usize / 16, 16)?;
    let c = span(output, bytes, 2)?;
    disjoint(&a, &c)?;
    disjoint(&b, &c)?;
    disjoint(&s, &c)?;
    Ok(bytes)
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_m16(
        &self,
        projection: SharedProjection,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        rows: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
        overlap: bool,
    ) -> Result<()> {
        let feature = &self.m5_projections.shared;
        let r = SharedResources {
            comm: ctx.comm.is_some(),
            lora: self.lora.is_some(),
            nvfp4: self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4,
            transposed: self.shared_gate_t.is_some()
                && self.shared_up_t.is_some()
                && self.shared_down_t.is_some(),
            fp8_cache: self.shared_gate_fp8.is_some()
                || self.shared_up_fp8.is_some()
                || self.shared_down_fp8.is_some(),
            bf16_override: self.bf16_shared_expert.is_some(),
            // Hook is after the earlier exact-M5 GEMV arm returned.
            exact_gemv: false,
        };
        let launch = |kernel| {
            ops::w4a16_gemm_n128(ctx.gpu, kernel, input, weight, output, rows, n, k, stream)
        };
        if !feature.enabled() || !shared_eligible(ctx.config, rows, r) {
            return launch(self.w4a16_gemm_t);
        }
        let (owner, capacity) = projection.owner(ctx);
        let bytes = checked_output(projection, n, k, input, weight, output, owner, capacity)?;
        // These other scratch outputs remain live through the shared chain.
        let c = span(output, bytes, 2)?;
        for (other, len) in [
            (ctx.buffers.ssm_deinterleaved(), 20480),
            (ctx.buffers.ssm_qkvz(), 20480),
            (ctx.buffers.attn_output(), 40960),
        ] {
            if other != owner {
                disjoint(&c, &span(other, len, 2)?)?;
            }
        }
        feature.run(
            projection.geometry().2,
            output,
            bytes,
            self.w4a16_gemm_t,
            ctx.gpu,
            ctx.graph_capture,
            stream,
            overlap,
            launch,
        )
    }
}

#[cfg(test)]
#[path = "shared_m16_tests.rs"]
mod tests;
