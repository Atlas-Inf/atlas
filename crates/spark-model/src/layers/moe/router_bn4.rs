// SPDX-License-Identifier: AGPL-3.0-only
//! Called only from the existing exact-M5 BF16 router arm.
use super::m5_projections::{RouterResources, disjoint, router_eligible, span};
use super::*;

fn checked_output(
    input: DevicePtr,
    weight: DevicePtr,
    output: DevicePtr,
    owner: DevicePtr,
    capacity: usize,
) -> Result<usize> {
    let bytes = 5 * 288 * 2;
    anyhow::ensure!(
        output == owner && capacity >= bytes,
        "GLM BN4 router output owner/capacity"
    );
    let a = span(input, 5 * 4096 * 2, 8)?;
    let b = span(weight, 288 * 4096 * 2, 8)?;
    let c = span(output, bytes, 2)?;
    disjoint(&a, &c)?;
    disjoint(&b, &c)?;
    Ok(bytes)
}

impl MoeLayer {
    pub(super) fn run_router_bn4(
        &self,
        input: DevicePtr,
        output: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let feature = &self.m5_projections.router;
        let resources = RouterResources {
            comm: ctx.comm.is_some(),
            lora: self.lora.is_some(),
            bf16: self.gate_fp8.is_none() && self.gate_nvfp4.is_none(),
            pre_norm: self.weights.router_pre_norm.is_some() || self.pre_expert_norm.is_some(),
            hash: self.tid2eid_dev.is_some(),
            bias: self.correction_bias_dev.is_some_and(|p| !p.is_null()),
            // The parent arm has already resolved the old M5 flag and handle.
            old_m5: self.dense_gemm_router_m5.0 != 0,
        };
        let old = |kernel| {
            ops::dense_gemm_router_m5(
                ctx.gpu,
                kernel,
                input,
                &self.weights.gate,
                output,
                5,
                288,
                4096,
                stream,
            )
        };
        if !feature.enabled() || !router_eligible(ctx.config, 5, resources) {
            return old(self.dense_gemm_router_m5);
        }
        let bytes = checked_output(
            input,
            self.weights.gate.weight,
            output,
            ctx.buffers.gate_logits(),
            ctx.buffers.sizes().gate_logits,
        )?;
        feature.run(
            1,
            output,
            bytes,
            self.dense_gemm_router_m5,
            ctx.gpu,
            ctx.graph_capture,
            stream,
            false,
            |kernel| {
                if kernel.0 == self.dense_gemm_router_m5.0 {
                    old(kernel)
                } else {
                    ops::glm_router_bn4(
                        ctx.gpu,
                        kernel,
                        input,
                        self.weights.gate.weight,
                        output,
                        stream,
                    )
                }
            },
        )
    }
}

#[cfg(test)]
#[path = "router_bn4_tests.rs"]
mod tests;
