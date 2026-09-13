// SPDX-License-Identifier: AGPL-3.0-only

//! W4A8 DP4A M=4 arm for the GDN mixer's NVFP4 QKVZ / out_proj batched
//! projections. Mirrors `model/trait_impl/lm_head_dp4a.rs`: the NVFP4 weight
//! layout is already what the DP4A kernel consumes, so the only addition is a
//! hoisted int8 activation quant per projection input.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::super::*;
use crate::weight_map::QuantizedWeight;

impl Qwen3SsmLayer {
    /// `true` when the guard-free M=4 DP4A arm applies: `ATLAS_W4A16_DP4A=1`,
    /// n == 4 exactly (the `_d4` kernel writes all four rows), both kernel
    /// handles resolved, and the shared int8 scratch non-null.
    pub(crate) fn ssm_dp4a_ready(&self, ctx: &ForwardContext<'_>, m: u32) -> bool {
        ops::dp4a_batch4_eligible(
            m,
            ops::dp4a_enabled(),
            self.dp4a_quant_batch4_k,
            self.dp4a_gemv_batch4_k,
            ctx.buffers.ffn_act_a(),
            ctx.buffers.ffn_act_scale(),
        )
    }

    /// Quantize `input` ([4, k] BF16) to int8 and run the M=4 DP4A GEMV into
    /// `output` ([4, n] BF16). Caller must have checked `ssm_dp4a_ready`.
    ///
    /// Reuses the dense-FFN int8 scratch: the SSM block and the FFN run
    /// sequentially on one stream and the FFN re-quantizes its own input, so
    /// the lifetimes do not overlap.
    pub(crate) fn ssm_dp4a_batch4_proj(
        &self,
        ctx: &ForwardContext<'_>,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let a_q = ctx.buffers.ffn_act_a();
        let a_scale = ctx.buffers.ffn_act_scale();
        ops::quantize_act_int8_batch4(
            ctx.gpu,
            self.dp4a_quant_batch4_k,
            input,
            a_q,
            a_scale,
            4,
            k,
            stream,
        )?;
        ops::w4a16_gemv_dp4a_batch4(
            ctx.gpu,
            self.dp4a_gemv_batch4_k,
            a_q,
            a_scale,
            weight,
            output,
            4,
            n,
            k,
            stream,
        )?;
        ops::dp4a_arm_active_once();
        Ok(())
    }

    /// FP4 batched mixer projection: the W4A8 DP4A M=4 arm when eligible,
    /// else the float `w4a16_gemv_batchm` GEMV — the previous `_ =>` arm.
    /// `label` names the projection in the M>16 fail-fast.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn ssm_fp4_proj(
        &self,
        ctx: &ForwardContext<'_>,
        label: &'static str,
        gemv_k: KernelHandle,
        input: DevicePtr,
        weight: &QuantizedWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.ssm_dp4a_ready(ctx, m) {
            return self.ssm_dp4a_batch4_proj(ctx, input, weight, output, n, k, stream);
        }
        // w4a16_gemv_batch16 is a MAX_M=16 template: at M>16 it silently
        // computes rows 0..15 and never writes rows 16.. — garbage, not a
        // crash. The eligibility gate makes this unreachable at n>16 today;
        // fail fast if that drifts.
        anyhow::ensure!(
            m <= 16,
            "SSM batchm {label} GEMV caps at M=16 (n={m}); tile-GEMM twins required"
        );
        ops::w4a16_gemv_batchm(ctx.gpu, gemv_k, input, weight, output, m, n, k, stream)
    }
}
