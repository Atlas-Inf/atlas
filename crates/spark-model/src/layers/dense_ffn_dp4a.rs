// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::ops;

impl DenseFfnLayer {
    pub(super) fn forward_dp4a_single(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        if !ops::dp4a_enabled()
            || self.activation != FfnActivation::SiLU
            || self.dp4a_quant_k.0 == 0
            || self.dp4a_silu_quant_k.0 == 0
            || self.dp4a_gemv_k.0 == 0
        {
            return Ok(None);
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let quantized = ctx.buffers.ffn_act_a();
        let scales = ctx.buffers.ffn_act_scale();
        if quantized.is_null() || scales.is_null() {
            return Ok(None);
        }
        if ctx.stats.once("log:decode_ffn_dp4a_single") {
            tracing::info!("Dense FFN W4A8 DP4A single-row path engaged");
        }
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        ops::quantize_act_int8(
            ctx.gpu,
            self.dp4a_quant_k,
            input,
            quantized,
            scales,
            h,
            stream,
        )?;
        ops::w4a16_gemv_dp4a(
            ctx.gpu,
            self.dp4a_gemv_k,
            quantized,
            scales,
            &self.weights.gate_proj,
            gate_out,
            inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv_dp4a(
            ctx.gpu,
            self.dp4a_gemv_k,
            quantized,
            scales,
            &self.weights.up_proj,
            up_out,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul_quant_int8(
            ctx.gpu,
            self.dp4a_silu_quant_k,
            gate_out,
            up_out,
            quantized,
            scales,
            inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        ops::w4a16_gemv_dp4a(
            ctx.gpu,
            self.dp4a_gemv_k,
            quantized,
            scales,
            &self.weights.down_proj,
            output,
            h,
            inter,
            stream,
        )?;
        Ok(Some(output))
    }

    pub(super) fn forward_dp4a_batch(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !ops::dp4a_enabled()
            || self.activation != FfnActivation::SiLU
            || !(2..=8).contains(&m)
            || self.dp4a_quant_batch4_k.0 == 0
            || self.dp4a_gemv_batch4_k.0 == 0
            || self.dp4a_dual_batch4_k.0 == 0
            || self.dp4a_gemv_batch4_dyn_k.0 == 0
            || self.dp4a_dual_batch4_dyn_k.0 == 0
        {
            return Ok(false);
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let quantized = ctx.buffers.ffn_act_a();
        let scales = ctx.buffers.ffn_act_scale();
        if quantized.is_null() || scales.is_null() {
            return Ok(false);
        }
        if ctx.stats.once("log:decode_ffn_dp4a_batch") {
            tracing::info!(rows = m, "Dense FFN W4A8 DP4A batched path engaged");
        }
        // Guard-free specialization per exact tier; the _dyn twins keep the
        // runtime `row >= M` skip that makes the unused rows free. Every arm
        // is bit-identical per emitted row.
        // vl2 (ATLAS_GEMV_VL2, default on) applies to the gate+up DUAL only:
        // m 5..=8 → `dual_batch8_d4_dyn_vl2` when linked (the guarded twin
        // beat the fixed one at m==8: 880 vs 1066 us/call — the dyn's leaner
        // register profile wins). The single down-proj GEMV stays on the vl1
        // batch8 tiers: at K=17408 it is already near the streaming roofline
        // and vl2's halved grid cost more than the extra loads bought (+85%).
        // Bit-identical either way; a missing handle drops to the vl1 tier.
        let vl2 = ops::gemv_vl2_enabled();
        let (gemv_k, dual_k, dual_vl2) = if m >= 5 {
            let gk = if m == 8 {
                self.dp4a_gemv_batch8_k
            } else {
                self.dp4a_gemv_batch8_dyn_k
            };
            if gk.0 == 0 {
                return Ok(false);
            }
            let dvl2 = vl2 && self.dp4a_dual_batch8_dyn_vl2_k.0 != 0;
            let dk = if dvl2 {
                self.dp4a_dual_batch8_dyn_vl2_k
            } else if m == 8 {
                self.dp4a_dual_batch8_k
            } else {
                self.dp4a_dual_batch8_dyn_k
            };
            if dk.0 == 0 {
                return Ok(false);
            }
            (gk, dk, dvl2)
        } else if m == 4 {
            (self.dp4a_gemv_batch4_k, self.dp4a_dual_batch4_k, false)
        } else {
            (
                self.dp4a_gemv_batch4_dyn_k,
                self.dp4a_dual_batch4_dyn_k,
                false,
            )
        };
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        ops::quantize_act_int8_batch4(
            ctx.gpu,
            self.dp4a_quant_batch4_k,
            input,
            quantized,
            scales,
            m,
            h,
            stream,
        )?;
        let dual_launch = if dual_vl2 {
            ops::w4a16_gemv_dp4a_dual_batch8_vl2
        } else {
            ops::w4a16_gemv_dp4a_dual_batch4
        };
        dual_launch(
            ctx.gpu,
            dual_k,
            quantized,
            scales,
            &self.weights.gate_proj,
            gate_out,
            &self.weights.up_proj,
            up_out,
            m,
            inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        ops::quantize_act_int8_batch4(
            ctx.gpu,
            self.dp4a_quant_batch4_k,
            gate_out,
            quantized,
            scales,
            m,
            inter,
            stream,
        )?;
        ops::w4a16_gemv_dp4a_batch4(
            ctx.gpu,
            gemv_k,
            quantized,
            scales,
            &self.weights.down_proj,
            ctx.buffers.moe_output(),
            m,
            h,
            inter,
            stream,
        )?;
        Ok(true)
    }
}
