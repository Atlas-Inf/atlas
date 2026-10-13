// SPDX-License-Identifier: AGPL-3.0-only

//! GLM K=4 verification built from the proven parallel K=2 routed path plus
//! one four-row shared-expert GEMM. This preserves routed-expert occupancy and
//! removes the four redundant reads of GLM's always-on shared expert.

use super::*;

impl MoeLayer {
    /// Returns the buffer containing four output rows. The optimized arm is
    /// intentionally restricted to GLM's unified NVFP4, ungated-shared layout;
    /// every other model retains the established two-K2 composition.
    pub fn forward_k4(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.btile_input_guard(input, 4, ctx, stream)?;
        let optimized = self.lora.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.has_mixed_bf16_shared_expert()
            && matches!(
                self.experts_scale_kind,
                crate::weight_map::WeightQuantFormat::Nvfp4
            )
            && self.use_btile_or_t_decode()
            && self.weights.shared_expert_gate.weight.is_null()
            && !self.weights.shared_expert.gate_proj.is_null()
            && !self.weights.shared_expert.up_proj.is_null()
            && !self.weights.shared_expert.down_proj.is_null()
            && self.w4a16_batchm.kernel(4).0 != 0;

        if !optimized {
            return self.forward_k4_as_k2_pairs(input, ctx, stream);
        }

        let h = ctx.config.hidden_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let gate_out = ctx.buffers.logits();
        let up_out = ctx.buffers.ssm_qkvz();
        let shared_down = ctx.buffers.attn_output();
        let sh_gate = &self.weights.shared_expert.gate_proj;
        let sh_up = &self.weights.shared_expert.up_proj;
        let sh_down = &self.weights.shared_expert.down_proj;

        // Exact-M=4 GEMV retains the decode-native layout and reads each
        // projection weight once without padding the four rows to a GEMM tile.
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            self.w4a16_batchm.kernel(4),
            input,
            sh_gate,
            gate_out,
            4,
            shared_inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            self.w4a16_batchm.kernel(4),
            input,
            sh_up,
            up_out,
            4,
            shared_inter,
            h,
            stream,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            gate_out,
            up_out,
            gate_out,
            4 * shared_inter,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            self.w4a16_batchm.kernel(4),
            gate_out,
            sh_down,
            shared_down,
            4,
            h,
            shared_inter,
            stream,
        )?;

        // Preserve K2's parallel expert CTAs and EP reduction. Each pair skips
        // its shared branch; pair outputs are staged over input rows only after
        // the shared expert has consumed all four original rows.
        for pair in 0..2 {
            let row_offset = pair * 2 * h as usize * 2;
            self.forward_k2_routed_only(input.offset(row_offset), ctx, stream)?;
            ctx.gpu.copy_d2d_async(
                ctx.buffers.moe_output(),
                input.offset(row_offset),
                2 * h as usize * 2,
                stream,
            )?;
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add,
            input,
            shared_down,
            4 * h,
            stream,
        )?;
        Ok(input)
    }

    fn forward_k4_as_k2_pairs(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size;
        for pair in 0..2 {
            let row_offset = pair * 2 * h * 2;
            self.forward_k2(input.offset(row_offset), ctx, stream)?;
            ctx.gpu.copy_d2d_async(
                ctx.buffers.moe_output(),
                input.offset(row_offset),
                2 * h * 2,
                stream,
            )?;
        }
        Ok(input)
    }
}
