// SPDX-License-Identifier: AGPL-3.0-only
//! K3 shared-expert wrapper; routed implementation remains in the parent.
use super::*;

impl MoeLayer {
    /// Fused K=3 forward: process 3 tokens through MoE in 5 kernel launches.
    ///
    /// Gate GEMV batch3 → batched topK → fused expert gate+up → fused silu+down → fused wsum+blend.
    /// Expert buffers sized for 3*top_k slots. Output at moe_output() [3, H].
    pub fn forward_k3(
        &self,
        input: DevicePtr, // [3, H] BF16 — normed MoE input for 3 tokens
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.btile_input_guard(input, 3, ctx, stream)?;
        if self.glm_c3_grouped(ctx, 3) || ctx.config.expert_tp {
            return self.forward_prefill(input, 3, ctx, stream);
        }
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
            && self.w4a16_gemv_batch3.0 != 0;

        if !optimized {
            return self.forward_k3_impl(input, ctx, stream, true, true, None);
        }

        let h = ctx.config.hidden_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let gate_out = ctx.buffers.logits();
        let up_out = ctx.buffers.ssm_qkvz();
        let shared_down = ctx.buffers.attn_output();
        let sh_gate = &self.weights.shared_expert.gate_proj;
        let sh_up = &self.weights.shared_expert.up_proj;
        let sh_down = &self.weights.shared_expert.down_proj;

        // GLM's shared expert is identical for all verifier rows. Batch the
        // three rows so each NVFP4 projection is fetched once instead of once
        // per row in the fused routed-expert kernel.
        ops::w4a16_gemv_batch3(
            ctx.gpu,
            self.w4a16_gemv_batch3,
            input,
            sh_gate,
            gate_out,
            shared_inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv_batch3(
            ctx.gpu,
            self.w4a16_gemv_batch3,
            input,
            sh_up,
            up_out,
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
            3 * shared_inter,
            stream,
        )?;
        ops::w4a16_gemv_batch3(
            ctx.gpu,
            self.w4a16_gemv_batch3,
            gate_out,
            sh_down,
            shared_down,
            h,
            shared_inter,
            stream,
        )?;

        // Keep the routed K3 path and EP all-reduce unchanged, then add the
        // precomputed always-on shared expert once on each rank.
        self.forward_k3_impl(input, ctx, stream, false, true, None)?;
        ops::residual_add(
            ctx.gpu,
            self.residual_add,
            ctx.buffers.moe_output(),
            shared_down,
            3 * h,
            stream,
        )
    }
}
