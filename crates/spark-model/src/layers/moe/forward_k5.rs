// SPDX-License-Identifier: AGPL-3.0-only

//! GLM K=5 verification built from the proven K2/K3 routed paths plus one
//! exact-M=5 shared-expert pass.

use super::*;

/// `ATLAS_GLM_K5_GROUPED_MOE=1`: the five verify rows take the grouped
/// prefill MoE.
pub(crate) fn k5_grouped_moe_requested() -> bool {
    std::env::var("ATLAS_GLM_K5_GROUPED_MOE").as_deref() == Ok("1")
}

/// `ATLAS_GLM_K5_FUSED_MOE_HC=1`: on that grouped pass the shared expert is
/// blended whole in the mHC post-step, after the EP reduce, instead of split
/// across the pair before it (`forward_pair_shared`). Both ranks must run the
/// same two values (`model::startup_parity`), or the reduce sums one rank's
/// half beside the other's whole.
pub(crate) fn k5_fused_moe_hc_requested() -> bool {
    std::env::var("ATLAS_GLM_K5_FUSED_MOE_HC").as_deref() == Ok("1")
}

impl MoeLayer {
    /// Materialize the established shared-expert blend after
    /// [`Self::forward_k5_for_hc`] deferred it. Used only by the one-shot exactness
    /// oracle; the optimized path consumes the same operands in mHC directly.
    pub fn finish_k5_deferred_shared_blend(
        &self,
        routed: DevicePtr,
        shared: DevicePtr,
        input: DevicePtr,
        gate_weight: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.btile_input_guard(input, 5, ctx, stream)?;
        anyhow::ensure!(
            gate_weight.0 == self.weights.shared_expert_gate.weight.0,
            "GLM K=5 deferred shared gate does not belong to this MoE layer"
        );
        ops::moe_batched_blend(
            ctx.gpu,
            self.moe_batched_blend,
            routed,
            shared,
            input,
            gate_weight,
            ctx.config.hidden_size as u32,
            5,
            stream,
        )
    }

    /// K=5 variant for a caller that can consume the shared-expert blend
    /// directly inside its hyperconnection post kernel. The returned optional
    /// pointer is the shared gate weight; `Some` means `moe_output` contains
    /// the globally reduced routed contribution while `attn_output` contains
    /// the unblended shared contribution.
    pub fn forward_k5_for_hc(
        &self,
        input: DevicePtr,
        allow_deferred_shared_hc: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        self.btile_input_guard(input, 5, ctx, stream)?;
        let defer = allow_deferred_shared_hc
            && k5_fused_moe_hc_requested()
            && self.use_btile_or_t_prefill()
            && k5_grouped_moe_requested()
            && ctx.config.model_type == "glm5_next"
            && ctx.config.ep_world_size == 2
            && ctx.comm.is_some()
            && ctx.config.shared_expert_intermediate_size > 0;
        if defer {
            self.forward_prefill_impl(input, 5, ctx, stream, true)?;
            return Ok((
                ctx.buffers.moe_output(),
                Some(self.weights.shared_expert_gate.weight),
            ));
        }
        Ok((self.forward_k5(input, ctx, stream)?, None))
    }

    /// Returns the buffer containing five output rows. The optimized arm is
    /// intentionally restricted to GLM's unified NVFP4, ungated-shared layout.
    pub fn forward_k5(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.btile_input_guard(input, 5, ctx, stream)?;
        // Experimental Marlin-shaped verifier path: keep the source activations
        // in BF16, then sort all five rows by expert and run Atlas's weight-only
        // W4A16 grouped GEMM. The GB10 kernel converts each activation/weight tile
        // to E4M3 on chip for FP8 MMA; unlike NVFP4 MMQ, it does not quantize and
        // stage the verifier activations in FP4. This preserves acceptance while
        // amortizing routed weights across the verifier batch.
        if self.use_btile_or_t_prefill() && k5_grouped_moe_requested() {
            self.forward_prefill(input, 5, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
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
            && self.tid2eid_dev.is_none()
            && ctx.config.scoring_func != "sqrtsoftplus"
            && self.w4a16_batchm.kernel(5).0 != 0;

        if !optimized {
            self.forward_batched(input, 5, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
        }

        let h = ctx.config.hidden_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let gate_out = ctx.buffers.logits();
        let up_out = ctx.buffers.ssm_qkvz();
        let shared_down = ctx.buffers.attn_output();
        let sh_gate = &self.weights.shared_expert.gate_proj;
        let sh_up = &self.weights.shared_expert.up_proj;
        let sh_down = &self.weights.shared_expert.down_proj;
        let batch5 = self.w4a16_batchm.kernel(5);

        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            input,
            sh_gate,
            gate_out,
            5,
            shared_inter,
            h,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            input,
            sh_up,
            up_out,
            5,
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
            5 * shared_inter,
            stream,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            gate_out,
            sh_down,
            shared_down,
            5,
            h,
            shared_inter,
            stream,
        )?;

        // Route all five rows once, then hand contiguous slices to the proven
        // K2/K3 expert waves below. This preserves their expert working sets
        // while avoiding a second read of the router matrix and top-k launch.
        let router_in = self.router_input(input, 5, h, ctx, stream)?;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                batch5,
                router_in,
                nvfp4,
                gate_logits,
                5,
                num_experts,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm,
                router_in,
                &self.weights.gate,
                gate_logits,
                5,
                num_experts,
                h,
                stream,
            )?;
        }
        let scratch = ctx.buffers.scratch();
        let routes = PrecomputedRoutes {
            indices: scratch,
            weights: scratch.offset(5 * top_k as usize * 4),
        };
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                routes.indices,
                routes.weights,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                5,
                stream,
            )?;
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                routes.indices,
                routes.weights,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                5,
                stream,
            )?;
        }
        super::union_stats::maybe_sample_expert_union(
            ctx,
            routes.indices,
            5,
            top_k as usize,
            stream,
        );

        // Run routed experts through their fused small-M kernels. Consume each
        // temporary moe_output before the next group overwrites it.
        self.forward_k2_routed_local(input, routes, ctx, stream)?;
        ctx.gpu
            .copy_d2d_async(ctx.buffers.moe_output(), input, 2 * h as usize * 2, stream)?;
        let row3 = input.offset(2 * h as usize * 2);
        let routes3 = PrecomputedRoutes {
            indices: routes.indices.offset(2 * top_k as usize * 4),
            weights: routes.weights.offset(2 * top_k as usize * 4),
        };
        self.forward_k3_routed_local(row3, routes3, ctx, stream)?;
        ctx.gpu
            .copy_d2d_async(ctx.buffers.moe_output(), row3, 3 * h as usize * 2, stream)?;

        // K2 and K3 produced rank-local routed contributions. Reducing their
        // contiguous five-row result once avoids a second EP collective per
        // MoE layer while retaining the faster small-M expert kernels.
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(input.0, 5 * h as usize * 2)?;
            } else {
                comm.all_reduce_async(input.0, 5 * h as usize * 2, stream)?;
            }
        }
        ops::residual_add(
            ctx.gpu,
            self.residual_add,
            input,
            shared_down,
            5 * h,
            stream,
        )?;
        Ok(input)
    }
}
