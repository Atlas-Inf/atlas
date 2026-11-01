// SPDX-License-Identifier: AGPL-3.0-only
//! Router gate GEMV and batched top-k of the K=3 verify.
use super::*;

impl MoeLayer {
    /// Route all three rows: gate GEMV batch3 then batched top-k. Returns the
    /// `(indices, weights)` device buffers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_k3_route(
        &self,
        input: DevicePtr,
        h: u32,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, 3, h, ctx, stream)?;
        // 1. Gate GEMV batch3: reads gate weight once for 3 tokens
        let gate_logits = ctx.buffers.gate_logits();
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv_batch3(
                ctx.gpu,
                self.w4a16_gemv_batch3,
                router_in,
                nvfp4,
                gate_logits,
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
                3,
                num_experts,
                h,
                stream,
            )?;
        }

        // 2. Batched topK for 3 tokens. Sigmoid+bias for MiniMax/DeepSeek-V3,
        //    softmax otherwise.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(3 * top_k as usize * 4);
        if let Some(bias) = self.correction_bias_dev {
            ops::moe_topk_sigmoid_batched(
                ctx.gpu,
                self.moe_topk_sigmoid_batched_k,
                gate_logits,
                bias,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                3,
                stream,
            )?;
        } else {
            ops::moe_topk_softmax_batched(
                ctx.gpu,
                self.moe_topk_batched,
                gate_logits,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                3,
                stream,
            )?;
        }

        super::union_stats::maybe_sample_expert_union(ctx, indices_dev, 3, top_k as usize, stream);
        Ok((indices_dev, weights_dev))
    }
}
