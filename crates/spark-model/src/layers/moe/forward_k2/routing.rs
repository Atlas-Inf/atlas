// SPDX-License-Identifier: AGPL-3.0-only

//! Router gate GEMV and batched top-k of the K=2 verify. Split from
//! `forward_k2.rs` (500-LoC cap) as a child module so field access is unchanged.

use super::super::*;

impl MoeLayer {
    /// Route both rows: gate GEMV batch2 then batched top-k. Returns the
    /// `(indices, weights)` device buffers.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_k2_route(
        &self,
        input: DevicePtr,
        h: u32,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, DevicePtr)> {
        // Gemma-4 router pre-norm (no-op for other models).
        let router_in = self.router_input(input, 2, h, ctx, stream)?;
        // 1. Gate GEMV batch2: reads gate weight once for 2 tokens
        let gate_logits = ctx.buffers.gate_logits(); // [2, 512] BF16
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2,
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
                2,
                num_experts,
                h,
                stream,
            )?;
        }

        // 2. Batched topK for 2 tokens: [2, 512] → [2*top_k] indices + [2*top_k] weights.
        //    Sigmoid+bias for MiniMax/DeepSeek-V3, softmax otherwise.
        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch; // [2*top_k] u32
        let weights_dev = scratch.offset(2 * top_k as usize * 4); // [2*top_k] f32
        if let Some(bias) = self.correction_bias_dev {
            // DeepSeek-V4 scores experts with sqrt(softplus(.)); sigmoid otherwise
            // (MiniMax/DeepSeek-V3). Must match the prefill/single-token paths or
            // decode routing diverges from prefill.
            if ctx.config.scoring_func == "sqrtsoftplus" {
                // Use the PROVEN non-batched sqrtsoftplus kernel per token (the
                // _batched variant is unexercised — the K2 verify is the only
                // user and it never ran for V4 before). gate_logits is BF16
                // [2, num_experts] (2-byte stride); indices/weights are
                // [2, top_k] (u32 / f32, 4-byte stride).
                for t in 0..2usize {
                    ops::moe_topk_sqrtsoftplus(
                        ctx.gpu,
                        self.moe_topk_sqrtsoftplus_k,
                        gate_logits.offset(t * num_experts as usize * 2),
                        bias,
                        indices_dev.offset(t * top_k as usize * 4),
                        weights_dev.offset(t * top_k as usize * 4),
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        ctx.config.routed_scaling_factor as f32,
                        stream,
                    )?;
                }
            } else {
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
                    2,
                    stream,
                )?;
            }
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
                2,
                stream,
            )?;
        }
        super::union_stats::maybe_sample_expert_union(ctx, indices_dev, 2, top_k as usize, stream);
        Ok((indices_dev, weights_dev))
    }
}
