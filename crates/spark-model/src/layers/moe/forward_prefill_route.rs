// SPDX-License-Identifier: AGPL-3.0-only

//! Router gate GEMM and top-k dispatch (steps 1-2) of `MoeLayer::forward_prefill`.

use super::*;

impl MoeLayer {
    /// Step 1: router gate GEMM `[N, H] x [H, num_experts]` into `gate_logits`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_gate_gemm(
        &self,
        router_in: DevicePtr,
        gate_logits: DevicePtr,
        n: u32,
        num_experts: u32,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(fp8) = self.gate_fp8 {
            ops::fp8_gemm_n128(
                ctx.gpu,
                self.fp8_gemm_k,
                router_in,
                fp8,
                gate_logits,
                n,
                // = num_experts everywhere except LongCat (zero-expert logits).
                self.router_logits_n,
                h,
                stream,
            )?;
        } else if let Some(ref nvfp4) = self.gate_nvfp4 {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm,
                router_in,
                nvfp4,
                gate_logits,
                n,
                self.router_logits_n,
                h,
                stream,
            )?;
        } else if self.independent_grouped(ctx, n) {
            self.independent_router_logits(router_in, gate_logits, n as usize, ctx, stream)?;
        } else if self.glm_c3_grouped(ctx, n) {
            // Preserve forward_k3's router logits and expert weights exactly;
            // the experiment changes routed activation precision, not routing.
            self.c3_router_logits(router_in, gate_logits, n, num_experts, h, ctx, stream)?;
        } else if self.glm_c2_grouped(ctx, n) {
            self.independent_router_logits(router_in, gate_logits, 2, ctx, stream)?;
        } else if self.glm_c4_grouped(ctx, n) {
            self.c4_router_logits(router_in, gate_logits, ctx, stream)?;
        } else {
            // Selection numerics — see router_gate_gemm_dense for why this
            // must stay on the scalar kernel and why ATLAS_CUBLAS_GEMM must
            // not reroute it either (2026-08-12 BFCL regression: a rerouted
            // router GEMM flips top-k on borderline tokens deterministically).
            self.router_gate_gemm_dense(
                router_in,
                gate_logits,
                n,
                self.router_logits_n,
                h,
                ctx,
                stream,
            )?;
        }
        Ok(())
    }

    /// Step 2: batched top-k over `gate_logits` into `indices_dev` / `weights_dev`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_topk(
        &self,
        gate_logits: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        n: u32,
        num_experts: u32,
        top_k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(tid2eid) = self.tid2eid_dev {
            // DeepSeek-V4 hash routing (hash_moe layer): static
            // `tid2eid[token_id]` selection, sqrtsoftplus-weighted.
            let token_ids = ctx.token_ids.ok_or_else(|| {
                anyhow::anyhow!(
                    "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (prefill grouped)"
                )
            })?;
            ops::moe_hash_route_batched(
                ctx.gpu,
                self.moe_hash_route_batched_k,
                gate_logits,
                tid2eid,
                token_ids,
                indices_dev,
                weights_dev,
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                ctx.config.routed_scaling_factor as f32,
                n,
                stream,
            )?;
        } else if let Some(bias) = self.correction_bias_dev {
            // DeepSeek-V4 scores experts with sqrtsoftplus (NOT sigmoid); the
            // bias selects experts, weights gather pre-bias scores. Other
            // sigmoid+bias models (DeepSeek-V3 / MiniMax-M2) keep sigmoid.
            if ctx.config.scoring_func == "sqrtsoftplus" {
                ops::moe_topk_sqrtsoftplus_batched(
                    ctx.gpu,
                    self.moe_topk_sqrtsoftplus_batched_k,
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    ctx.config.norm_topk_prob,
                    ctx.config.routed_scaling_factor as f32,
                    n,
                    stream,
                )?;
            } else if ctx.config.scoring_func == "softmax" {
                self.router_softmax_bias_batched(
                    gate_logits,
                    bias,
                    indices_dev,
                    weights_dev,
                    num_experts,
                    top_k,
                    n,
                    ctx,
                    stream,
                )?;
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
                    n,
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
                n,
                stream,
            )?;
        }
        Ok(())
    }
}
