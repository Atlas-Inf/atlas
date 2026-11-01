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
            // ATLAS_QWEN4EXP_PREFILL_MOE: the same logits, byte for byte.
            if !self.try_q38_router(
                router_in,
                nvfp4,
                gate_logits,
                n,
                self.router_logits_n,
                h,
                ctx,
                stream,
            )? {
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
            }
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
        } else if self.try_bf16_router_gemm(router_in, gate_logits, n, h, ctx.gpu, stream)? {
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

    /// FP4-Local's BF16 router on `n` rows into `logits`, when the layer is
    /// flagged row-invariant (`set_row_invariant_bf16_router`): the shipped
    /// BF16 gate already sits in the layout the pipelined tile GEMM reads, so
    /// no NVFP4 dequant step is needed (the same kernel `try_q38_router` runs
    /// after dequantizing the NVFP4 pack's router). Chunk-invariant (every
    /// logit is an independent in-order m16n8k16 chain), which the
    /// cached-prefix gate and an SP rank routing its own rows
    /// (`forward_prefill_route_sp`) require. `Ok(false)`: not served,
    /// nothing launched.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn try_bf16_router_gemm(
        &self,
        router_in: DevicePtr,
        logits: DevicePtr,
        n: u32,
        h: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<bool> {
        if !self.row_invariant_bf16_router || !h.is_multiple_of(16) {
            return Ok(false);
        }
        let Ok(k_gemm) = gpu.kernel("gemm", "dense_gemm_bf16_pipelined") else {
            return Ok(false);
        };
        super::forward_prefill_route_sp::q38_router_gemm(
            gpu,
            k_gemm,
            [router_in, self.weights.gate.weight, logits],
            [n, self.router_logits_n, h],
            stream,
        )?;
        Ok(true)
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
