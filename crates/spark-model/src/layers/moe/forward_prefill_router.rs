// SPDX-License-Identifier: AGPL-3.0-only

//! LongCat-Flash router for batched prefill: softmax over
//! `num_experts + zero_expert_num` logits, `e_score_correction_bias` for
//! SELECTION only, and the zero-computation (identity) experts folded inside
//! the kernel. Decode twin lives in `forward.rs`.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::MoeLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn router_softmax_bias_batched(
        &self,
        gate_logits: DevicePtr,
        bias: DevicePtr,
        indices_dev: DevicePtr,
        weights_dev: DevicePtr,
        num_experts: u32,
        top_k: u32,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ops::moe_topk_softmax_bias_batched(
            ctx.gpu,
            self.moe_topk_softmax_bias_batched_k,
            gate_logits,
            bias,
            indices_dev,
            weights_dev,
            self.zero_accum_dev,
            self.router_logits_n,
            num_experts,
            top_k,
            ctx.config.norm_topk_prob,
            ctx.config.routed_scaling_factor as f32,
            n,
            stream,
        )
    }
}
