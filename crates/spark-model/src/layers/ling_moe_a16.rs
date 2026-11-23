// SPDX-License-Identifier: AGPL-3.0-only

//! BF16-activation / NVFP4-weight Ling FFN path.
//!
//! Primitive's Ling-3.2-air-NVFP4A16 checkpoint leaves the router and shared
//! expert dense, while all routed expert matrices are resident compressed-
//! tensors NVFP4.  Keep this path separate from the original FP8 pager so a
//! format mismatch cannot silently reinterpret a packed tensor.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{LingMoeFfn, ops, routed_swiglu_limit, shared_swiglu_limit};
use crate::layer::ForwardContext;

impl LingMoeFfn {
    pub(super) fn forward_a16(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(ctx.config.model_type == "bailing_moe_v3_vl");
        ensure!(
            !ctx.graph_capture,
            "Ling native A16 FFN does not support CUDA graph capture yet"
        );
        ensure!(ctx.config.hidden_size == self.hidden);
        let dense_kernel = self
            .kernels
            .dense_gemv
            .ok_or_else(|| anyhow::anyhow!("Ling A16 dense GEMV kernel is unavailable"))?;
        let silu_kernel = self
            .kernels
            .silu_bf16
            .ok_or_else(|| anyhow::anyhow!("Ling A16 BF16 SwiGLU kernel is unavailable"))?;

        let shared_gate = self
            .shared_gate
            .dense()
            .ok_or_else(|| anyhow::anyhow!("Ling A16 shared gate is not dense BF16"))?;
        let shared_up = self
            .shared_up
            .dense()
            .ok_or_else(|| anyhow::anyhow!("Ling A16 shared up is not dense BF16"))?;
        let shared_down = self
            .shared_down
            .dense()
            .ok_or_else(|| anyhow::anyhow!("Ling A16 shared down is not dense BF16"))?;
        ops::dense_gemv(
            ctx.gpu,
            dense_kernel,
            input,
            &shared_gate,
            ctx.buffers.expert_gate_out(),
            self.intermediate as u32,
            self.hidden as u32,
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            dense_kernel,
            input,
            &shared_up,
            ctx.buffers.expert_up_out(),
            self.intermediate as u32,
            self.hidden as u32,
            stream,
        )?;
        ops::ling_silu_mul_bf16(
            ctx.gpu,
            silu_kernel,
            ctx.buffers.expert_gate_out(),
            ctx.buffers.expert_up_out(),
            self.activation_bf16,
            1,
            self.intermediate as u32,
            shared_swiglu_limit(self.layer_idx),
            stream,
        )?;
        ops::dense_gemv(
            ctx.gpu,
            dense_kernel,
            self.activation_bf16,
            &shared_down,
            ctx.buffers.attn_output(),
            self.hidden as u32,
            self.intermediate as u32,
            stream,
        )?;

        let ids = ctx.buffers.scratch();
        let weights = ids.offset(8 * std::mem::size_of::<i32>());
        ensure!(ctx.buffers.scratch_bytes() >= 8 * 8 * std::mem::size_of::<u32>());
        ops::dense_gemv_bf16_fp32out(
            ctx.gpu,
            self.kernels.router_gemv,
            input,
            &self.router,
            ctx.buffers.gate_logits_f32(),
            512,
            self.hidden as u32,
            stream,
        )?;
        ops::ling_grouped_top8(
            ctx.gpu,
            self.kernels.router,
            ctx.buffers.gate_logits_f32(),
            self.expert_bias,
            ids,
            weights,
            stream,
        )?;
        let tables = self
            .a16_tables
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ling A16 pointer tables are unavailable"))?;
        let gate_out = self.a16_gate_up;
        let up_out = gate_out.offset(8 * self.intermediate * std::mem::size_of::<u16>());
        ops::moe_expert_gemv_gate_up(
            ctx.gpu,
            self.kernels
                .moe_gate_up
                .ok_or_else(|| anyhow::anyhow!("Ling A16 gate/up kernel is unavailable"))?,
            input,
            tables.gate.packed_ptrs,
            tables.gate.scale_ptrs,
            tables.gate.scale2_vals,
            gate_out,
            tables.up.packed_ptrs,
            tables.up.scale_ptrs,
            tables.up.scale2_vals,
            up_out,
            ids,
            self.intermediate as u32,
            self.hidden as u32,
            8,
            stream,
        )?;
        // The gate output is safe to reuse in-place: every thread reads its
        // gate/up pair before writing the same activation element.
        ops::ling_silu_mul_bf16(
            ctx.gpu,
            silu_kernel,
            gate_out,
            up_out,
            gate_out,
            8,
            self.intermediate as u32,
            routed_swiglu_limit(self.layer_idx),
            stream,
        )?;
        ops::moe_expert_gemv(
            ctx.gpu,
            self.kernels
                .moe_down
                .ok_or_else(|| anyhow::anyhow!("Ling A16 down kernel is unavailable"))?,
            gate_out,
            tables.down.packed_ptrs,
            tables.down.scale_ptrs,
            tables.down.scale2_vals,
            ctx.buffers.expert_down_out(),
            ids,
            self.hidden as u32,
            self.intermediate as u32,
            8,
            self.intermediate as u32,
            1,
            stream,
        )?;
        ops::ling_weighted_sum(
            ctx.gpu,
            self.kernels.weighted_sum,
            ctx.buffers.expert_down_out(),
            weights,
            ctx.buffers.moe_output(),
            self.hidden as u32,
            8,
            stream,
        )?;
        ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            ctx.buffers.moe_output(),
            ctx.buffers.attn_output(),
            self.hidden as u32,
            stream,
        )?;
        Ok(ctx.buffers.moe_output())
    }
}
