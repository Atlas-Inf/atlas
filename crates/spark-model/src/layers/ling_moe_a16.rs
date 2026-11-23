// SPDX-License-Identifier: AGPL-3.0-only

//! BF16-activation / NVFP4-weight Ling FFN path.
//!
//! Primitive's Ling-3.2-air-NVFP4A16 checkpoint leaves the router and shared
//! expert dense, while all routed expert matrices are resident compressed-
//! tensors NVFP4.  Keep this path separate from the original FP8 pager so a
//! format mismatch cannot silently reinterpret a packed tensor.

use crate::layers::moe::ExpertPtrTable;
use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::{EXPERTS, LingMoeFfn, TOP_K, ops, routed_swiglu_limit, shared_swiglu_limit};
use crate::layer::ForwardContext;

impl LingMoeFfn {
    /// Batched A16 prefill.  Shared BF16 projections use tensor-core GEMM;
    /// routed NVFP4 projections reuse the existing device-pointer GEMV with
    /// its token dimension in `grid.z`, so no expert ids or weights cross the
    /// host boundary.
    pub(super) fn forward_a16_prefill(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        ensure!(self.a16, "Ling A16 prefill called for strict FP8 weights");
        ensure!(
            num_tokens > 0,
            "Ling A16 prefill requires at least one token"
        );
        ensure!(ctx.config.model_type == "bailing_moe_v3_vl");
        ensure!(ctx.config.hidden_size == self.hidden);
        let rows = u32::try_from(num_tokens)
            .map_err(|_| anyhow::anyhow!("Ling A16 prefill row count exceeds CUDA launch range"))?;
        let dense_kernel = self
            .kernels
            .dense_gemm
            .ok_or_else(|| anyhow::anyhow!("Ling A16 dense GEMM kernel is unavailable"))?;
        let router_gemm = self
            .kernels
            .router_gemm
            .ok_or_else(|| anyhow::anyhow!("Ling A16 router GEMM kernel is unavailable"))?;
        let silu_kernel = self
            .kernels
            .silu_bf16
            .ok_or_else(|| anyhow::anyhow!("Ling A16 BF16 SwiGLU kernel is unavailable"))?;
        let expert_kernel = self
            .kernels
            .moe_wide
            .ok_or_else(|| anyhow::anyhow!("Ling A16 batched expert GEMV kernel is unavailable"))?;
        let blend_kernel = self
            .kernels
            .blend_rows
            .ok_or_else(|| anyhow::anyhow!("batched MoE blend kernel is unavailable"))?;
        let tables = self
            .a16_tables
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("Ling A16 pointer tables are unavailable"))?;
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

        let bf16 = std::mem::size_of::<u16>();
        let inter_bytes = num_tokens
            .checked_mul(self.intermediate)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("Ling A16 shared activation size overflow"))?;
        let route_inter_bytes = num_tokens
            .checked_mul(TOP_K)
            .and_then(|n| n.checked_mul(self.intermediate))
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("Ling A16 routed activation size overflow"))?;
        let hidden_bytes = num_tokens
            .checked_mul(self.hidden)
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("Ling A16 hidden size overflow"))?;
        let route_down_bytes = num_tokens
            .checked_mul(TOP_K)
            .and_then(|n| n.checked_mul(self.hidden))
            .and_then(|n| n.checked_mul(bf16))
            .ok_or_else(|| anyhow::anyhow!("Ling A16 routed down size overflow"))?;
        let route_meta_bytes = num_tokens
            .checked_mul(TOP_K)
            .and_then(|n| n.checked_mul(2 * std::mem::size_of::<u32>()))
            .ok_or_else(|| anyhow::anyhow!("Ling A16 route metadata size overflow"))?;
        let sizes = ctx.buffers.sizes();
        ensure!(
            inter_bytes <= sizes.expert_gate_out && inter_bytes <= sizes.expert_up_out,
            "Ling A16 shared activation arena is too small for {num_tokens} rows"
        );
        ensure!(
            route_inter_bytes <= sizes.expert_gate_out && route_inter_bytes <= sizes.expert_up_out,
            "Ling A16 routed activation arena is too small for {num_tokens} rows"
        );
        ensure!(
            route_down_bytes <= sizes.expert_down_out,
            "Ling A16 routed output arena is too small for {num_tokens} rows"
        );
        ensure!(
            hidden_bytes <= sizes.attn_output && hidden_bytes <= sizes.moe_output,
            "Ling A16 hidden arena is too small for {num_tokens} rows"
        );
        ensure!(
            route_meta_bytes <= sizes.gate_logits,
            "Ling A16 route metadata arena is too small for {num_tokens} rows"
        );

        // Shared expert first.  Its gate/up buffers are intentionally reused
        // for routed rows after the down GEMM has consumed the activation.
        ops::dense_gemm_tc(
            ctx.gpu,
            dense_kernel,
            input,
            &shared_gate,
            ctx.buffers.expert_gate_out(),
            rows,
            self.intermediate as u32,
            self.hidden as u32,
            stream,
        )?;
        ops::dense_gemm_tc(
            ctx.gpu,
            dense_kernel,
            input,
            &shared_up,
            ctx.buffers.expert_up_out(),
            rows,
            self.intermediate as u32,
            self.hidden as u32,
            stream,
        )?;
        ops::ling_silu_mul_bf16(
            ctx.gpu,
            silu_kernel,
            ctx.buffers.expert_gate_out(),
            ctx.buffers.expert_up_out(),
            ctx.buffers.expert_gate_out(),
            rows,
            self.intermediate as u32,
            shared_swiglu_limit(self.layer_idx),
            stream,
        )?;
        ops::dense_gemm_tc(
            ctx.gpu,
            dense_kernel,
            ctx.buffers.expert_gate_out(),
            &shared_down,
            ctx.buffers.attn_output(),
            rows,
            self.hidden as u32,
            self.intermediate as u32,
            stream,
        )?;

        // Keep routing entirely on device.  The existing Ling router block
        // indexes logits by blockIdx.x, so launching one block per row gives
        // exactly the one-token tie-breaking implementation for every row.
        ops::dense_gemm(
            ctx.gpu,
            router_gemm,
            input,
            &self.router,
            ctx.buffers.gate_logits_f32(),
            rows,
            EXPERTS as u32,
            self.hidden as u32,
            stream,
        )?;
        let ids = ctx.buffers.gate_logits();
        let weights = ids.offset(num_tokens * TOP_K * std::mem::size_of::<u32>());
        ops::ling_grouped_top8_rows(
            ctx.gpu,
            self.kernels.router,
            ctx.buffers.gate_logits_f32(),
            self.expert_bias,
            ids,
            weights,
            rows,
            stream,
        )?;

        if rows >= 128 {
            return self.grouped_a16_prefill(input, ids, weights, rows, ctx, stream);
        }
        // The common pointer-table expert GEMV already supports a token grid
        // in z.  Gate/up share the input row; down consumes each routed row
        // with input_stride=intermediate.
        ops::moe_expert_gemv_wide(
            ctx.gpu,
            expert_kernel,
            input,
            tables.gate.packed_ptrs,
            tables.gate.scale_ptrs,
            tables.gate.scale2_vals,
            ctx.buffers.expert_gate_out(),
            ids,
            self.intermediate as u32,
            self.hidden as u32,
            TOP_K as u32,
            0,
            rows,
            stream,
        )?;
        ops::moe_expert_gemv_wide(
            ctx.gpu,
            expert_kernel,
            input,
            tables.up.packed_ptrs,
            tables.up.scale_ptrs,
            tables.up.scale2_vals,
            ctx.buffers.expert_up_out(),
            ids,
            self.intermediate as u32,
            self.hidden as u32,
            TOP_K as u32,
            0,
            rows,
            stream,
        )?;
        let routed_rows = rows
            .checked_mul(TOP_K as u32)
            .ok_or_else(|| anyhow::anyhow!("Ling A16 routed row count overflow"))?;
        ops::ling_silu_mul_bf16(
            ctx.gpu,
            silu_kernel,
            ctx.buffers.expert_gate_out(),
            ctx.buffers.expert_up_out(),
            ctx.buffers.expert_gate_out(),
            routed_rows,
            self.intermediate as u32,
            routed_swiglu_limit(self.layer_idx),
            stream,
        )?;
        ops::moe_expert_gemv_wide(
            ctx.gpu,
            expert_kernel,
            ctx.buffers.expert_gate_out(),
            tables.down.packed_ptrs,
            tables.down.scale_ptrs,
            tables.down.scale2_vals,
            ctx.buffers.expert_down_out(),
            ids,
            self.hidden as u32,
            self.intermediate as u32,
            TOP_K as u32,
            self.intermediate as u32,
            rows,
            stream,
        )?;
        // A null gate weight selects the common kernel's always-on shared
        // branch.  This is Ling's residual separation: shared output is added
        // here, while the caller adds the FFN output to the transformer
        // residual exactly once.
        ops::moe_weighted_sum_blend_rows(
            ctx.gpu,
            blend_kernel,
            ctx.buffers.moe_output(),
            ctx.buffers.expert_down_out(),
            weights,
            ctx.buffers.attn_output(),
            DevicePtr::NULL,
            DevicePtr::NULL,
            self.hidden as u32,
            TOP_K as u32,
            0,
            rows,
            self.hidden as u32,
            stream,
        )?;
        Ok(())
    }

    fn grouped_a16_prefill(
        &self,
        input: DevicePtr,
        ids: DevicePtr,
        weights: DevicePtr,
        rows: u32,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<()> {
        let [sort, gemm, unpermute] = self
            .kernels
            .grouped_a16
            .expect("A16 grouped kernels loaded");
        let tables = self.a16_tables.as_ref().expect("A16 expert tables loaded");
        let expanded = rows as usize * TOP_K;
        // Routing has consumed the FP32 logits. Reuse that allocation for the
        // permutation, while retaining the chosen ids and weights in gate_logits.
        ensure!(
            (expanded * 3 + EXPERTS + 1) * 4 <= ctx.buffers.sizes().gate_logits_f32,
            "Ling A16 sort workspace too small"
        );
        let sorted = ctx.buffers.gate_logits_f32();
        let expert_ids = sorted.offset(expanded * 4);
        let offsets = expert_ids.offset(expanded * 4);
        let inverse = offsets.offset((EXPERTS + 1) * 4);
        ops::moe_sort_by_expert(
            ctx.gpu,
            sort,
            ids,
            sorted,
            expert_ids,
            offsets,
            inverse,
            expanded as u32,
            EXPERTS as u32,
            TOP_K as u32,
            stream,
        )?;
        let chunks = expanded.div_ceil(EXPERTS * 16).clamp(1, 4) as u32;
        let project = |x, table: &ExpertPtrTable, permutation, output, n: usize, k: usize| {
            KernelLaunch::new(ctx.gpu, gemm)
                .grid([n.div_ceil(32) as u32, chunks, EXPERTS as u32])
                .block([128, 1, 1])
                .arg_ptr(x)
                .arg_ptr(table.packed_ptrs)
                .arg_ptr(table.scale_ptrs)
                .arg_ptr(table.scale2_vals)
                .arg_ptr(permutation)
                .arg_ptr(offsets)
                .arg_ptr(output)
                .arg_u32(n as u32)
                .arg_u32(k as u32)
                .launch(stream)
        };
        let gate = ctx.buffers.expert_gate_out();
        let up = ctx.buffers.expert_up_out();
        project(
            input,
            &tables.gate,
            sorted,
            gate,
            self.intermediate,
            self.hidden,
        )?;
        project(
            input,
            &tables.up,
            sorted,
            up,
            self.intermediate,
            self.hidden,
        )?;
        ops::ling_silu_mul_bf16(
            ctx.gpu,
            self.kernels.silu_bf16.expect("A16 SwiGLU"),
            gate,
            up,
            gate,
            expanded as u32,
            self.intermediate as u32,
            routed_swiglu_limit(self.layer_idx),
            stream,
        )?;
        let routed = ctx.buffers.expert_down_out();
        project(
            gate,
            &tables.down,
            DevicePtr::NULL,
            routed,
            self.hidden,
            self.intermediate,
        )?;
        ops::moe_unpermute_reduce_indexed(
            ctx.gpu,
            unpermute,
            routed,
            ctx.buffers.moe_output(),
            inverse,
            weights,
            self.hidden as u32,
            rows,
            TOP_K as u32,
            stream,
        )?;
        ops::residual_add(
            ctx.gpu,
            self.kernels.residual_add,
            ctx.buffers.moe_output(),
            ctx.buffers.attn_output(),
            self.hidden as u32 * rows,
            stream,
        )
    }

    pub(super) fn forward_a16(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext<'_>,
        stream: u64,
    ) -> Result<DevicePtr> {
        ensure!(ctx.config.model_type == "bailing_moe_v3_vl");
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
