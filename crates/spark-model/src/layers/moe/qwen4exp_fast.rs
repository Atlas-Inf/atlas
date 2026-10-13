// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_MOE_FAST=1`: dispatch of Qwen3.8-Flash-Next's unified-layout
//! MoE decode pair (`ops::qwen4exp_moe_{gate_up,silu_down}_t`).
//!
//! It takes the transposed-table (`_t`) launches of single-row decode
//! (`dispatch_unified_t_decode`), the K=2 and K=3 verify/batch kernels
//! (`_batch2_t`, `_batch3_t`), and `forward_batched`'s per-row loop, four
//! rows a launch: TP1's unified layout serves all of them, TP2's hybrid
//! layout reaches only the last. Every replaced launch writes the same
//! bytes, so the callers' LoRA hooks, blends and EP reductions are untouched.

use super::*;

impl MoeLayer {
    /// The fast pair serves this layer's transposed NVFP4 tables.
    pub(super) fn qwen4exp_fast_t(&self) -> bool {
        self.qwen4exp_moe_fast.ready()
            && !self.btile_storage.is_published()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
    }

    /// Routed + shared gate/up of `rows` rows of `input` on the transposed
    /// tables: `gate_out`/`up_out` `[rows * top_k, inter]`, `sh_*_out`
    /// `[rows, inter]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4exp_fast_gate_up(
        &self,
        ctx: &ForwardContext,
        input: DevicePtr,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        indices: DevicePtr,
        sh_gate_t: &QuantizedWeight,
        sh_gate_out: DevicePtr,
        sh_up_t: &QuantizedWeight,
        sh_up_out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let gate_t = self
            .gate_ptrs_t
            .as_ref()
            .expect("qwen4exp_fast_t: gate_ptrs_t");
        let up_t = self.up_ptrs_t.as_ref().expect("qwen4exp_fast_t: up_ptrs_t");
        ops::qwen4exp_moe_gate_up_t(
            ctx.gpu,
            self.qwen4exp_moe_fast.gate_up,
            input,
            gate_t.packed_ptrs,
            gate_t.scale_ptrs,
            gate_t.scale2_vals,
            gate_out,
            up_t.packed_ptrs,
            up_t.scale_ptrs,
            up_t.scale2_vals,
            up_out,
            indices,
            sh_gate_t,
            sh_gate_out,
            sh_up_t,
            sh_up_out,
            ctx.config.moe_intermediate_size as u32,
            ctx.config.hidden_size as u32,
            ctx.config.num_experts_per_tok as u32,
            rows,
            stream,
        )
    }

    /// SiLU(gate)*up then down for `rows` rows: `down_out`
    /// `[rows * top_k, hidden]`, `sh_down_out` `[rows, hidden]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn qwen4exp_fast_silu_down(
        &self,
        ctx: &ForwardContext,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_out: DevicePtr,
        indices: DevicePtr,
        sh_gate_in: DevicePtr,
        sh_up_in: DevicePtr,
        sh_down_t: &QuantizedWeight,
        sh_down_out: DevicePtr,
        rows: usize,
        stream: u64,
    ) -> Result<()> {
        let down_t = self
            .down_ptrs_t
            .as_ref()
            .expect("qwen4exp_fast_t: down_ptrs_t");
        ops::qwen4exp_moe_silu_down_t(
            ctx.gpu,
            self.qwen4exp_moe_fast.silu_down,
            gate_out,
            up_out,
            down_t.packed_ptrs,
            down_t.scale_ptrs,
            down_t.scale2_vals,
            down_out,
            indices,
            sh_gate_in,
            sh_up_in,
            sh_down_t,
            sh_down_out,
            ctx.config.hidden_size as u32,
            ctx.config.moe_intermediate_size as u32,
            ctx.config.num_experts_per_tok as u32,
            rows,
            stream,
        )
    }

    /// `forward_batched`'s transposed-table rows in one gate/up and one down
    /// launch per chunk of up to four rows instead of a pair per row; `gate`
    /// is its router output. Routing (each row's own top-k, into adjacent
    /// index/weight slots), each row's blend and each row's EP all-reduce are
    /// the loop's calls in the loop's order, so `moe_output` and the
    /// collectives are unchanged. Returns `false`, having launched nothing,
    /// wherever the loop would take another branch or hook.
    pub(super) fn forward_batched_qwen4exp_fast(
        &self,
        input: DevicePtr,
        num_tokens: usize,
        (gate_logits, fp32_gate, gate_elem): (DevicePtr, bool, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !(self.qwen4exp_fast_t()
            && self.use_t_layout_for_prefill()
            && num_tokens > 0
            && self.lora.is_none()
            && ctx
                .attn_metadata
                .as_ref()
                .is_none_or(|m| m.moe_row_adapter.is_null())
            && self.tid2eid_dev.is_none()
            && self.correction_bias_dev.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.has_mixed_bf16_shared_expert()
            && self.router_logits_n as usize == ctx.config.num_experts)
        {
            return Ok(false);
        }
        let h = ctx.config.hidden_size;
        let top_k = ctx.config.num_experts_per_tok;
        let bf16 = 2usize;
        let scratch = ctx.buffers.scratch();
        let indices = scratch;
        let weights = scratch.offset(num_tokens * top_k * 4);
        for t in 0..num_tokens {
            ops::moe_topk_softmax(
                ctx.gpu,
                if fp32_gate {
                    self.moe_topk_f32
                } else {
                    self.moe_topk
                },
                gate_logits.offset(t * self.router_logits_n as usize * gate_elem),
                indices.offset(t * top_k * 4),
                weights.offset(t * top_k * 4),
                ctx.config.num_experts as u32,
                top_k as u32,
                ctx.config.norm_topk_prob,
                stream,
            )?;
        }
        let last = (num_tokens - 1) * top_k * 4;
        super::dump::dump_expert_ids(
            ctx.gpu,
            stream,
            indices.offset(last),
            weights.offset(last),
            1,
            top_k as u32,
        )?;

        let null_qw = QuantizedWeight::null();
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // The loop's shared scratch (see the aliasing note in forward.rs),
        // here a chunk's rows deep, as the K=2/K=3 paths use it.
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_out = ctx.buffers.attn_output();
        for first in (0..num_tokens).step_by(ops::QWEN4EXP_MOE_MAX_ROWS) {
            let rows = (num_tokens - first).min(ops::QWEN4EXP_MOE_MAX_ROWS);
            let chunk_indices = indices.offset(first * top_k * 4);
            self.qwen4exp_fast_gate_up(
                ctx,
                input.offset(first * h * bf16),
                expert_gate_out,
                expert_up_out,
                chunk_indices,
                self.shared_gate_t.as_ref().unwrap_or(&null_qw),
                shared_gate_scratch,
                self.shared_up_t.as_ref().unwrap_or(&null_qw),
                shared_up_scratch,
                rows,
                stream,
            )?;
            self.qwen4exp_fast_silu_down(
                ctx,
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                chunk_indices,
                shared_gate_scratch,
                shared_up_scratch,
                self.shared_down_t.as_ref().unwrap_or(&null_qw),
                shared_out,
                rows,
                stream,
            )?;
            for t in 0..rows {
                let row = first + t;
                let output_t = ctx.buffers.moe_output().offset(row * h * bf16);
                ops::moe_weighted_sum_blend(
                    ctx.gpu,
                    self.moe_weighted_sum_blend,
                    output_t,
                    expert_down_out.offset(t * top_k * h * bf16),
                    weights.offset(row * top_k * 4),
                    shared_out.offset(t * h * bf16),
                    input.offset(row * h * bf16),
                    self.weights.shared_expert_gate.weight,
                    h as u32,
                    top_k as u32,
                    h as u32,
                    stream,
                )?;
                if let Some(comm) = ctx.comm
                    && ctx.config.ep_world_size > 1
                {
                    if ctx.graph_capture {
                        comm.all_reduce(output_t.0, h * 2)?;
                    } else {
                        comm.all_reduce_async(output_t.0, h * 2, stream)?;
                    }
                }
            }
        }
        Ok(true)
    }
}
