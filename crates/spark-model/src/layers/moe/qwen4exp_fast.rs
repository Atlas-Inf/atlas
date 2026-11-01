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
        // gfx1151 has no qwen4exp_moe_fast kernels (cp.async.bulk prefetch;
        // `qwen4exp_moe_fast_requested` is off under atlas_hip), so this arm used
        // to fall through to the per-token loop: one routed+shared gate/up and
        // one silu/down launch per token per layer. That is every prefill piece
        // under ATLAS_HIP_MOE_GROUPED_MIN_TOKENS: the snapshot-split tail of a
        // cold prompt with SSM snapshots on (framework f169s8: 1,152 per-token
        // gate/up + 1,152 silu/down for a 24-token tail, ~0.45 s of a 2.2 s
        // TTFT) and every short agentic suffix after a prefix hit. There the
        // verify-row dedup kernels run the chunk instead (`rows_t`, up to 8 rows
        // a launch): byte-identical per (row, slot) to the single-row `_t`
        // kernels the loop launches, with the loop's own top-k and blends.
        let fast = self.qwen4exp_fast_t();
        let rows_t = !fast
            && cfg!(atlas_hip)
            && num_tokens >= 2
            && self.t_rows_eligible(2, ctx)
            && self.t_rows_eligible(T_ROWS_CHUNK, ctx)
            && t_rows_scratch_bytes(num_tokens, ctx.config.num_experts_per_tok) <= 32 * 1024;
        if !((fast || rows_t)
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
        // Union scratch for the rows_t arm, after every token's indices and weights.
        let u_base = scratch.offset(2 * num_tokens * top_k * 4);
        let u_slots = T_ROWS_CHUNK * top_k;
        let (u_experts, u_rowmask) = (u_base, u_base.offset(u_slots * 4));
        let u_rowslot = u_base.offset(2 * u_slots * 4);
        let u_count = u_base.offset(2 * u_slots * 4 + 8 * u_slots);
        let mut first = 0;
        while first < num_tokens {
            let rows = if fast {
                (num_tokens - first).min(ops::QWEN4EXP_MOE_MAX_ROWS)
            } else {
                t_rows_chunk(num_tokens - first)
            };
            let chunk_indices = indices.offset(first * top_k * 4);
            // A checkpoint-native separate shared expert never reaches the
            // fused kernels: NULL `_t` weights make them skip it, and the
            // separate chain fills `shared_out` for the same rows.
            let separate = self.has_separate_shared_expert();
            let (sh_gate_t, sh_up_t, sh_down_t) = if separate {
                (&null_qw, &null_qw, &null_qw)
            } else {
                (
                    self.shared_gate_t.as_ref().unwrap_or(&null_qw),
                    self.shared_up_t.as_ref().unwrap_or(&null_qw),
                    self.shared_down_t.as_ref().unwrap_or(&null_qw),
                )
            };
            if fast {
                self.qwen4exp_fast_gate_up(
                    ctx,
                    input.offset(first * h * bf16),
                    expert_gate_out,
                    expert_up_out,
                    chunk_indices,
                    sh_gate_t,
                    shared_gate_scratch,
                    sh_up_t,
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
                    sh_down_t,
                    shared_out,
                    rows,
                    stream,
                )?;
            } else {
                self.rows_t_experts(
                    input.offset(first * h * bf16),
                    rows,
                    chunk_indices,
                    (u_experts, u_rowmask, u_rowslot, u_count),
                    (expert_gate_out, expert_up_out, expert_down_out),
                    (shared_gate_scratch, shared_up_scratch, shared_out),
                    (sh_gate_t, sh_up_t, sh_down_t),
                    ctx,
                    stream,
                )?;
            }
            if separate {
                self.run_separate_shared_expert(
                    input.offset(first * h * bf16),
                    rows as u32,
                    h as u32,
                    ctx.config.shared_expert_intermediate_size as u32,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared_out,
                    ctx,
                    true,
                    stream,
                )?;
            }
            // EP fix (the `forward_row_local` contract): the shared expert is
            // replicated on all ranks, so the pre-reduce blend takes a zero
            // row and `forward_ep_reduce` adds `shared_out` once per row
            // after the all-reduce. Outside EP, each row blends ITS OWN
            // shared output (`shared_out.offset(t * h * bf16)`).
            // The zero row may itself be NULL (`ep_blend_zero_row` under
            // ATLAS_QWEN4EXP_DECODE_FUSE, read by the kernel as zeros), so
            // select on EP, never on the pointer.
            let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
            // (An MXFP8 shared expert's slice blends here: `shared_mxfp8.rs`.)
            let zero_row = if is_ep && !shared_out.is_null() && !self.shared_joins_reduce() {
                Some(self.ep_blend_zero_row(ctx, stream)?)
            } else {
                None
            };
            for t in 0..rows {
                let row = first + t;
                let output_t = ctx.buffers.moe_output().offset(row * h * bf16);
                let shared_t = zero_row.unwrap_or_else(|| shared_out.offset(t * h * bf16));
                ops::moe_weighted_sum_blend(
                    ctx.gpu,
                    self.moe_weighted_sum_blend,
                    output_t,
                    expert_down_out.offset(t * top_k * h * bf16),
                    weights.offset(row * top_k * 4),
                    shared_t,
                    input.offset(row * h * bf16),
                    self.weights.shared_expert_gate.weight,
                    h as u32,
                    top_k as u32,
                    h as u32,
                    stream,
                )?;
            }
            self.forward_ep_reduce(
                ctx.buffers.moe_output().offset(first * h * bf16),
                shared_out,
                input.offset(first * h * bf16),
                rows,
                None,
                ctx,
                stream,
            )?;
            first += rows;
        }
        Ok(true)
    }

    /// The routed + shared experts of one `rows`-row chunk through the
    /// verify-row dedup kernels (`moe_expert_union_t` + the `_t_rows` pair), as
    /// `forward_rows_t` launches them, from the chunk's top-k already in
    /// `indices`. Writes the per-row-loop layout: `down_out` rows
    /// `[(r * top_k + slot), h]` and `sh_out` rows `[r, h]`.
    #[allow(clippy::too_many_arguments)]
    fn rows_t_experts(
        &self,
        input: DevicePtr,
        rows: usize,
        indices: DevicePtr,
        (u_experts, u_rowmask, u_rowslot, u_count): (DevicePtr, DevicePtr, DevicePtr, DevicePtr),
        (gate_out, up_out, down_out): (DevicePtr, DevicePtr, DevicePtr),
        (sh_gate_scratch, sh_up_scratch, sh_out): (DevicePtr, DevicePtr, DevicePtr),
        (sh_gate_t, sh_up_t, sh_down_t): (&QuantizedWeight, &QuantizedWeight, &QuantizedWeight),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let slots = rows as u32 * top_k;
        let gate_t = self.gate_ptrs_t.as_ref().expect("gate_ptrs_t under rows_t");
        let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under rows_t");
        let down_t = self.down_ptrs_t.as_ref().expect("down_ptrs_t under rows_t");
        ops::moe_expert_union_t(
            ctx.gpu,
            self.moe_expert_union_t_k,
            indices,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            rows as u32,
            top_k,
            stream,
        )?;
        ops::moe_expert_gate_up_shared_t_rows(
            ctx.gpu,
            self.moe_expert_gate_up_shared_t_rows_k[rows],
            input,
            gate_t.packed_ptrs,
            gate_t.scale_ptrs,
            gate_t.scale2_vals,
            gate_out,
            up_t.packed_ptrs,
            up_t.scale_ptrs,
            up_t.scale2_vals,
            up_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            slots,
            sh_gate_t,
            sh_gate_scratch,
            sh_up_t,
            sh_up_scratch,
            inter,
            h,
            top_k,
            h,
            rows as u32,
            256,
            stream,
        )?;
        ops::moe_expert_silu_down_shared_t_rows(
            ctx.gpu,
            self.moe_expert_silu_down_shared_t_rows_k[rows],
            gate_out,
            up_out,
            down_t.packed_ptrs,
            down_t.scale_ptrs,
            down_t.scale2_vals,
            down_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            slots,
            sh_gate_scratch,
            sh_up_scratch,
            sh_down_t,
            sh_out,
            h,
            inter,
            top_k,
            rows as u32,
            256,
            stream,
        )
    }
}

/// Rows per rows_t chunk (the largest `_t_rows` instantiation).
const T_ROWS_CHUNK: usize = 8;

/// The next chunk of `remaining` rows: up to 8, never leaving a single row
/// behind (there is no 1-row `_t_rows` kernel), so 9 -> 7 + 2.
fn t_rows_chunk(remaining: usize) -> usize {
    let rows = remaining.min(T_ROWS_CHUNK);
    if remaining > rows && remaining - rows == 1 {
        rows - 1
    } else {
        rows
    }
}

/// Routing scratch the rows_t arm needs for `num_tokens` tokens: every token's
/// indices and weights, then one chunk's union buffers (u_experts, u_rowmask,
/// u_rowslot at 8 B a slot, u_count).
fn t_rows_scratch_bytes(num_tokens: usize, top_k: usize) -> usize {
    let u_slots = T_ROWS_CHUNK * top_k;
    2 * num_tokens * top_k * 4 + 2 * u_slots * 4 + 8 * u_slots + 4
}

#[cfg(test)]
mod rows_t_tail_tests {
    use super::{t_rows_chunk, t_rows_scratch_bytes};

    #[test]
    fn chunks_never_leave_one_row() {
        for n in 2..300 {
            let (mut left, mut total) = (n, 0);
            while left > 0 {
                let c = t_rows_chunk(left);
                assert!((2..=8).contains(&c), "n={n} chunk {c}");
                left -= c;
                total += c;
            }
            assert_eq!(total, n);
        }
    }

    #[test]
    fn scratch_fits_under_the_grouped_threshold() {
        // Flash-Next top-10: every sub-256-token piece fits the 32 KiB scratch.
        assert!(t_rows_scratch_bytes(255, 10) <= 32 * 1024);
    }
}
