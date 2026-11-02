// SPDX-License-Identifier: AGPL-3.0-only

//! `MoeLayer::forward_rows_t`: the transposed-table decode arm of
//! `forward_row_local` for `rows` rows at once — the verify-batch case on
//! FP4-Local (`use_t_layout_for_decode`). Each row still gets its own router
//! GEMV and top-k (identical calls to the per-row arm); only the routed +
//! fused shared expert GEMVs are deduplicated: `moe_expert_union_t` maps the
//! `rows * top_k` slots onto the union of experts, then the `_t_rows`
//! gate/up and silu/down kernels stage each unique expert's tile once and
//! apply it to every row that selected it. Accumulation order per
//! (row, slot) is the single-row `_t` smem kernel's, so each row's expert
//! outputs are byte-identical to running the row alone.
//!
//! Reads ONLY the `*_ptrs_t` transposed tables — the row-major originals
//! (`gate_ptrs`/`up_ptrs`/`down_ptrs`) are freed on FP4-Local.
//!
//! Gated on [`Self::t_rows_eligible`]; `ATLAS_QWEN4EXP_BATCH_FAST_BISECT` bit
//! `ops::BISECT_MOE_T_ROWS` is the kill switch.

use super::*;

impl MoeLayer {
    /// Whether [`Self::forward_rows_t`] reproduces `forward_row_local`'s
    /// transposed-table decode arm for `rows` rows: the t-layout arm taken,
    /// the plain NVFP4 `_t` kernels (not btile / qwen4exp_fast / e8m0), no
    /// router or norm hook the per-row arm would interpose, all three dedup
    /// kernels resolved, and the bisect bit off.
    pub(super) fn t_rows_eligible(&self, rows: usize, ctx: &ForwardContext) -> bool {
        (2..=8).contains(&rows)
            && self.use_t_layout_for_decode()
            && !self.btile_storage.is_published()
            && !self.qwen4exp_fast_t()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.moe_expert_union_t_k.0 != 0
            && self.moe_expert_gate_up_shared_t_rows_k[rows].0 != 0
            && self.moe_expert_silu_down_shared_t_rows_k[rows].0 != 0
            && !ctx.levers.batch_bisect(ops::BISECT_MOE_T_ROWS)
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.nvfp4_mmq_layout
            && self.lora.is_none()
            && self.pre_expert_norm.is_none()
            && self.weights.router_pre_norm.is_none()
            && self.tid2eid_dev.is_none()
            && self.correction_bias_dev.is_none()
            && self.router_logits_n as usize == ctx.config.num_experts
    }

    /// `rows` rows of `input` (`[rows, h]` BF16) through the dedup `_t_rows`
    /// kernels — the routed+shared GEMVs read once for the union of experts,
    /// the per-row router/top-k and blend identical to `forward_row_local`.
    /// `output`/`shared_out` are the per-row destinations `forward_rows`
    /// computes (`shared_out` holds each row's shared-expert down output
    /// until `forward_ep_reduce` blends it). `_single_seq_decode` mirrors the
    /// per-row arm's signature; the LoRA hooks it would feed are excluded by
    /// `t_rows_eligible` (`self.lora.is_none()`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn forward_rows_t(
        &self,
        input: DevicePtr,
        rows: usize,
        output: DevicePtr,
        shared_out: DevicePtr,
        _single_seq_decode: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        static ENGAGED_LOGGED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if !ENGAGED_LOGGED.swap(true, std::sync::atomic::Ordering::Relaxed) {
            tracing::info!("forward_rows_t: verify-row dedup arm engaged ({rows} rows)");
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let bf16 = 2usize;
        let slots = rows * top_k as usize;

        // Scratch layout: indices | weights | u_experts | u_rowmask |
        // u_rowslot (8 B per slot) | u_count. 28 B/slot + 4 B; the per-row
        // arm's indices/weights land in the same place it writes them.
        let scratch = ctx.buffers.scratch();
        let indices = scratch;
        let weights = scratch.offset(slots * 4);
        let u_experts = scratch.offset(2 * slots * 4);
        let u_rowmask = scratch.offset(3 * slots * 4);
        let u_rowslot = scratch.offset(4 * slots * 4);
        let u_count = scratch.offset(4 * slots * 4 + 8 * slots);
        anyhow::ensure!(
            4 * slots * 4 + 8 * slots + 4 <= 32 * 1024,
            "MoE t_rows: {slots} slots overrun the routing scratch"
        );

        // Per-row router + top-k: the same launches the per-row arm makes,
        // landing adjacent so the union kernel sees `rows * top_k` ids.
        let gate_logits = ctx.buffers.gate_logits();
        let router_n = self.router_logits_n;
        for r in 0..rows {
            let in_r = input.offset(r * h as usize * bf16);
            let router_in = self.router_input(in_r, 1, h, ctx, stream)?;
            if let Some(ref nvfp4) = self.gate_nvfp4 {
                ops::w4a16_decode_gemv(
                    ctx.gpu,
                    self.w4a16_gemv,
                    self.w4a16_gemv_sw,
                    ctx.levers.gemv_sw,
                    router_in,
                    nvfp4,
                    gate_logits.offset(r * router_n as usize * bf16),
                    router_n,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv,
                    router_in,
                    &self.weights.gate,
                    gate_logits.offset(r * router_n as usize * bf16),
                    router_n,
                    h,
                    stream,
                )?;
            }
            ops::moe_topk_softmax(
                ctx.gpu,
                self.moe_topk,
                gate_logits.offset(r * router_n as usize * bf16),
                indices.offset(r * top_k as usize * 4),
                weights.offset(r * top_k as usize * 4),
                num_experts,
                top_k,
                ctx.config.norm_topk_prob,
                stream,
            )?;
        }

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

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // The per-row arm's shared-expert gate/up scratch, `rows` deep (each
        // row's shared gate/up at `+ r * inter * 2`), like forward_rows_pair.
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let null_qw = QuantizedWeight::null();
        let sh_gate_t = self.shared_gate_t.as_ref().unwrap_or(&null_qw);
        let sh_up_t = self.shared_up_t.as_ref().unwrap_or(&null_qw);
        let sh_down_t = self.shared_down_t.as_ref().unwrap_or(&null_qw);
        let gate_t = self.gate_ptrs_t.as_ref().expect("gate_ptrs_t under t_rows");
        let up_t = self.up_ptrs_t.as_ref().expect("up_ptrs_t under t_rows");
        let down_t = self.down_ptrs_t.as_ref().expect("down_ptrs_t under t_rows");

        ops::moe_expert_gate_up_shared_t_rows(
            ctx.gpu,
            self.moe_expert_gate_up_shared_t_rows_k[rows],
            input,
            gate_t.packed_ptrs,
            gate_t.scale_ptrs,
            gate_t.scale2_vals,
            expert_gate_out,
            up_t.packed_ptrs,
            up_t.scale_ptrs,
            up_t.scale2_vals,
            expert_up_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            slots as u32,
            sh_gate_t,
            shared_gate_scratch,
            sh_up_t,
            shared_up_scratch,
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
            expert_gate_out,
            expert_up_out,
            down_t.packed_ptrs,
            down_t.scale_ptrs,
            down_t.scale2_vals,
            expert_down_out,
            u_experts,
            u_rowmask,
            u_rowslot,
            u_count,
            slots as u32,
            shared_gate_scratch,
            shared_up_scratch,
            sh_down_t,
            shared_out,
            h,
            inter,
            top_k,
            rows as u32,
            256,
            stream,
        )?;

        if self.has_separate_shared_expert() {
            self.run_separate_shared_expert(
                input,
                rows as u32,
                h,
                shared_inter,
                shared_gate_scratch,
                shared_up_scratch,
                shared_out,
                ctx,
                true,
                stream,
            )?;
        }

        // Each row's blend, as `forward_row_local` runs it: under EP against
        // a zeroed shared row (the EP reduce adds the shared expert once).
        // (An MXFP8 shared expert's slice blends here: `shared_mxfp8.rs`.)
        let is_ep =
            ctx.comm.is_some() && ctx.config.ep_world_size > 1 && !self.shared_joins_reduce();
        let zero_row = if is_ep && !shared_out.is_null() {
            self.ep_blend_zero_row(ctx, stream)?
        } else {
            DevicePtr::NULL
        };
        for r in 0..rows {
            let shared_for_blend = if is_ep && !shared_out.is_null() {
                zero_row
            } else {
                shared_out.offset(r * h as usize * bf16)
            };
            ops::moe_weighted_sum_blend(
                ctx.gpu,
                self.moe_weighted_sum_blend,
                output.offset(r * h as usize * bf16),
                expert_down_out.offset(r * top_k as usize * h as usize * bf16),
                weights.offset(r * top_k as usize * 4),
                shared_for_blend,
                input.offset(r * h as usize * bf16),
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )?;
        }
        Ok(())
    }
}
