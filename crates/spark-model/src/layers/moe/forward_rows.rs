// SPDX-License-Identifier: AGPL-3.0-only

//! `MoeLayer::forward_rows`: several rows through exactly the arithmetic
//! [`MoeLayer::forward`] runs for one, with the weights read once for the
//! union of the rows' experts and the per-row collectives batched
//! (`ATLAS_QWEN4EXP_BATCH_FAST`, `model/qwen4exp_batch_fast.rs`).
//!
//! `forward` per row is router GEMV, top-k, the fused routed+shared gate/up
//! and silu/down, the weighted-sum blend (the shared expert left out under
//! EP), then the EP all-reduce of `[1, h]` and the shared expert added once.
//! The multi-sequence decode and verify loops called it once per row, so a C4
//! step paid four all-reduces per MoE layer (192 collectives a step, each a
//! round trip over the pair), re-read the router and the shared expert per
//! row, and launched the expert pair per row.
//!
//! Two forms, both byte-identical per row to `forward`:
//!
//! * **Row pair** ([`MoeLayer::forward_rows_pair`], the originals expert
//!   layout, 2+ rows): the router for every row in one weight pass
//!   (`w4a16_gemv_batchN`, rows byte-identical to the single-row GEMV), each
//!   row's own top-k, then ONE expert-sorted gate/up and ONE silu/down
//!   (`ops::Qwen4ExpMoeRows`): a gate/up CTA is the single-row kernel's CTA
//!   for one `(row, slot)`, ordered so every expert the rows share — and the
//!   shared expert — streams from DRAM once; a silu/down CTA stages one tile
//!   of an expert once for every row that picked it, each output the
//!   single-row kernel's operation sequence. Each row's blend is the
//!   single-row blend. Under `ATLAS_QWEN4EXP_BATCH_SMALL` the rows' top-k
//!   and blends are one launch each (`moe_topk_softmax_rows`,
//!   `moe_weighted_sum_blend_rows`: a block per row of the single-row body).
//!   Under `ATLAS_QWEN4EXP_MOE_UNITS` gate/up and silu/down are the expert
//!   units of `qwen4exp_moe_c8.cu` (up to 64 rows): each weight decoded once
//!   for all the rows that picked its expert, SiLU in the gate/up epilogue,
//!   the same output bytes.
//! * **Per-row local** (anything else): each row's local part is
//!   [`MoeLayer::forward_row_local`], the same calls on the same buffers.
//!
//! Then the EP tail runs once, over all rows ([`MoeLayer::forward_ep_reduce`]):
//! one all-reduce of `[rows, h]` (each element the same two-rank BF16 sum)
//! and one `moe_batched_blend` with a block per row (each block the
//! single-row kernel's arithmetic). Each row's shared-expert output keeps its
//! own home until then: row `r`'s at `attn_output + r * h`, the buffer
//! `forward` uses for row 0.

use super::*;

impl MoeLayer {
    /// Whether [`Self::forward_rows`] can serve this layer: `forward`'s own
    /// fused decode arm (not one of its grouped-prefill early returns).
    fn forward_rows_eligible(&self, ctx: &ForwardContext) -> bool {
        !(self.routed_scales_released || ctx.config.expert_tp || self.decode_via_grouped_arm())
    }

    /// Whether `forward_row_local` takes the originals-layout NVFP4 arm with
    /// the plain softmax router — the arm the row pair reproduces — with no
    /// hook (LoRA, pre-expert norm, router pre-norm, mixed BF16 shared
    /// expert) in between.
    fn rows_pair_eligible(&self, rows: usize, ctx: &ForwardContext) -> bool {
        let top_k = ctx.config.num_experts_per_tok;
        self.qwen4exp_moe_rows.ready()
            && !ctx.levers.batch_bisect(ops::BISECT_MOE_ROWS)
            && rows >= 2
            && rows * top_k * 3 * 4 <= 32 * 1024
            && rows * top_k <= ops::QWEN4EXP_MOE_ROWS_MAX_SLOTS
            && ctx.config.moe_intermediate_size == ops::QWEN4EXP_MOE_ROWS_SD_INTER as usize
            && ctx
                .config
                .hidden_size
                .is_multiple_of(ops::QWEN4EXP_MOE_ROWS_SD_TILE as usize)
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.nvfp4_mmq_layout
            && !self.btile_storage.is_published()
            && !self.use_t_layout_for_decode()
            && self.lora.is_none()
            && self.pre_expert_norm.is_none()
            && self.weights.router_pre_norm.is_none()
            && !self.has_mixed_bf16_shared_expert()
            && self.tid2eid_dev.is_none()
            && self.correction_bias_dev.is_none()
            && self.router_logits_n as usize == ctx.config.num_experts
            && if self.gate_nvfp4.is_some() {
                self.w4a16_batchm.has_base()
            } else {
                self.dense_gemv_batchm.0 != 0
            }
    }

    /// `rows` rows of `input` (`[rows, h]` BF16) -> `moe_output` rows
    /// `[0, rows)`, each byte-identical to `forward` on that row alone.
    /// `Ok(None)`, nothing launched, where `forward` would take a grouped
    /// prefill arm instead: the caller keeps its per-row loop. The flag is
    /// whether the layer's mHC post ran too: under EP with
    /// `ATLAS_QWEN4EXP_BATCH_SMALL`, `hc_post` fuses it with the rows'
    /// shared-expert blend (`moe_blend_hc_post`, T = rows), as the
    /// single-token decode-fuse tier does at T = 1.
    pub fn forward_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        hc_post: Option<ops::qwen4exp_decode_fuse::MoeHcPost>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<(DevicePtr, bool)>> {
        if rows == 0
            || !self.forward_rows_eligible(ctx)
            || ctx.levers.batch_bisect(ops::BISECT_MOE_FORWARD)
        {
            return Ok(None);
        }
        let h = ctx.config.hidden_size;
        let bf16 = 2usize;
        // `attn_output` and `moe_output` are sized for a prefill chunk; a
        // decode or verify batch is far inside either.
        anyhow::ensure!(
            rows <= ctx.buffers.max_batch_tokens(),
            "MoE forward_rows: {rows} rows exceed the {}-token arena",
            ctx.buffers.max_batch_tokens()
        );
        self.btile_input_guard(input, rows, ctx, stream)?;
        // Same LoRA stance as `forward` on a multi-sequence row: refused.
        let single_seq_decode = ctx.attn_metadata.as_ref().map_or(1, |m| m.num_seqs) <= 1;
        if !single_seq_decode {
            self.reject_decode_lora(ctx, "forward_rows")?;
        }
        let output = ctx.buffers.moe_output();
        let shared_out = ctx.buffers.attn_output();
        if self.rows_pair_eligible(rows, ctx) {
            self.forward_rows_pair(input, rows, output, shared_out, ctx, stream)?;
        } else {
            for r in 0..rows {
                let off = r * h * bf16;
                self.forward_row_local(
                    input.offset(off),
                    output.offset(off),
                    shared_out.offset(off),
                    single_seq_decode,
                    ctx,
                    stream,
                )?;
            }
        }
        let posted =
            self.forward_ep_reduce(output, shared_out, input, rows, hc_post, ctx, stream)?;
        Ok(Some((output, posted)))
    }

    /// The row pair (module docs): `forward_row_local`'s originals arm for
    /// `rows` rows with the router, the experts and the shared expert read
    /// once. Same buffers as the single-row arm, `rows` deep.
    fn forward_rows_pair(
        &self,
        input: DevicePtr,
        rows: usize,
        output: DevicePtr,
        shared_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let inter = ctx.config.moe_intermediate_size;
        let num_experts = ctx.config.num_experts;
        let top_k = ctx.config.num_experts_per_tok;
        let bf16 = 2usize;

        // Router: every row's logits from one pass over the gate weight.
        let gate_logits = ctx.buffers.gate_logits();
        let router_n = self.router_logits_n;
        if let Some(ref nvfp4) = self.gate_nvfp4 {
            // The widest exact tier a launch (batch16 past 8 rows).
            self.wide_rows.w4a16_rows(
                ctx.gpu,
                &self.w4a16_batchm,
                input,
                nvfp4,
                gate_logits,
                (rows as u32, router_n, h as u32),
                stream,
            )?;
        } else {
            self.wide_rows.dense_rows(
                ctx.gpu,
                self.dense_gemv_batchm,
                input,
                &self.weights.gate,
                gate_logits,
                (rows as u32, router_n, h as u32, router_n),
                stream,
            )?;
        }

        // Each row's own top-k, into adjacent slots; the plan's order after.
        let scratch = ctx.buffers.scratch();
        let slots = rows * top_k;
        let indices = scratch;
        let weights = scratch.offset(slots * 4);
        let order = scratch.offset(2 * slots * 4);
        let k = &self.qwen4exp_moe_rows;
        // ATLAS_QWEN4EXP_BATCH_SMALL: every row's top-k in one launch, each
        // the single-row kernel's bytes; else one launch a row.
        let topk_once = ctx.levers.qwen4exp_batch_small
            && k.topk_rows(
                ctx.gpu,
                gate_logits,
                indices,
                weights,
                (num_experts as u32, top_k as u32, rows as u32, router_n),
                ctx.config.norm_topk_prob,
                stream,
            )?;
        for r in (0..rows).filter(|_| !topk_once) {
            ops::moe_topk_softmax(
                ctx.gpu,
                self.moe_topk,
                gate_logits.offset(r * router_n as usize * bf16),
                indices.offset(r * top_k * 4),
                weights.offset(r * top_k * 4),
                num_experts as u32,
                top_k as u32,
                ctx.config.norm_topk_prob,
                stream,
            )?;
        }
        if let Some(path) = crate::model::qwen4exp_batch_fast::route_dump_path() {
            let launch = super::route_dump::Launch {
                indices,
                weights,
                input,
                rows,
                top_k,
                hidden: h,
                layer: self.gate_ptrs.packed_ptrs.0,
            };
            super::route_dump::append(ctx.gpu, &launch, path, stream)?;
        }

        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        let (h32, inter32, top_k32, rows32) = (h as u32, inter as u32, top_k as u32, rows as u32);
        let table = |t: &ExpertPtrTable| (t.packed_ptrs, t.scale_ptrs, t.scale2_vals);
        let shared = &self.weights.shared_expert;
        if k.units_ready() {
            // A launch takes up to QWEN4EXP_MOE_UNITS_MAX_ROWS rows; each row's
            // bytes are the same in any chunk (and under ATLAS_QWEN4EXP_MOE_TC
            // the kernels are row-invariant).
            for r0 in (0..rows).step_by(ops::QWEN4EXP_MOE_UNITS_MAX_ROWS) {
                let n = ops::QWEN4EXP_MOE_UNITS_MAX_ROWS.min(rows - r0);
                self.units_experts(input, indices, (r0, n), shared_out, ctx, stream)?;
            }
        } else {
            k.plan(ctx.gpu, indices, order, slots, stream)?;
            // The single-row arm's shared gate/up scratch, `rows` deep.
            let shared_gate_scratch = ctx.buffers.logits();
            let shared_up_scratch = ctx.buffers.ssm_qkvz();
            k.gate_up(
                ctx.gpu,
                input,
                table(&self.gate_ptrs),
                expert_gate_out,
                table(&self.up_ptrs),
                expert_up_out,
                indices,
                order,
                &shared.gate_proj,
                shared_gate_scratch,
                &shared.up_proj,
                shared_up_scratch,
                (inter32, h32, top_k32, rows32),
                stream,
            )?;
            k.silu_down(
                ctx.gpu,
                expert_gate_out,
                expert_up_out,
                table(&self.down_ptrs),
                expert_down_out,
                indices,
                order,
                shared_gate_scratch,
                shared_up_scratch,
                &shared.down_proj,
                shared_out,
                (h32, inter32, top_k32, rows32),
                stream,
            )?;
        }

        // Each row's blend, as `forward_row_local` runs it: under EP against a
        // zeroed shared row (the EP reduce adds the shared expert once).
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let zero_row = if is_ep {
            self.ep_blend_zero_row(ctx, stream)?
        } else {
            DevicePtr::NULL
        };
        // ATLAS_QWEN4EXP_BATCH_SMALL: the rows' blends in one launch (the EP
        // zero row shared at stride 0).
        let blend_once = ctx.levers.qwen4exp_batch_small
            && k.blend_rows(
                ctx.gpu,
                output,
                expert_down_out,
                weights,
                if is_ep {
                    (zero_row, 0)
                } else {
                    (shared_out, h32)
                },
                input,
                self.weights.shared_expert_gate.weight,
                (h32, top_k32, rows32),
                stream,
            )?;
        for r in (0..rows).filter(|_| !blend_once) {
            ops::moe_weighted_sum_blend(
                ctx.gpu,
                self.moe_weighted_sum_blend,
                output.offset(r * h * bf16),
                expert_down_out.offset(r * top_k * h * bf16),
                weights.offset(r * top_k * 4),
                if is_ep {
                    zero_row
                } else {
                    shared_out.offset(r * h * bf16)
                },
                input.offset(r * h * bf16),
                self.weights.shared_expert_gate.weight,
                h32,
                top_k32,
                h32,
                stream,
            )?;
        }
        Ok(())
    }

    /// The expert units (`ATLAS_QWEN4EXP_MOE_UNITS`, or on tensor cores under
    /// `ATLAS_QWEN4EXP_MOE_TC`) for rows `[r0, r0 + rows)` of `input` routed at
    /// `indices` (`[*, top_k]`): down into `expert_down_out` (`[*, top_k, h]`),
    /// the shared expert's into `shared_out` (`[*, h]`), at those rows. The
    /// gate/up rows are not materialized, so their buffers hold the
    /// activations (FP32; BF16 under TC v3) and the plan. Serial decode's single row comes here too
    /// under the tensor-core switch (`forward_row_local`).
    pub(super) fn units_experts(
        &self,
        input: DevicePtr,
        indices: DevicePtr,
        (r0, rows): (usize, usize),
        shared_out: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (h, inter) = (ctx.config.hidden_size, ctx.config.moe_intermediate_size);
        let top_k = ctx.config.num_experts_per_tok;
        let slots = rows * top_k;
        let sizes = ctx.buffers.sizes();
        anyhow::ensure!(
            (slots + rows) * inter * 4 <= sizes.expert_gate_out
                && ops::QWEN4EXP_MOE_UNITS_WS_BYTES <= sizes.expert_up_out,
            "qwen4exp MoE units: {rows} rows overrun the expert gate/up buffers"
        );
        let table = |t: &ExpertPtrTable| (t.packed_ptrs, t.scale_ptrs, t.scale2_vals);
        let shared = &self.weights.shared_expert;
        self.qwen4exp_moe_rows.units(
            ctx.gpu,
            input.offset(r0 * h * 2),
            table(&self.gate_ptrs),
            table(&self.up_ptrs),
            table(&self.down_ptrs),
            indices.offset(r0 * top_k * 4),
            (ctx.buffers.expert_up_out(), ctx.buffers.expert_gate_out()),
            (&shared.gate_proj, &shared.up_proj, &shared.down_proj),
            ctx.buffers.expert_down_out().offset(r0 * top_k * h * 2),
            shared_out.offset(r0 * h * 2),
            (h as u32, inter as u32, top_k as u32, rows as u32),
            stream,
        )
    }
}
