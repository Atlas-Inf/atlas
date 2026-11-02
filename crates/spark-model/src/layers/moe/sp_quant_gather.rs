// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_SP_MOE_QUANT_GATHER=1`: a four-rank sequence-parallel MoE
//! gathers its input already routed and quantized.
//!
//! The seam before a chain MoE all-gathers the BF16 normed rows (8 KiB a row
//! at hidden 4096) so that every rank can route and quantize all of them.
//! Every consumer of those peer rows is row-local: the router GEMM on its
//! CUTLASS tiles (no split K, a config set by N and K alone), the sigmoid
//! top-k (a CTA a token) and the NVFP4 quantization (a CTA a row, scale2
//! 1.0). The shared expert and its blend take this rank's rows only. So each
//! rank routes and quantizes its own rows, and the gather moves what the
//! routed FFN reads -- the packed NVFP4 row, its E4M3 group scales and the
//! row's top-k ids and weights: 2368 bytes a row, in one staged gather. The
//! sort, the K128W gate/up and down and the unpermute then see the bytes the
//! BF16 gather would have led to.

use super::route_fuse::Routes;
use super::*;
use crate::layers::glm_sp::SpRows;
use std::cell::Cell;

/// Whether the switch is on.
fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_SP_MOE_QUANT_GATHER").as_deref() == Ok("1"))
}

/// The bytes of one gathered row: packed NVFP4 (`h / 2`), group scales
/// (`h / 16`), top-k ids and weights (`top_k * 4` each), in that order.
fn staged_row(h: usize, top_k: usize) -> [usize; 4] {
    [h / 2, h / 16, top_k * 4, top_k * 4]
}

impl MoeLayer {
    /// Whether this layer's SP prefill over `sp` gathers its input routed
    /// and quantized: the caller then gathers no BF16 rows, and
    /// [`Self::forward_prefill`] reads only this rank's rows of its input.
    pub fn sp_quant_gather(&self, sp: SpRows, ctx: &ForwardContext) -> bool {
        let (n, h) = (sp.total() as u32, ctx.config.hidden_size as u32);
        let inter = ctx.config.routed_inter_local() as u32;
        requested()
            && sp.ranks > 2
            && !sp.full_shared
            && ctx.config.model_type == "glm5_next"
            && ctx.comm.is_some()
            && ctx.config.ep_world_size > 1
            && !ctx.graph_capture
            && !crate::det_trace::on()
            && !super::dump::enabled()
            && self.lora.is_none()
            && self.tid2eid_dev.is_none()
            && self.correction_bias_dev.is_some()
            && ctx.config.scoring_func == "sigmoid"
            && self.weights.router_pre_norm.is_none()
            && self.pre_expert_norm.is_none()
            && self.router_logits_n as usize == ctx.config.num_experts
            && !self.route_fused_applies(n, h, inter, ctx)
            && self.router_prefill_row_blocks(n, sp.rows as u32, ctx)
            && self.k128w_prequant(n, h, inter, self.verify_decode_batch(false, ctx, n))
            && !self.ep_prefill_zeroes(ctx)
            && !self.q38_routed_serves(h, inter, ctx)
            && h.is_multiple_of(32)
    }

    /// Where the quantized gather keeps the routes, the NVFP4 input and the
    /// staged rows of an `[n, h]` pass: `(routes, a_packed, a_scale, staged,
    /// staged row bytes)`.
    fn quant_gather_layout(
        &self,
        n: usize,
        ctx: &ForwardContext,
    ) -> Result<(Routes, DevicePtr, DevicePtr, DevicePtr, usize)> {
        let h = ctx.config.hidden_size;
        let (ne, top_k) = (ctx.config.num_experts, ctx.config.num_experts_per_tok);
        let te = n * top_k;
        // The layout `route_and_sort` leaves.
        let gate_logits = ctx.buffers.gate_logits();
        let scratch = ctx.buffers.scratch();
        let routes = Routes {
            indices: scratch,
            weights: scratch.offset(te * 4),
            sorted_token_ids: gate_logits,
            expert_offsets: gate_logits.offset(te * 4 * 2),
            token_to_perm: gate_logits.offset(te * 4 * 2 + (ne + 1) * 4),
            prerouted: false,
        };
        // `prequant_fp4_gate_up`'s input.
        let a_packed = ctx.buffers.expert_down_out();
        let a_scale = a_packed.offset(n * h / 2);
        let row: usize = staged_row(h, top_k).iter().sum();
        let staged = ctx.gpu.op_cache().scratch(
            ctx.gpu,
            "glm_sp_moe_quant_gather",
            ctx.buffers.max_batch_tokens() * row,
        )?;
        Ok((routes, a_packed, a_scale, staged, row))
    }

    /// The staged rows the quantized gather moves (`[total, width]` in BF16
    /// widths), for a seam that gathers them per block
    /// (`qwen3_attention::SeamGather`).
    pub(crate) fn sp_route_staging(
        &self,
        sp: SpRows,
        ctx: &ForwardContext,
    ) -> Result<(DevicePtr, usize)> {
        let (.., staged, row) = self.quant_gather_layout(sp.total(), ctx)?;
        Ok((staged, row / 2))
    }

    /// Route, quantize and stage this rank's rows `[r0, r0 + rows)` of `input`
    /// (`[sp.total(), h]`): the router logits (CUTLASS tiles, the whole
    /// call's bytes at any row count), top-k and NVFP4 rows. The last block
    /// leaves the gather to the seam (`take_routed_ahead`).
    pub(crate) fn sp_route_rows(
        &self,
        input: DevicePtr,
        sp: SpRows,
        (r0, rows): (usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let (ne, top_k) = (ctx.config.num_experts, ctx.config.num_experts_per_tok);
        let (routes, a_packed, a_scale, staged, row) = self.quant_gather_layout(sp.total(), ctx)?;
        let first = sp.row0 + r0;
        let route = top_k * 4;
        let ours = |base: DevicePtr, width: usize| base.offset(first * width);
        let block_in = ours(input, h * 2);
        let logits = ours(
            ctx.buffers.gate_logits(),
            ne * ctx.config.router_logit_bytes(),
        );
        self.router_bf16_gemm(
            block_in,
            logits,
            [rows as u32, self.router_logits_n, h as u32],
            stream,
        )?;
        self.prefill_topk(
            logits,
            ours(routes.indices, route),
            ours(routes.weights, route),
            rows as u32,
            ne as u32,
            top_k as u32,
            ctx,
            stream,
        )?;
        ops::quantize_bf16_to_nvfp4_scaled(
            ctx.gpu,
            self.quantize_nvfp4_k,
            block_in,
            ours(a_packed, h / 2),
            ours(a_scale, h / 16),
            self.a1_scale2(),
            rows as u32,
            h as u32,
            stream,
        )?;
        let parts = [a_packed, a_scale, routes.indices, routes.weights];
        let mut column = 0;
        for (&part, &width) in parts.iter().zip(&staged_row(h, top_k)) {
            ctx.gpu.copy_d2d_2d_async(
                ours(part, width),
                width,
                ours(staged, row).offset(column),
                row,
                width,
                rows,
                stream,
            )?;
            column += width;
        }
        if r0 + rows == sp.rows {
            ROUTED_AHEAD.with(|r| r.set(true));
        }
        Ok(())
    }

    /// Steps 1-3 of `forward_prefill` over `sp` with the quantized gather:
    /// route and quantize this rank's rows of `input` into the routes and
    /// the routed FFN's NVFP4 input (`expert_down_out`), gather every rank's
    /// rows of them (unless the seam before did both, block by block), then
    /// sort all of them. Callers check [`Self::sp_quant_gather`].
    pub(super) fn route_sp_quantized(
        &self,
        input: DevicePtr,
        sp: SpRows,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Routes> {
        let h = ctx.config.hidden_size;
        let (ne, top_k) = (ctx.config.num_experts, ctx.config.num_experts_per_tok);
        let n = sp.total();
        let ahead = ROUTED_AHEAD.with(Cell::take);
        if ctx.gpu.op_cache().once("glm:sp_moe_quant_gather") {
            tracing::info!(
                "ATLAS_GLM_SP_MOE_QUANT_GATHER: MoE input gathered as NVFP4 rows + routes ({} rows a part, {})",
                sp.rows,
                if ahead {
                    "per seam block"
                } else {
                    "after the seam"
                }
            );
        }
        let (routes, a_packed, a_scale, staged, row) = self.quant_gather_layout(n, ctx)?;
        if !ahead {
            self.sp_route_rows(input, sp, (0, sp.rows), ctx, stream)?;
            ROUTED_AHEAD.with(|r| r.set(false));
            sp.all_gather(staged, row / 2, ctx, stream)?;
        }
        let parts = [a_packed, a_scale, routes.indices, routes.weights];
        let mut column = 0;
        for (&part, &width) in parts.iter().zip(&staged_row(h, top_k)) {
            ctx.gpu
                .copy_d2d_2d_async(staged.offset(column), row, part, width, width, n, stream)?;
            column += width;
        }
        // 3. The sort of every row, as `route_and_sort` runs it.
        let te = n * top_k;
        ops::moe_sort_by_expert(
            ctx.gpu,
            self.moe_sort_by_expert,
            routes.indices,
            routes.sorted_token_ids,
            ctx.buffers.gate_logits().offset(te * 4),
            routes.expert_offsets,
            routes.token_to_perm,
            te as u32,
            ne as u32,
            top_k as u32,
            stream,
        )?;
        Ok(routes)
    }
}

thread_local! {
    /// The seam before routed, quantized and gathered every block
    /// (`sp_route_rows`); `route_sp_quantized` takes it.
    static ROUTED_AHEAD: Cell<bool> = const { Cell::new(false) };
}

#[cfg(test)]
mod tests {
    use super::staged_row;

    #[test]
    fn a_staged_row_is_under_a_third_of_the_bf16_row() {
        let row: usize = staged_row(4096, 8).iter().sum();
        assert_eq!(row, 2368);
        assert!(row * 3 < 4096 * 2);
        assert!(row.is_multiple_of(16));
    }
}

#[cfg(test)]
#[path = "sp_quant_gather_test_gpu.rs"]
mod gpu_tests;
