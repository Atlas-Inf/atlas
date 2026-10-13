// SPDX-License-Identifier: AGPL-3.0-only

//! Pre-quantized native-FP4 routed-expert helpers.

use super::*;

#[derive(Clone, Copy)]
pub(super) struct CompactMoeWorklist {
    pub worklist: DevicePtr,
    pub total_tiles: DevicePtr,
    pub max_tiles: u32,
}

fn c3_grouped_shape(config: &atlas_core::config::ModelConfig, rows: u32, decode_rows: u32) -> bool {
    (rows == 3 || (rows.is_multiple_of(3) && rows <= 12 && OWNER_ROWS.with(|r| r.get()) == rows))
        && decode_rows >= 3
        && glm_grouped_shape(config)
}

thread_local! {
    /// Rows of an owner-batched long-context K3 verify FFN in progress, else 0.
    static OWNER_ROWS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Run `f` as the joint FFN of an owner-batched K3 verify of `rows` rows: the
/// C3 grouped arithmetic (router, routed and shared) is applied row-for-row as
/// in a single-owner K3 verify, with each routed expert read once for all rows.
pub(crate) fn with_owner_rows<R>(rows: u32, f: impl FnOnce() -> R) -> R {
    OWNER_ROWS.with(|r| r.set(rows));
    let out = f();
    OWNER_ROWS.with(|r| r.set(0));
    out
}

pub(super) fn c4_grouped_shape(
    config: &atlas_core::config::ModelConfig,
    rows: u32,
    decode_rows: u32,
) -> bool {
    rows == 4 && decode_rows >= 4 && glm_grouped_shape(config)
}

pub(super) fn glm_grouped_shape(config: &atlas_core::config::ModelConfig) -> bool {
    config.model_type == "glm5_next"
        && config.hidden_size == 4096
        && config.moe_intermediate_size == 2048
        && config.shared_expert_intermediate_size == 2048
        && config.num_experts == 288
        && config.num_experts_per_tok == 8
        && config.tp_world_size == 2
        && config.ep_world_size == 2
        && config.scoring_func == "sigmoid"
}

/// Row tiles of the K128W kernels over `rows` sorted rows: the device prefix
/// of the local experts' M64 tiles, the schedule covering them, and the
/// fused gate/up and down kernels to launch over it (the K128W pair, or
/// their M16 decode twins, whose `prefix` is the compact worklist scratch).
#[derive(Clone, Copy)]
pub(super) struct MtileGrid {
    pub prefix: DevicePtr,
    pub schedule: ops::K128wSchedule,
    pub rows: u32,
    pub gate_up_silu: ops::K128wKernel,
    pub down: ops::K128wKernel,
}

/// Sorted rows below which the persistent K128W schedule stays off: at 2K-4K
/// token chunks (16-32K rows) the grid kernels already stream the weights at
/// the DRAM rate and the persistent twins measured 2-7% slower on GB10; from
/// 8K tokens (65536 rows, the serving chunk) they measured 6-9% faster on
/// gate/up (moe_prefill_bench).
pub(super) const K128W_PERSIST_MIN_ROWS: u32 = 65536;

/// The K128W schedule over `rows` sorted rows: `persist_ctas` persistent CTAs
/// claiming work from `next_work` when enabled (nonzero) and the chunk is
/// large enough, else the grid over every sorted row in M64 tiles plus one
/// partial tile per expert.
pub(super) fn k128w_schedule(
    persist_ctas: u32,
    rows: u32,
    num_experts: u32,
    next_work: DevicePtr,
) -> ops::K128wSchedule {
    if persist_ctas > 0 && rows >= K128W_PERSIST_MIN_ROWS {
        ops::K128wSchedule::Persistent {
            ctas: persist_ctas,
            next_work,
        }
    } else {
        ops::K128wSchedule::Grid {
            bound: rows.div_ceil(64) + num_experts,
        }
    }
}

pub(super) fn compact_gate_up_worklist_bytes(rows: u32, top_k: u32, inter: u32) -> usize {
    16 + rows as usize * top_k as usize * inter.div_ceil(128) as usize * 8
}

impl MoeLayer {
    /// C3-only prototype: stateless MoE work is independent of sequence
    /// ownership. Keep the control's router and shared expert, changing only
    /// routed GEMV to the established prequantized native-FP4 grouped path.
    pub(super) fn glm_c3_grouped(&self, ctx: &ForwardContext, rows: u32) -> bool {
        // ATLAS_GLM_CANONICAL_VERIFY: no K=3-only arm (router, shared, tiles).
        !crate::layers::canonical_verify::enabled()
            && std::env::var("ATLAS_GLM_C3_GROUPED_MOE").as_deref() == Ok("1")
            && c3_grouped_shape(ctx.config, rows, ctx.levers.max_decode_seqs)
            && self.glm_grouped_resources(ctx)
            && self.w4a16_gemv_batch3.0 != 0
    }

    pub(super) fn glm_c4_grouped(&self, ctx: &ForwardContext, rows: u32) -> bool {
        crate::model::glm_c4::enabled(&ctx.config.model_type)
            && super::forward_c4::c4_grouped_requested()
            && c4_grouped_shape(ctx.config, rows, ctx.levers.max_decode_seqs)
            && ctx.attn_metadata.is_some_and(|m| m.num_seqs == 4)
            && self.glm_grouped_resources(ctx)
            && self.w4a16_batchm.kernel(4).0 != 0
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
    }

    pub(super) fn glm_grouped_resources(&self, ctx: &ForwardContext) -> bool {
        self.glm_native_moe_resources(ctx)
            && self.nvfp4_prequant_moe
            && self.nvfp4_fused_silu_quant
            && self.moe_w4a4_prequant_t_k64.0 != 0
            && self.moe_w4a4_prequant_t_k64_compact.0 != 0
            && self.moe_build_tile_worklist_k.0 != 0
            // Prequantization reuses expert_down_out before routed down.
            // Remote expert rows remain scratch, so reduction must skip them.
            && self.moe_unpermute_reduce_ep.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && self.silu_mul_quant_nvfp4_k.0 != 0
    }

    pub(super) fn glm_native_moe_resources(&self, ctx: &ForwardContext) -> bool {
        ctx.comm.is_some()
            && self.lora.is_none()
            && self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && !self.has_mixed_bf16_shared_expert()
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && !self.nvfp4_mmq_layout
            && self.use_btile_or_t_prefill()
            && self.gate_fp8.is_none()
            && self.gate_nvfp4.is_none()
            && self.correction_bias_dev.is_some()
            && self.tid2eid_dev.is_none()
            && self.pre_expert_norm.is_none()
            && self.weights.shared_expert_gate.weight.is_null()
            && !self.weights.shared_expert.gate_proj.is_null()
            && !self.weights.shared_expert.up_proj.is_null()
            && !self.weights.shared_expert.down_proj.is_null()
            && !super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
            && !super::forward_prefill_routed::grouped_cutlass_down_enabled()
    }

    /// Prequantized native-FP4 gate and up projections of the sorted rows.
    /// Returns true when the K128W launch also applied `silu_mul_quant_nvfp4`
    /// (its NVFP4 bytes in `expert_up_out`, as `fused_silu_prequant_fp4_down`
    /// leaves them), which then must not run.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prequant_fp4_gate_up(
        &self,
        expert_input: DevicePtr,
        gate: &ExpertPtrTable,
        up: &ExpertPtrTable,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        n: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        max_m_tiles: u32,
        compact: Option<CompactMoeWorklist>,
        wide: Option<MtileGrid>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        // expert_down_out is not consumed until after gate/up and has ample
        // capacity for packed token-major A followed by its group scales.
        let a_packed = ctx.buffers.expert_down_out();
        let a_scale = a_packed.offset(n as usize * h as usize / 2);
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.quantize_nvfp4_k,
            expert_input,
            a_packed,
            a_scale,
            n,
            h,
            stream,
        )?;
        let grouped_kernel = if self.nvfp4_vecscale && self.moe_w4a4_prequant_t_k64_vecscale.0 != 0
        {
            self.moe_w4a4_prequant_t_k64_vecscale
        } else {
            self.moe_w4a4_prequant_t_k64
        };
        if let Some(work) = compact
            && (std::env::var("ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP").as_deref() == Ok("1")
                || self.glm_c3_grouped(ctx, n)
                || self.glm_c2_grouped(ctx, n)
                || self.independent_grouped(ctx, n)
                || self.glm_c4_grouped(ctx, n))
        {
            let fused_kernel = if self.nvfp4_vecscale
                && self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0 != 0
            {
                self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up
            } else {
                self.moe_w4a4_prequant_t_k64_compact_gate_up
            };
            if fused_kernel.0 != 0 {
                self.m16_gate_up.run(
                    super::gate_up_m16::GateUpCall {
                        rows: n,
                        n: inter,
                        k: h,
                        experts: num_experts,
                        work,
                        native_resources: self.m16_gate_up.enabled()
                            && self.glm_grouped_resources(ctx),
                        vector: self.nvfp4_vecscale
                            && self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0 != 0,
                        original: fused_kernel,
                        outputs: [expert_gate_out, expert_up_out],
                        sorted_tokens: sorted_token_ids,
                        gate_table: gate.packed_ptrs,
                    },
                    ctx,
                    stream,
                    |kernel| {
                        ops::moe_w4a4_grouped_gemm_prequant_compact_gate_up_n128(
                            ctx.gpu,
                            kernel,
                            a_packed,
                            a_scale,
                            gate.packed_ptrs,
                            gate.scale_ptrs,
                            gate.scale2_vals,
                            expert_gate_out,
                            up.packed_ptrs,
                            up.scale_ptrs,
                            up.scale2_vals,
                            expert_up_out,
                            expert_offsets,
                            sorted_token_ids,
                            num_experts,
                            inter,
                            h,
                            work.worklist,
                            work.total_tiles,
                            work.max_tiles,
                            stream,
                        )
                    },
                )?;
                return Ok(false);
            }
        }
        if let Some(grid) = wide
            && grid.gate_up_silu.grid.0 != 0
            && self.nvfp4_fused_silu_quant
            && self.silu_mul_quant_nvfp4_k.0 != 0
            && self.lora.is_none()
            && inter.is_multiple_of(128)
            && h.is_multiple_of(128)
        {
            ops::moe_w4a4_grouped_gemm_prequant_gate_up_silu_k128w(
                ctx.gpu,
                grid.gate_up_silu,
                a_packed,
                a_scale,
                [gate.packed_ptrs, gate.scale_ptrs, gate.scale2_vals],
                [up.packed_ptrs, up.scale_ptrs, up.scale2_vals],
                expert_up_out,
                expert_up_out.offset(grid.rows as usize * inter as usize / 2),
                expert_offsets,
                sorted_token_ids,
                num_experts,
                inter,
                h,
                grid.prefix,
                grid.schedule,
                stream,
            )?;
            return Ok(true);
        }
        for (weight, output) in [(gate, expert_gate_out), (up, expert_up_out)] {
            if let Some(work) = compact {
                let compact_kernel = if self.nvfp4_vecscale
                    && self.moe_w4a4_prequant_t_k64_vecscale_compact.0 != 0
                {
                    self.moe_w4a4_prequant_t_k64_vecscale_compact
                } else {
                    self.moe_w4a4_prequant_t_k64_compact
                };
                ops::moe_w4a4_grouped_gemm_prequant_compact_n128(
                    ctx.gpu,
                    compact_kernel,
                    a_packed,
                    a_scale,
                    weight.packed_ptrs,
                    weight.scale_ptrs,
                    weight.scale2_vals,
                    output,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    inter,
                    h,
                    work.worklist,
                    work.total_tiles,
                    work.max_tiles,
                    stream,
                )?;
            } else {
                self.prequant_grouped(
                    grouped_kernel,
                    [a_packed, a_scale],
                    weight,
                    output,
                    expert_offsets,
                    sorted_token_ids,
                    num_experts,
                    [inter, h],
                    max_m_tiles,
                    wide,
                    ctx,
                    stream,
                )?;
            }
        }
        Ok(false)
    }

    /// The local experts' row-tile grid for the K128W kernel over
    /// `total_expanded` sorted rows, when it is loaded. The prefix lives in the
    /// router FP32 workspace, which GLM's correction-bias router leaves dead
    /// through the routed FFN.
    pub(super) fn mtile_grid(
        &self,
        expert_offsets: DevicePtr,
        local_ptrs: DevicePtr,
        total_expanded: u32,
        num_experts: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<MtileGrid>> {
        if self.moe_w4a4_prequant_t_k128w.grid.0 == 0 || self.moe_mtile_prefix_k.0 == 0 {
            return Ok(None);
        }
        // The persistent schedule's work counter follows the prefix.
        let prefix = ctx.buffers.moe_router_in_f32();
        let next_work = prefix.offset((num_experts as usize + 1) * 4);
        let schedule = k128w_schedule(
            self.k128w_persist_ctas,
            total_expanded,
            num_experts,
            next_work,
        );
        let persistent = matches!(schedule, ops::K128wSchedule::Persistent { .. });
        let words = num_experts as usize + 1 + usize::from(persistent);
        anyhow::ensure!(
            num_experts <= 1024 && ctx.buffers.sizes().moe_router_in_f32 >= words * 4,
            "K128W row-tile prefix exceeds router scratch"
        );
        ops::moe_mtile_prefix(
            ctx.gpu,
            self.moe_mtile_prefix_k,
            expert_offsets,
            local_ptrs,
            prefix,
            num_experts,
            stream,
        )?;
        Ok(Some(MtileGrid {
            prefix,
            schedule,
            rows: total_expanded,
            gate_up_silu: self.moe_w4a4_prequant_gate_up_silu,
            down: self.moe_w4a4_prequant_t_k128w,
        }))
    }

    /// One prequant grouped projection `[rows, k] x [k, n]`: the K128W kernel
    /// over `wide` when it fits, else the K128 kernel (256 threads) when
    /// enabled and K128/N128-aligned, else `fallback` (128 threads) over the
    /// dense `(n/128, max_m_tiles, experts)` grid. All three agree bit for bit.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prequant_grouped(
        &self,
        fallback: KernelHandle,
        [a_packed, a_scale]: [DevicePtr; 2],
        weight: &ExpertPtrTable,
        output: DevicePtr,
        expert_offsets: DevicePtr,
        sorted_token_ids: DevicePtr,
        num_experts: u32,
        [n, k]: [u32; 2],
        max_m_tiles: u32,
        wide: Option<MtileGrid>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(grid) = wide
            && n % 256 == 0
            && k % 128 == 0
        {
            return ops::moe_w4a4_grouped_gemm_prequant_k128w(
                ctx.gpu,
                grid.down,
                a_packed,
                a_scale,
                weight.packed_ptrs,
                weight.scale_ptrs,
                weight.scale2_vals,
                output,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                n,
                k,
                grid.prefix,
                grid.schedule,
                stream,
            );
        }
        let (kernel, threads) =
            if self.moe_w4a4_prequant_t_k128.0 != 0 && n % 128 == 0 && k % 128 == 0 {
                (self.moe_w4a4_prequant_t_k128, 256)
            } else {
                (fallback, 128)
            };
        ops::moe_w4a4_grouped_gemm_prequant_n128(
            ctx.gpu,
            kernel,
            a_packed,
            a_scale,
            weight.packed_ptrs,
            weight.scale_ptrs,
            weight.scale2_vals,
            output,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            n,
            k,
            max_m_tiles,
            threads,
            stream,
        )
    }
}

#[cfg(test)]
#[path = "prequant_fp4_tests.rs"]
mod c3_tests;
