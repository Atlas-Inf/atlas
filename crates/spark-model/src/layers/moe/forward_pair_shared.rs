// SPDX-License-Identifier: AGPL-3.0-only
//! Explicit shared width within the joint routed FFN; legacy K5 is separate.
use super::*;

/// `ATLAS_GLM_SHARED_TP_SPLIT=1`. Read once. Both ranks must run the same
/// value (`model::startup_parity`): a rank splitting alone adds its half to
/// the peer's whole shared output.
pub(crate) fn shared_tp_split_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_SHARED_TP_SPLIT").as_deref() == Ok("1"))
}

/// `ATLAS_MOE_SHARED_REDUCE_OVERLAP=1`: run the shared expert on the prefill
/// stream while the EP all-reduce is in flight. A chunk that overlaps does
/// not take the TP split, so this must match across the ranks too.
pub(crate) fn shared_reduce_overlap_requested() -> bool {
    std::env::var("ATLAS_MOE_SHARED_REDUCE_OVERLAP").as_deref() == Ok("1")
}

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    /// Shared expert for an owner-batched verify of `rows` (9..=32) rows:
    /// the native NVFP4 projections through one tensor-core weight pass each
    /// (`kernel` from `w4a16_gemv_tiers::tc_kernel`), the arithmetic the
    /// <=8-row verify already uses.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_shared_tc(
        &self,
        kernel: KernelHandle,
        input: DevicePtr,
        shared_gate_out: DevicePtr,
        shared_up_out: DevicePtr,
        shared_down_out: DevicePtr,
        rows: u32,
        h: u32,
        shared_inter: u32,
        ctx: &ForwardContext,
        aux: u64,
    ) -> Result<()> {
        let shared = &self.weights.shared_expert;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kernel,
            input,
            &shared.gate_proj,
            shared_gate_out,
            rows,
            shared_inter,
            h,
            aux,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kernel,
            input,
            &shared.up_proj,
            shared_up_out,
            rows,
            shared_inter,
            h,
            aux,
        )?;
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            rows * shared_inter,
            aux,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            kernel,
            shared_gate_out,
            &shared.down_proj,
            shared_down_out,
            rows,
            h,
            shared_inter,
            aux,
        )
    }

    /// Whether the shared expert can run TP-split for `rows` replicated rows
    /// (`ATLAS_GLM_SHARED_TP_SPLIT=1`, EP2, NVFP4 shared weights with scalar
    /// scale2, strided tensor-core tiers for `rows`).
    pub(super) fn shared_split_ready(&self, ctx: &ForwardContext, rows: u32) -> bool {
        let shared = &self.weights.shared_expert;
        let inter = ctx.config.shared_expert_intermediate_size;
        shared_tp_split_requested()
            && ctx.config.ep_world_size == 2
            && ctx.comm.is_some_and(|c| c.world_size() == 2)
            && inter.is_multiple_of(32)
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && [&shared.gate_proj, &shared.up_proj, &shared.down_proj]
                .iter()
                .all(|w| !w.is_null() && !w.has_per_row_scale2())
            && crate::layers::w4a16_gemv_tiers::tc_kernel(rows).0 != 0
            && crate::layers::w4a16_gemv_tiers::tc_ld_kernel(rows).0 != 0
    }

    /// This rank's half of the shared expert: intermediate columns
    /// `[rank * inter/2, +inter/2)` — gate/up rows (a pointer offset) and the
    /// matching K-slice of down (strided tier). `attn_output` then holds this
    /// rank's partial of the shared output; the EP all-reduce sums the two.
    pub(super) fn run_shared_split(
        &self,
        input: DevicePtr,
        rows: u32,
        h: u32,
        inter: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let shared = &self.weights.shared_expert;
        let half = inter / 2;
        let col0 = (ctx.config.ep_rank as u32 * half) as usize;
        let gate_up_rows = |w: &QuantizedWeight| QuantizedWeight {
            weight: w.weight.offset(col0 * h as usize / 2),
            weight_scale: w.weight_scale.offset(col0 * h as usize / 16),
            ..*w
        };
        let down_cols = QuantizedWeight {
            weight: shared.down_proj.weight.offset(col0 / 2),
            weight_scale: shared.down_proj.weight_scale.offset(col0 / 16),
            ..shared.down_proj
        };
        let (gate_out, up_out, down_out) = (
            ctx.buffers.ssm_deinterleaved(),
            ctx.buffers.ssm_qkvz(),
            ctx.buffers.attn_output(),
        );
        let tc = crate::layers::w4a16_gemv_tiers::tc_kernel(rows);
        let (gate, up) = (
            gate_up_rows(&shared.gate_proj),
            gate_up_rows(&shared.up_proj),
        );
        let gate_up = [(&gate, gate_out), (&up, up_out)];
        // ATLAS_GLM_DECODE_GEMV_BATCH: gate and up in one launch of the tc
        // body, both touched during the PDL wait (`ops::gemv_touch`).
        match ops::w4a16_pair_touch(ctx.gpu, tc, input, gate_up, rows, half, h, stream) {
            Some(done) => done?,
            None => {
                for (weight, out) in gate_up {
                    ops::w4a16_gemv_batchm(ctx.gpu, tc, input, weight, out, rows, half, h, stream)?;
                }
            }
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            gate_out,
            up_out,
            gate_out,
            rows * half,
            stream,
        )?;
        let (ld_half, ld_groups) = (inter / 2, inter / 16);
        let touch = crate::layers::w4a16_gemv_tiers::tc_rows(tc)
            .and_then(ops::w4a16_tc_twin)
            .and_then(ops::gemv_touch);
        if let Some(touch) = touch {
            return touch.w4a16_tc(
                ctx.gpu, gate_out, &down_cols, down_out, rows, h, half, ld_half, ld_groups, stream,
            );
        }
        ops::w4a16_gemv_tc_ld(
            ctx.gpu,
            crate::layers::w4a16_gemv_tiers::tc_ld_kernel(rows),
            gate_out,
            &down_cols,
            down_out,
            rows,
            h,
            half,
            ld_half,
            ld_groups,
            stream,
        )
    }

    pub(super) fn run_exact_k5_shared(
        &self,
        input: DevicePtr,
        shared_gate_out: DevicePtr,
        shared_up_out: DevicePtr,
        shared_down_out: DevicePtr,
        h: u32,
        shared_inter: u32,
        ctx: &ForwardContext,
        aux: u64,
    ) -> Result<()> {
        let batch5 = self.w4a16_batchm.kernel(5);
        let fused_gate_up = self.w4a16_batch5_dual_k.0 != 0
            && std::env::var("ATLAS_GLM_K5_FUSED_SHARED_GATE_UP").as_deref() == Ok("1");
        if fused_gate_up {
            ops::w4a16_gemv_batch5_dual(
                ctx.gpu,
                self.w4a16_batch5_dual_k,
                input,
                &self.weights.shared_expert.gate_proj,
                &self.weights.shared_expert.up_proj,
                shared_gate_out,
                shared_up_out,
                5,
                shared_inter,
                h,
                aux,
            )?;
        } else {
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                batch5,
                input,
                &self.weights.shared_expert.gate_proj,
                shared_gate_out,
                5,
                shared_inter,
                h,
                aux,
            )?;
            ops::w4a16_gemv_batchm(
                ctx.gpu,
                batch5,
                input,
                &self.weights.shared_expert.up_proj,
                shared_up_out,
                5,
                shared_inter,
                h,
                aux,
            )?;
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_act_mul,
            shared_gate_out,
            shared_up_out,
            shared_gate_out,
            5 * shared_inter,
            aux,
        )?;
        ops::w4a16_gemv_batchm(
            ctx.gpu,
            batch5,
            shared_gate_out,
            &self.weights.shared_expert.down_proj,
            shared_down_out,
            5,
            h,
            shared_inter,
            aux,
        )?;
        Ok(())
    }
}
