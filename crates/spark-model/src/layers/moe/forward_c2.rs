// SPDX-License-Identifier: AGPL-3.0-only
//! Default-off independent C2 compact FFN; no temporal K2 or resident layout.
use super::*;

pub(super) fn parse_toggle(value: Option<&str>) -> Result<bool> {
    match value {
        None | Some("0") => Ok(false),
        Some("1") => Ok(true),
        _ => anyhow::bail!("ATLAS_GLM_C2_COMPACT_MOE requires 0 or 1"),
    }
}

/// `ATLAS_GLM_C2_COMPACT_MOE=1`: a two-sequence decode runs this MoE once
/// over both rows, so one EP reduce instead of one a row. Both ranks must run
/// the same value (`model::startup_parity`).
pub(crate) fn c2_compact_requested() -> Result<bool> {
    let toggle = match std::env::var("ATLAS_GLM_C2_COMPACT_MOE") {
        Ok(value) => Some(value),
        Err(std::env::VarError::NotPresent) => None,
        Err(error) => return Err(error.into()),
    };
    parse_toggle(toggle.as_deref())
}

impl MoeLayer {
    pub(crate) fn c2_compact_enabled(&self) -> bool {
        self.c2_compact_moe
    }

    pub(crate) fn forward_c2_compact(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        anyhow::ensure!(
            self.glm_c2_grouped(ctx, 2)
                && !self.is_dflash_capture_layer
                && input == ctx.buffers.norm_output()
                && ctx
                    .comm
                    .is_some_and(|c| c.world_size() == 2 && c.rank() == ctx.config.ep_rank),
            "C2 compact requires independent active4 TP2/EP2 GLM native-T resources"
        );
        super::forward_c4::validate_independent_moe_arenas(ctx.config, ctx.buffers.sizes(), 2)?;
        self.forward_prefill(input, 2, ctx, stream)?;
        Ok(ctx.buffers.moe_output())
    }

    pub(super) fn glm_c2_grouped(&self, ctx: &ForwardContext, rows: u32) -> bool {
        self.c2_compact_moe
            && rows == 2
            && ctx.levers.max_decode_seqs == 4
            && ctx.attn_metadata.is_some_and(|m| m.num_seqs == 2)
            && super::prequant_fp4::glm_grouped_shape(ctx.config)
            && self.glm_grouped_resources(ctx)
            && self.use_t_layout_for_prefill()
            && matches!(self.btile_storage, gate_up_repack::Storage::Legacy)
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
            && self.w4a16_gemv_batch2.0 != 0
            && (if self.nvfp4_vecscale {
                self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up.0 != 0
            } else {
                self.moe_w4a4_prequant_t_k64_compact_gate_up.0 != 0
            })
    }

    pub(super) fn c2_shared_expert(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.shared_expert_intermediate_size as u32;
        let gate = ctx.buffers.ssm_deinterleaved();
        let up = ctx.buffers.ssm_qkvz();
        for (weight, out) in [
            (&self.weights.shared_expert.gate_proj, gate),
            (&self.weights.shared_expert.up_proj, up),
        ] {
            ops::w4a16_gemv_batch2(
                ctx.gpu,
                self.w4a16_gemv_batch2,
                input,
                weight,
                out,
                inter,
                h,
                stream,
            )?;
        }
        ops::silu_mul(
            ctx.gpu,
            self.moe_silu_mul,
            gate,
            up,
            gate,
            2 * inter,
            stream,
        )?;
        ops::w4a16_gemv_batch2(
            ctx.gpu,
            self.w4a16_gemv_batch2,
            gate,
            &self.weights.shared_expert.down_proj,
            ctx.buffers.attn_output(),
            h,
            inter,
            stream,
        )
    }
}
