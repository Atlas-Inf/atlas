// SPDX-License-Identifier: AGPL-3.0-only
//! Exact independent widths2..8 over the existing native-T/M64 pipeline.
use super::*;
pub(crate) fn validate_independent_environment() -> Result<()> {
    let (fp8, verify) = super::shared_fp8_cache::flags()?;
    anyhow::ensure!(
        !fp8 && !verify,
        "independent decode excludes target shared FP8 cache"
    );
    anyhow::ensure!(
        !super::forward_prefill_routed::grouped_cutlass_gate_up_enabled()
            && !super::forward_prefill_routed::grouped_cutlass_down_enabled()
            && !super::forward_prefill_routed::env_flag("ATLAS_NVFP4_MMQ_MOE"),
        "independent decode excludes CUTLASS/MMQ MoE"
    );
    Ok(())
}
impl MoeLayer {
    pub(super) fn validate_independent_handles(&self) -> Result<()> {
        let handles = std::array::from_fn(|i| match i + 2 {
            2 => self.w4a16_gemv_batch2.0,
            3 => self.w4a16_gemv_batch3.0,
            n => self.w4a16_batchm.kernel(n as u32).0,
        });
        crate::model::glm_independent::validate_projection_handles(handles, self.dense_gemv.0)?;
        let fused = if self.nvfp4_vecscale {
            self.moe_w4a4_prequant_t_k64_vecscale_compact_gate_up
        } else {
            self.moe_w4a4_prequant_t_k64_compact_gate_up
        };
        anyhow::ensure!(
            fused.0 != 0
                && self.moe_build_tile_worklist_k.0 != 0
                && self.moe_unpermute_reduce_ep.0 != 0
                && self.moe_w4a4_prequant_t_k64_compact.0 != 0
                && self.moe_w4a4_prequant_t_k64.0 != 0
                && self.quantize_nvfp4_k.0 != 0
                && self.silu_mul_quant_nvfp4_k.0 != 0,
            "independent MoE needs every selected native grouped handle"
        );
        Ok(())
    }
    pub(super) fn independent_grouped(&self, ctx: &ForwardContext, rows: u32) -> bool {
        // The public entry performs fallible validation before entering prefill.
        crate::model::glm_independent::ffn_rows_selected(ctx, rows as usize).unwrap_or(false)
            && super::prequant_fp4::glm_grouped_shape(ctx.config)
            && self.glm_grouped_resources(ctx)
            && self.use_t_layout_for_prefill()
            && matches!(self.btile_storage, gate_up_repack::Storage::Legacy)
            && self.shared_experts_scale_kind == crate::weight_map::WeightQuantFormat::Nvfp4
    }
    pub(crate) fn forward_independent(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        // A DFlash verify block on experts without the transposed layout
        // (e.g. ATLAS_MOE_GROUPED_CUTLASS) takes the grouped prefill MoE.
        if !self.independent_grouped(ctx, rows as u32)
            && !crate::model::glm_independent::selected(ctx, rows)?
            && crate::model::glm_independent::ffn_rows_selected(ctx, rows)?
        {
            self.forward_prefill(input, rows, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
        }
        anyhow::ensure!(
            crate::model::glm_independent::ffn_rows_selected(ctx, rows)?
                && self.independent_grouped(ctx, rows as u32)
                && !self.is_dflash_capture_layer
                && input == ctx.buffers.norm_output()
                && ctx
                    .comm
                    .is_some_and(|c| c.world_size() == 2 && c.rank() == ctx.config.ep_rank),
            "independent MoE requires native-T exact live rows and local EP2 context"
        );
        self.validate_independent_handles()?;
        super::forward_c4::validate_independent_moe_arenas(ctx.config, ctx.buffers.sizes(), rows)?;
        self.forward_prefill(input, rows, ctx, stream)?;
        Ok(ctx.buffers.moe_output())
    }
    pub(super) fn independent_shared_expert(
        &self,
        input: DevicePtr,
        rows: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if rows == 2 {
            return self.c2_shared_expert(input, ctx, stream);
        }
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.shared_expert_intermediate_size as u32;
        let gate = ctx.buffers.ssm_deinterleaved();
        let up = ctx.buffers.ssm_qkvz();
        let down = ctx.buffers.attn_output();
        if rows == 3 {
            return self.c3_shared_expert(input, gate, up, down, h, inter, 3, ctx, stream);
        }
        self.independent_shared_batchm(input, gate, up, down, rows, h, inter, ctx, stream)
    }
}
