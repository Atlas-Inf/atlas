// SPDX-License-Identifier: AGPL-3.0-only

//! Prequantized native-FP4 routed down projection and its fused SiLU quantization.

use super::prequant_fp4::MtileGrid;
use super::*;

impl MoeLayer {
    /// The routed down projection into `expert_down_out`: BF16 rows, or with
    /// `ATLAS_GLM_MOE_DOWN_E4M3` on the K128W grid (`wide`) E4M3 rows and
    /// their FP32 scales (lossy), whose pointer it returns.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prequant_fp4_down(
        &self,
        expert_up_out: DevicePtr,
        down: &ExpertPtrTable,
        expert_down_out: DevicePtr,
        expert_offsets: DevicePtr,
        total_expanded: u32,
        h: u32,
        inter: u32,
        num_experts: u32,
        max_m_tiles: u32,
        wide: Option<MtileGrid>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Option<DevicePtr>> {
        // Up output is dead after SiLU·mul; reuse it for packed down A and
        // scales. No allocation or persistent memory is introduced.
        let a_packed = expert_up_out;
        if self.a16_stream() {
            // ATLAS_GLM_FP4L_DECODE_A16: the gate/up twin left BF16 SiLU·mul
            // rows there; the W4A16 down twin reads them and the plain table.
            let grid = wide.filter(|g| {
                g.down.grid.0 != 0 && h.is_multiple_of(256) && inter.is_multiple_of(128)
            });
            anyhow::ensure!(
                grid.is_some(),
                "ATLAS_GLM_FP4L_DECODE_A16: {total_expanded} routed rows left the W4A16 stream twins' grid"
            );
            let grid = grid.expect("checked above");
            ops::moe_w4a4_grouped_gemm_prequant_k128w(
                ctx.gpu,
                grid.down,
                a_packed,
                DevicePtr::NULL,
                down.packed_ptrs,
                down.scale_ptrs,
                down.scale2_vals,
                expert_down_out,
                expert_offsets,
                DevicePtr::NULL,
                num_experts,
                h,
                inter,
                grid.prefix,
                grid.schedule,
                stream,
            )?;
            return Ok(None);
        }
        let a_scale = a_packed.offset(total_expanded as usize * inter as usize / 2);
        if self.a16_stream() {
            // ATLAS_GLM_FP4L_DECODE_A16: the gate/up twin left BF16 SiLU·mul
            // rows there; the W4A16 down twin reads them and the plain table.
            let grid = wide.filter(|g| {
                g.down.grid.0 != 0 && h.is_multiple_of(256) && inter.is_multiple_of(128)
            });
            anyhow::ensure!(
                grid.is_some(),
                "ATLAS_GLM_FP4L_DECODE_A16: {total_expanded} routed rows left the W4A16 stream twins' grid"
            );
            let grid = grid.expect("checked above");
            ops::moe_w4a4_grouped_gemm_prequant_k128w(
                ctx.gpu,
                grid.down,
                a_packed,
                DevicePtr::NULL,
                down.packed_ptrs,
                down.scale_ptrs,
                down.scale2_vals,
                expert_down_out,
                expert_offsets,
                DevicePtr::NULL,
                num_experts,
                h,
                inter,
                grid.prefix,
                grid.schedule,
                stream,
            )?;
            return Ok(None);
        }
        // ATLAS_GLM_FP4L_INPUT_SCALES: the table folding each expert's down
        // input scale back (`input_scales.rs`).
        let down = self.a4_table(down)?;
        if let (Some(e4m3), Some(grid)) = (self.down_e4m3, wide.as_ref())
            // Prefill chunks of over 64 tokens on the K128W row-tile grid
            // only: a decode or verify batch, and the M16 grid (the same
            // MtileGrid, its prefix a worklist), keep their BF16 down.
            && total_expanded > 64 * ctx.config.num_experts_per_tok as u32
            && grid.down.grid.0 == self.moe_w4a4_prequant_t_k128w.grid.0
            && h.is_multiple_of(256)
            && inter.is_multiple_of(128)
            // The routed down LoRA folds BF16 deltas onto these rows.
            && self.lora.is_none()
        {
            // E4M3 bytes fill the first half of the BF16-sized buffer; the
            // scales follow every row's bytes.
            let c_scale = expert_down_out.offset(total_expanded as usize * h as usize);
            ops::moe_w4a4_grouped_gemm_prequant_k128w_into(
                ctx.gpu,
                e4m3.down,
                [a_packed, a_scale],
                [down.packed_ptrs, down.scale_ptrs, down.scale2_vals],
                expert_down_out,
                Some(c_scale),
                [expert_offsets, DevicePtr(0), grid.prefix],
                [num_experts, h, inter],
                grid.schedule,
                stream,
            )?;
            return Ok(Some(c_scale));
        }
        let grouped_kernel = if self.nvfp4_vecscale && self.moe_w4a4_prequant_t_k64_vecscale.0 != 0
        {
            self.moe_w4a4_prequant_t_k64_vecscale
        } else {
            self.moe_w4a4_prequant_t_k64
        };
        self.prequant_grouped(
            grouped_kernel,
            [a_packed, a_scale],
            down,
            expert_down_out,
            expert_offsets,
            DevicePtr(0),
            num_experts,
            [h, inter],
            max_m_tiles,
            wide,
            ctx,
            stream,
        )?;
        Ok(None)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn fused_silu_prequant_fp4_down(
        &self,
        expert_gate_out: DevicePtr,
        expert_up_out: DevicePtr,
        total_expanded: u32,
        inter: u32,
        [expert_offsets, num_experts]: [u64; 2],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Writing compact output directly into expert_up_out would race its
        // still-live BF16 rows across independently scheduled row CTAs. Stage
        // in expert_down_out, then compact-copy into the now-dead up buffer.
        let staged_packed = ctx.buffers.expert_down_out();
        let packed_bytes = total_expanded as usize * inter as usize / 2;
        let scale_bytes = total_expanded as usize * inter as usize / 16;
        let staged_scale = staged_packed.offset(packed_bytes);
        let out_bf16 = if self.lora.is_some() {
            expert_gate_out
        } else {
            DevicePtr::NULL
        };
        // ATLAS_GLM_FP4L_INPUT_SCALES: each expert's rows at its own down
        // input scale (`input_scales.rs`).
        if let Some(s) = &self.input_scales {
            ops::silu_mul_quant_nvfp4_by_expert(
                ctx.gpu,
                s.silu_by_expert,
                [expert_gate_out, expert_up_out],
                [staged_packed, staged_scale, out_bf16],
                [total_expanded, inter],
                [DevicePtr(expert_offsets), s.a2_gscale],
                num_experts as u32,
                stream,
            )?;
        } else {
            ops::silu_mul_quant_nvfp4(
                ctx.gpu,
                self.silu_mul_quant_nvfp4_k,
                expert_gate_out,
                expert_up_out,
                staged_packed,
                staged_scale,
                out_bf16,
                total_expanded,
                inter,
                stream,
            )?;
        }
        ctx.gpu.copy_d2d_async(
            staged_packed,
            expert_up_out,
            packed_bytes + scale_bytes,
            stream,
        )
    }
}
