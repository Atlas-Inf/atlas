// SPDX-License-Identifier: AGPL-3.0-only

//! Prequantized native-FP4 routed down projection and its fused SiLU quantization.

use super::prequant_fp4::MtileGrid;
use super::*;

impl MoeLayer {
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
    ) -> Result<()> {
        // Up output is dead after SiLU·mul; reuse it for packed down A and
        // scales. No allocation or persistent memory is introduced.
        let a_packed = expert_up_out;
        let a_scale = a_packed.offset(total_expanded as usize * inter as usize / 2);
        // ATLAS_GLM_FP4L_INPUT_SCALES: the table folding each expert's down
        // input scale back (`input_scales.rs`).
        let down = self.a4_table(down)?;
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
        )
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
