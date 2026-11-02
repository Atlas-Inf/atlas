// SPDX-License-Identifier: AGPL-3.0-only
//! Real quantizer/worklist producers feeding the promoted SSOT GU readers.
use super::super::grouped::{GroupedMode, InputLayout, ScalePolicy};
use super::*;
use crate::{layer::ForwardContext, layers::ops};
use spark_runtime::gpu::DevicePtr;

impl MoeLayer {
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::moe) fn dispatch_btile_grouped(
        &self,
        input: DevicePtr,
        offsets: DevicePtr,
        sorted: DevicePtr,
        rows: usize,
        compact: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.btile_input_guard(input, rows, ctx, stream)?;
        let Storage::Published(ready) = &self.btile_storage else {
            anyhow::bail!("grouped B-tile requires publication");
        };
        anyhow::ensure!(
            self.nvfp4_prequant_moe,
            "B-tile BF16 grouped reader unsupported"
        );
        let fused = compact
            && (std::env::var("ATLAS_GLM_K5_FUSED_COMPACT_GATE_UP").as_deref() == Ok("1")
                || self.glm_c3_grouped(ctx, rows as u32)
                || self.glm_c4_grouped(ctx, rows as u32));
        let mode = if fused {
            GroupedMode::Fused {
                prefer_small: self.m16_gate_up.enabled(),
            }
        } else if compact {
            GroupedMode::SeparateCompact
        } else {
            GroupedMode::Dense
        };
        let checked = ready.checked(ctx, stream)?;
        let plan = checked.plan_grouped(ctx.buffers, rows, InputLayout::Gathered, mode)?;
        anyhow::ensure!(
            offsets == plan.offsets && sorted == ctx.buffers.gate_logits(),
            "grouped metadata outside actual producer arena"
        );
        // All geometry/capacity/live-range validation precedes any producer.
        if compact {
            ops::moe_build_tile_worklist(
                ctx.gpu,
                self.moe_build_tile_worklist_k,
                offsets,
                ready.tables[0].packed_ptrs,
                plan.work,
                plan.total,
                288,
                16,
                64,
                stream,
            )?;
        }
        ops::quantize_bf16_to_nvfp4(
            ctx.gpu,
            self.quantize_nvfp4_k,
            input,
            plan.a,
            plan.a_scale,
            rows as u32,
            4096,
            stream,
        )?;
        checked.run_grouped(
            plan,
            if self.nvfp4_vecscale {
                ScalePolicy::Vector
            } else {
                ScalePolicy::Scalar
            },
        )
    }
}
