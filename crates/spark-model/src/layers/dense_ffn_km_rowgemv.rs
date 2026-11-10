// SPDX-License-Identifier: AGPL-3.0-only

//! K=m DFlash verify rows (9..=16) on the strix-hip row-GEMV
//! (`ATLAS_FFN_ROWGEMV=1`, `ops::RowGemvKernels`) instead of the scalar
//! `w4a16_gemv_batch16` tier. dp4a serves 2..=8 rows, so 9..=16 is where the
//! 27B's γ>=8 verify lands; kbench 42775662809 (27B M=9): gate/up 576 -> 278
//! us, down 587 -> 272 us. The row-GEMV reads the transposed twins with exact
//! f32 weights, so its rows differ from batch16's in the last bits (numerics:
//! the switch needs its ST-995 before it can default on).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::DenseFfnLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl DenseFfnLayer {
    /// `Ok(false)`, nothing launched, unless the lever is on, `m` is 9..=16,
    /// the plain SiLU arm applies and every projection has a live twin the
    /// row-GEMV fits.
    pub(super) fn forward_km_rowgemv(
        &self,
        input: DevicePtr,
        m: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.intermediate_size as u32;
        let w = &self.weights;
        let (Some(gt), Some(ut), Some(dt)) = (&w.gate_proj_t, &w.up_proj_t, &w.down_proj_t) else {
            return Ok(false);
        };
        let rg = &self.w4a16_rowgemv;
        if !ctx.levers.ffn_rowgemv
            || !(9..=16).contains(&m)
            || self.act_mul.0 == 0
            || [gt, ut, dt].iter().any(|t| t.weight.is_null())
            || !rg.fits(m, inter, h)
            || !rg.fits(m, h, inter)
        {
            return Ok(false);
        }
        if ctx.stats.once("log:decode_ffn_km_rowgemv") {
            tracing::info!(rows = m, "Dense FFN K=m verify on the row-GEMV");
        }
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        rg.launch(ctx.gpu, input, gt, gate_out, m, inter, h, stream)?;
        rg.launch(ctx.gpu, input, ut, up_out, m, inter, h, stream)?;
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate_out,
            up_out,
            gate_out,
            m * inter,
            stream,
        )?;
        let output = ctx.buffers.moe_output();
        rg.launch(ctx.gpu, gate_out, dt, output, m, h, inter, stream)?;
        Ok(true)
    }
}
