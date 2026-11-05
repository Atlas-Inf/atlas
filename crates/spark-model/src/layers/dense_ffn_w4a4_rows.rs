// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-row dense SiLU FFN on the native W4A4 tensor cores with the
//! checkpoint's calibrated activation scales (ModelOpt NVFP4 static
//! `input_scale`): the format the checkpoint was quantized for, and what the
//! reference engine (SGLang `modelopt_fp4`) runs. Ricursive opts in for every
//! multi-row pass (DFlash verify and prefill); single-token decode keeps W4A16.
//!
//! gate and up share one activation quantization at `max(input_scale)`, as a
//! fused gate_up projection does; down quantizes `silu(gate) * up` at its own.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use super::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Static per-tensor activation scales of one dense FFN (checkpoint values).
#[derive(Clone, Copy, Debug)]
pub struct W4a4InputScales {
    pub gate_up: f32,
    pub down: f32,
}

/// The W4A4 rows configuration: scales plus the small-M GEMV for verify widths.
#[derive(Clone, Copy, Debug)]
pub(super) struct W4a4Rows {
    pub scales: W4a4InputScales,
    pub gemv_m32: (KernelHandle, KernelHandle),
}

impl DenseFfnLayer {
    /// Serve every multi-row pass on W4A4 at the given activation scales.
    pub fn enable_w4a4_rows(&mut self, gpu: &dyn GpuBackend, scales: W4a4InputScales) -> Result<()> {
        ensure!(
            self.w4a4_gemm_mfast_k.0 != 0 && self.quantize_nvfp4_k.0 != 0,
            "W4A4 rows need w4a4_gemm_mfast and quantize_bf16_to_nvfp4 in this kernel target"
        );
        ensure!(
            scales.gate_up > 0.0 && scales.down > 0.0,
            "W4A4 rows need positive input scales, got {scales:?}"
        );
        ensure!(
            self.activation == FfnActivation::SiLU && self.lora.is_none(),
            "W4A4 rows cover the SiLU FFN without LoRA"
        );
        let gemv_m32 = (
            gpu.kernel("w4a4_gemv_m32", "w4a4_gemv_m32_bn32")?,
            gpu.kernel("w4a4_gemv_m32", "w4a4_gemv_m32_bn16")?,
        );
        self.w4a4_rows = Some(W4a4Rows { scales, gemv_m32 });
        Ok(())
    }

    pub(super) fn try_w4a4_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(W4a4Rows { scales, gemv_m32 }) = self.w4a4_rows.filter(|_| rows >= 2) else {
            return Ok(false);
        };
        let gpu: &dyn GpuBackend = ctx.gpu;
        let w = &self.weights;
        let (m, h, inter) = (
            rows as u32,
            ctx.config.hidden_size as u32,
            ctx.config.intermediate_size as u32,
        );
        let (a, a_sf) = (ctx.buffers.ffn_act_a(), ctx.buffers.ffn_act_scale());
        let (gate, up) = (ctx.buffers.expert_gate_out(), ctx.buffers.expert_up_out());
        let q = self.quantize_nvfp4_k;
        let mm = self.w4a4_gemm_mfast_k;
        // Verify widths take the small-M GEMV; prefill chunks the tile GEMM.
        let gemm = |s: f32, wt, out, n, k| {
            if m <= ops::W4A4_GEMV_MAX_M {
                ops::w4a4_gemv_m32(gpu, gemv_m32, a, a_sf, s, wt, out, m, n, k, stream)
            } else {
                ops::w4a4_gemm_mfast_scaled(gpu, mm, a, a_sf, s, wt, out, m, n, k, stream)
            }
        };
        ops::quantize_bf16_to_nvfp4_scaled(gpu, q, input, a, a_sf, scales.gate_up, m, h, stream)?;
        gemm(scales.gate_up, &w.gate_proj, gate, inter, h)?;
        gemm(scales.gate_up, &w.up_proj, up, inter, h)?;
        ops::silu_mul(gpu, self.act_mul, gate, up, gate, m * inter, stream)?;
        ops::quantize_bf16_to_nvfp4_scaled(gpu, q, gate, a, a_sf, scales.down, m, inter, stream)?;
        let out = ctx.buffers.moe_output();
        gemm(scales.down, &w.down_proj, out, h, inter)?;
        Ok(true)
    }
}
