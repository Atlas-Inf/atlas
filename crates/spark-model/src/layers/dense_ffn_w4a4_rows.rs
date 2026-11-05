// SPDX-License-Identifier: AGPL-3.0-only

//! Multi-row dense SiLU FFN on the native W4A4 tensor cores with the
//! checkpoint's calibrated activation scales (ModelOpt NVFP4 static
//! `input_scale`): the format the checkpoint was quantized for, and what the
//! reference engine (SGLang `modelopt_fp4`) runs. Ricursive opts in for every
//! multi-row pass (DFlash verify and prefill); single-token decode keeps W4A16.
//!
//! gate and up share one activation quantization at `max(input_scale)`, as a
//! fused gate_up projection does; down quantizes `silu(gate) * up` at its own.
//!
//! Prefill chunks (M >= [`CUTLASS_MIN_ROWS`]) run CUTLASS's SM120 block-scaled
//! GEMM on the same operands (the kernel family the reference engine uses):
//! weight SFBs packed once at load, the SwiGLU fused into the down input's
//! quantization. Verify widths keep the small-M GEMV.

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

/// Rows from which prefill takes the CUTLASS GEMM. Below it the swizzled SFA
/// atom (rows padded to 128) may not fit the shared `ffn_act_scale` scratch,
/// and the tile GEMM is as fast.
pub const CUTLASS_MIN_ROWS: usize = 64;

/// The W4A4 rows configuration: scales plus the small-M GEMV for verify widths.
#[derive(Clone, Copy, Debug)]
pub(super) struct W4a4Rows {
    pub scales: W4a4InputScales,
    pub gemv_m32: (KernelHandle, KernelHandle),
    /// SwiGLU clamp of the fused down-input quantization (`f32::INFINITY`: none).
    pub swiglu_limit: f32,
    /// Load-time SFBs of gate/up/down for the CUTLASS prefill GEMM.
    pub cutlass: Option<[spark_runtime::cutlass::Nvfp4WeightSfb; 3]>,
}

impl DenseFfnLayer {
    /// Serve every multi-row pass on W4A4 at the given activation scales.
    /// `swiglu_limit` must match the layer's `act_mul` kernel (the CUTLASS
    /// prefill fuses its own copy of the activation); `[hidden, intermediate]`
    /// size the load-time SFBs.
    pub fn enable_w4a4_rows(
        &mut self,
        gpu: &dyn GpuBackend,
        scales: W4a4InputScales,
        swiglu_limit: f32,
        [h, inter]: [usize; 2],
    ) -> Result<()> {
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
        let cutlass = self.pack_cutlass_sfbs(gpu, h, inter)?;
        self.w4a4_rows = Some(W4a4Rows {
            scales,
            gemv_m32,
            swiglu_limit,
            cutlass,
        });
        Ok(())
    }

    /// Pack each projection's native `[N, K/16]` scales into a CUTLASS SFB, or
    /// `None` (with a warning) on a build without CUTLASS.
    fn pack_cutlass_sfbs(
        &self,
        gpu: &dyn GpuBackend,
        h: usize,
        inter: usize,
    ) -> Result<Option<[spark_runtime::cutlass::Nvfp4WeightSfb; 3]>> {
        use spark_runtime::cutlass::{Nvfp4WeightSfb, nvfp4_sf_bytes, pack_weight_sfb};
        let w = &self.weights;
        let stream = gpu.default_stream();
        let mut out = [Nvfp4WeightSfb { packed: 0, sfb: 0 }; 3];
        let projections = [
            (&w.gate_proj, inter, h),
            (&w.up_proj, inter, h),
            (&w.down_proj, h, inter),
        ];
        for (slot, (q, n, k)) in out.iter_mut().zip(projections) {
            let sfb = gpu.alloc(nvfp4_sf_bytes(n, k))?;
            if let Err(e) =
                pack_weight_sfb(q.weight_scale.0, sfb.0, n as u32, k as u32, true, stream)
            {
                gpu.free(sfb)?;
                if gpu.op_cache().once("log:w4a4_rows_no_cutlass") {
                    tracing::warn!(
                        "W4A4 rows: no CUTLASS prefill GEMM ({e:#}); prefill uses the tile GEMM"
                    );
                }
                return Ok(None);
            }
            *slot = Nvfp4WeightSfb {
                packed: q.weight.0,
                sfb: sfb.0,
            };
        }
        gpu.synchronize(stream)?;
        Ok(Some(out))
    }

    pub(super) fn try_w4a4_rows(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(cfg) = self.w4a4_rows.filter(|_| rows >= 2) else {
            return Ok(false);
        };
        if let Some(sfbs) = cfg.cutlass.filter(|_| rows >= CUTLASS_MIN_ROWS) {
            self.w4a4_rows_cutlass(input, rows, &cfg, sfbs, ctx, stream)?;
            return Ok(true);
        }
        let W4a4Rows {
            scales, gemv_m32, ..
        } = cfg;
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

impl DenseFfnLayer {
    /// The prefill-width W4A4 FFN on CUTLASS: quantize at `gate_up`, gate and
    /// up GEMMs, the clamped SwiGLU fused into the down input's quantization at
    /// `down`, the down GEMM into `moe_output`.
    fn w4a4_rows_cutlass(
        &self,
        input: DevicePtr,
        rows: usize,
        cfg: &W4a4Rows,
        [gate_w, up_w, down_w]: [spark_runtime::cutlass::Nvfp4WeightSfb; 3],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        use spark_runtime::cutlass::{Nvfp4Act, nvfp4_gemm_prequant as gemm};
        use spark_runtime::cutlass::{nvfp4_quant_act_scaled, nvfp4_quant_swiglu_clamp};
        let (m, h, inter) = (
            rows as u32,
            ctx.config.hidden_size as u32,
            ctx.config.intermediate_size as u32,
        );
        let a = Nvfp4Act {
            packed: ctx.buffers.ffn_act_a().0,
            sf: ctx.buffers.ffn_act_scale().0,
        };
        let (gate, up) = (
            ctx.buffers.expert_gate_out().0,
            ctx.buffers.expert_up_out().0,
        );
        let (s, w) = (cfg.scales, &self.weights);
        nvfp4_quant_act_scaled(input.0, a, m, h, s.gate_up, stream)?;
        gemm(
            a,
            gate_w,
            s.gate_up * w.gate_proj.weight_scale_2,
            gate,
            m,
            inter,
            h,
            stream,
        )?;
        gemm(
            a,
            up_w,
            s.gate_up * w.up_proj.weight_scale_2,
            up,
            m,
            inter,
            h,
            stream,
        )?;
        nvfp4_quant_swiglu_clamp(gate, up, a, m, inter, s.down, cfg.swiglu_limit, stream)?;
        let out = ctx.buffers.moe_output().0;
        gemm(
            a,
            down_w,
            s.down * w.down_proj.weight_scale_2,
            out,
            m,
            h,
            inter,
            stream,
        )
    }
}
