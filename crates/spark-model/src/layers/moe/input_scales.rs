// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_GLM_FP4L_INPUT_SCALES=1`: the routed experts' NVFP4 activations
//! quantized with the checkpoint's calibrated input scales, as the engine
//! NVIDIA's GLM-5.3-Flash-FP4-Local card names (vLLM's b12x A4 prefill:
//! `vllm/model_executor/layers/fused_moe/b12x.py`, flashinfer b12x
//! `prefill_a4.py`), instead of with global scale 1.
//!
//! - gate/up input: one global scale for every expert, `g1 = 1 /
//!   max_e(input_scale)` over the gate and up projections (b12x takes the
//!   `amin` of `1 / w13_input_scale`; vLLM's FlashInfer lanes the same max),
//!   so a token is still quantized once: block scale `e4m3(amax * g1 / 6)`,
//!   codes `e2m1(x * g1 / sf)` (`nvfp4_quantize_group` with `scale2 = 1 /
//!   g1`).
//! - down input (the BF16-rounded SiLU·mul): each expert's own `g2[e] = 1 /
//!   w2_input_scale[e]` (`silu_nvfp4_group_scale`'s `g`).
//! - each GEMM's per-expert alpha takes the activation scale back: the gate
//!   and up tables' `weight_scale_2 * max input_scale`, the down table's
//!   `weight_scale_2 * input_scale[e]` (separate tables; the W4A16 kernels
//!   keep the plain ones).
//!
//! FP4-Local ships `input_scale` per expert in `model-inputscales.safetensors`
//! (the NVFP4 release one per layer). Without the switch, or a checkpoint
//! without them, nothing changes.

use super::*;
use anyhow::{bail, ensure};

/// `ATLAS_GLM_FP4L_INPUT_SCALES` is `1`. Read once.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_FP4L_INPUT_SCALES").as_deref() == Ok("1"))
}

/// The calibrated activation scales of one routed FFN and the GEMM tables
/// that undo them.
pub(crate) struct InputScales {
    /// gate / up / down tables of `gate_ptrs_t` / `up_ptrs_t` / `down_ptrs_t`
    /// with the activation scale folded into `scale2_vals`.
    pub(super) gate: ExpertPtrTable,
    pub(super) up: ExpertPtrTable,
    pub(super) down: ExpertPtrTable,
    /// The gate/up input's `scale2` (`max_e input_scale`), `1 / g1`.
    pub(super) a1: f32,
    /// `[E]` f32 `g2[e] = 1 / down input_scale[e]`.
    pub(super) a2_gscale: DevicePtr,
    /// `silu_mul_quant_nvfp4_by_expert`, for the unfused SiLU quantization.
    pub(super) silu_by_expert: KernelHandle,
}

/// The layer's activation scales from each expert's `[gate, up, down]` input
/// scales: `(a1, a2, present)`, `a1` the largest gate/up scale, `a2[e]` expert
/// `e`'s down scale (1 for an expert that ships none: a non-local expert under
/// expert parallelism, which no row reaches), `present` the experts that ship
/// all three. `None` when no expert ships any; an expert shipping only some
/// is refused.
pub(super) fn fold_scales(experts: &[[Option<f32>; 3]]) -> Result<Option<(f32, Vec<f32>, usize)>> {
    let (mut a1, mut a2) = (0.0f32, Vec::with_capacity(experts.len()));
    let mut present = 0usize;
    for e in experts {
        match *e {
            [Some(g), Some(u), Some(d)] => {
                a1 = a1.max(g).max(u);
                a2.push(d);
                present += 1;
            }
            [None, None, None] => a2.push(1.0),
            _ => bail!("ATLAS_GLM_FP4L_INPUT_SCALES: an expert ships only some input scales"),
        }
    }
    Ok((present > 0).then_some((a1, a2, present)))
}

fn read_f32s(gpu: &dyn GpuBackend, p: DevicePtr, n: usize) -> Result<Vec<f32>> {
    let mut b = vec![0u8; n * 4];
    gpu.copy_d2h(p, &mut b)?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn upload_f32s(gpu: &dyn GpuBackend, v: &[f32]) -> Result<DevicePtr> {
    let p = gpu.alloc(v.len() * 4)?;
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    gpu.copy_h2d(&bytes, p)?;
    Ok(p)
}

/// The positive finite input scale of `w`, if it ships one.
fn input_scale(gpu: &dyn GpuBackend, w: &QuantizedWeight) -> Result<Option<f32>> {
    if w.input_scale.is_null() {
        return Ok(None);
    }
    let s = read_f32s(gpu, w.input_scale, 1)?[0];
    ensure!(
        s.is_finite() && s > 0.0,
        "routed expert input_scale {s} is not positive"
    );
    Ok(Some(s))
}

impl MoeLayer {
    /// Under the switch, the layer's [`InputScales`] from its experts' input
    /// scales and transposed tables. Every local expert must ship gate, up
    /// and down scales; a layer whose experts ship none keeps global scale 1.
    pub(super) fn build_input_scales(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if !requested() {
            return Ok(());
        }
        let (Some(gate_t), Some(up_t), Some(down_t)) =
            (&self.gate_ptrs_t, &self.up_ptrs_t, &self.down_ptrs_t)
        else {
            bail!("ATLAS_GLM_FP4L_INPUT_SCALES: the routed tables are not built");
        };
        let n = self.weights.experts.len();
        let scales = self
            .weights
            .experts
            .iter()
            .map(|e| {
                Ok([
                    input_scale(gpu, &e.gate_proj)?,
                    input_scale(gpu, &e.up_proj)?,
                    input_scale(gpu, &e.down_proj)?,
                ])
            })
            .collect::<Result<Vec<_>>>()?;
        let Some((a1, a2, present)) = fold_scales(&scales)? else {
            if gpu.op_cache().once("moe:input_scales_none") {
                tracing::warn!(
                    "ATLAS_GLM_FP4L_INPUT_SCALES=1 ignored: the routed experts ship no input_scale"
                );
            }
            return Ok(());
        };
        let fold = |t: &ExpertPtrTable, by: &dyn Fn(usize) -> f32| -> Result<ExpertPtrTable> {
            let s2 = read_f32s(gpu, t.scale2_vals, n)?;
            let folded: Vec<f32> = s2.iter().enumerate().map(|(e, &v)| v * by(e)).collect();
            Ok(t.with_scale2(upload_f32s(gpu, &folded)?))
        };
        let gate = fold(gate_t, &|_| a1)?;
        let up = fold(up_t, &|_| a1)?;
        let down = fold(down_t, &|e| a2[e])?;
        let g2: Vec<f32> = a2.iter().map(|&s| 1.0 / s).collect();
        let a2_gscale = upload_f32s(gpu, &g2)?;
        let silu_by_expert =
            super::super::try_kernel(gpu, "moe_silu_mul", "silu_mul_quant_nvfp4_by_expert");
        ensure!(
            silu_by_expert.0 != 0,
            "ATLAS_GLM_FP4L_INPUT_SCALES: the target lacks silu_mul_quant_nvfp4_by_expert"
        );
        if gpu.op_cache().once("moe:input_scales") {
            tracing::info!(
                "ATLAS_GLM_FP4L_INPUT_SCALES: routed NVFP4 activations take the checkpoint's input scales (gate/up max {a1:.3e}, {present} experts)"
            );
        }
        self.input_scales = Some(InputScales {
            gate,
            up,
            down,
            a1,
            a2_gscale,
            silu_by_expert,
        });
        Ok(())
    }

    /// The tables a W4A4 launch over `table` reads: its calibrated twin under
    /// the switch, `table` itself without it.
    pub(super) fn a4_table<'a>(&'a self, table: &'a ExpertPtrTable) -> Result<&'a ExpertPtrTable> {
        let Some(s) = &self.input_scales else {
            return Ok(table);
        };
        for t in [&s.gate, &s.up, &s.down] {
            if t.packed_ptrs == table.packed_ptrs {
                return Ok(t);
            }
        }
        bail!("ATLAS_GLM_FP4L_INPUT_SCALES: a W4A4 launch over an unscaled expert table")
    }

    /// The gate/up input quantizer's `scale2` (1 without the switch).
    pub(super) fn a1_scale2(&self) -> f32 {
        self.input_scales.as_ref().map_or(1.0, |s| s.a1)
    }

    /// The down input quantizer's per-expert global scales (null without).
    pub(super) fn a2_gscale(&self) -> DevicePtr {
        self.input_scales
            .as_ref()
            .map_or(DevicePtr::NULL, |s| s.a2_gscale)
    }
}

#[cfg(test)]
#[path = "input_scales_tests.rs"]
mod tests;
