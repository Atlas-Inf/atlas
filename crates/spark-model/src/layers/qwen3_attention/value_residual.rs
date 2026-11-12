// SPDX-License-Identifier: AGPL-3.0-only

//! Value residuals (Ricursive Ripple): every layer's V projection reads
//! `lerp(x0, normed, sigmoid(lambda))` instead of `normed`, where `x0` is the
//! scaled embedding — the residual stream entering layer 0.
//!
//! Reference (`RicursiveAttention.forward`):
//!
//! ```text
//! blend = sigmoid(value_res_lambda[0]).to(bf16)
//! v = v_proj(torch.lerp(x0, normed, blend))
//! ```
//!
//! Layer 0 snapshots its input rows into the shared `x0` buffer before the
//! residual stream is updated; later layers of the same forward read them.
//! Every path that projects V (decode, multi-row verify, prefill) routes its
//! V input through [`Qwen3AttentionLayer::value_residual_input`].

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::Qwen3AttentionLayer;

/// Per-layer value-residual state. `x0` and `blended` are shared by every
/// layer of the model (one allocation each, `[max_rows, hidden]` BF16).
#[derive(Clone, Copy, Debug)]
pub struct ValueResidual {
    pub x0: DevicePtr,
    pub blended: DevicePtr,
    pub max_rows: usize,
    /// `bf16(sigmoid(bf16 lambda))`, as the reference rounds it.
    pub blend: f32,
    /// Layer 0 writes `x0`; every other layer only reads it.
    pub capture: bool,
    pub lerp_k: KernelHandle,
    /// `bf16_gemv_m32`: the wide-verify BF16 q|gate and o projections.
    pub wide_gemv_k: KernelHandle,
    /// `ricursive_verify_attn`: one sequence's verify rows against the cache.
    pub verify_attn_k: KernelHandle,
}

/// `bf16(sigmoid(lambda))` with the sigmoid evaluated in f32 on the BF16
/// parameter value, matching torch's BF16 `sigmoid`.
pub fn value_residual_blend(lambda_bf16: f32) -> f32 {
    let s = 1.0f32 / (1.0f32 + (-lambda_bf16).exp());
    half::bf16::from_f32(s).to_f32()
}

impl Qwen3AttentionLayer {
    /// The prefill-style verify attention (`attn_rows.rs`), when the target
    /// ships it (Ricursive). 0 elsewhere.
    pub(crate) fn verify_rows_attn_k(&self) -> Option<KernelHandle> {
        self.value_residual
            .map(|vr| vr.verify_attn_k)
            .filter(|k| k.0 != 0)
    }

    /// Wide (n > 8) multi-row BF16 projections take the cuBLASLt GEMM rather
    /// than the scalar `dense_gemm` tile kernel (~89 GB/s). Only Ricursive
    /// opts in, so other models' verify numerics are unchanged.
    pub(crate) fn wide_bf16_gemm(&self) -> bool {
        self.value_residual.is_some()
    }

    /// Multi-row prefill projections on a native BF16 weight take the BLAS GEMM
    /// (`bf16_gemm`: cuBLASLt, hipBLASLt on HIP) instead of
    /// `dense_gemm_bf16_pipelined` when `ATLAS_CUBLAS_GEMM` is set, and always
    /// for Ricursive on gfx1151, where the pipelined kernel runs at 8-18 TFLOPS
    /// against hipBLASLt's 20-34 at Ripple's q+gate / k|v / o shapes. Unlike the
    /// env flag this reserves no lazy BF16 copies (the weights are already BF16).
    pub(crate) fn prefill_bf16_blas(&self, cublas_gemm: bool, rows: u32) -> bool {
        rows > 1 && (cublas_gemm || (cfg!(atlas_hip) && self.wide_bf16_gemm()))
    }

    /// The small-M BF16 GEMV for `rows` verify rows, or `None` (wider, or a
    /// model without it): callers fall back to the cuBLASLt GEMM.
    pub(crate) fn wide_bf16_gemv(&self, rows: usize) -> Option<KernelHandle> {
        self.value_residual
            .map(|vr| vr.wide_gemv_k)
            .filter(|k| k.0 != 0 && rows as u32 <= crate::layers::ops::BF16_GEMV_MAX_M)
    }

    /// Every V path blends through `Self::value_residual_input`; the
    /// quantized Q/K/V arms project V from activations quantized from the
    /// un-blended rows, so value residuals require dense BF16 projections.
    pub fn set_value_residual(&mut self, vr: ValueResidual) -> Result<()> {
        ensure!(
            self.qkv_is_dense_bf16() && self.lora.is_none(),
            "value residuals need dense BF16 Q/K/V projections without LoRA"
        );
        self.value_residual = Some(vr);
        Ok(())
    }

    /// Layer 0 only: copy the `rows` input rows (the scaled embeddings) into `x0`.
    pub(crate) fn value_residual_capture(
        &self,
        gpu: &dyn GpuBackend,
        hidden: DevicePtr,
        rows: usize,
        h: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(vr) = self.value_residual.filter(|vr| vr.capture) else {
            return Ok(());
        };
        ensure!(
            rows <= vr.max_rows,
            "value residual: {rows} rows exceed the {}-row x0 buffer",
            vr.max_rows
        );
        gpu.copy_d2d_async(hidden, vr.x0, rows * h * 2, stream)
    }

    /// The V projection input for `rows` contiguous normed rows: `normed`
    /// itself without value residuals, else the blended rows.
    pub(crate) fn value_residual_input(
        &self,
        gpu: &dyn GpuBackend,
        normed: DevicePtr,
        rows: usize,
        h: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let Some(vr) = self.value_residual else {
            return Ok(normed);
        };
        ensure!(
            rows <= vr.max_rows,
            "value residual: {rows} rows exceed the {}-row blend buffer",
            vr.max_rows
        );
        let n = (rows * h) as u32;
        KernelLaunch::new(gpu, vr.lerp_k)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(vr.x0)
            .arg_ptr(normed)
            .arg_ptr(vr.blended)
            .arg_u32(n)
            .arg_f32(vr.blend)
            .launch(stream)?;
        Ok(vr.blended)
    }
}

#[cfg(test)]
mod tests {
    use super::value_residual_blend;

    #[test]
    fn blend_is_bf16_rounded_sigmoid() {
        // Layer 0 of Ripple-65B Preview ships lambda = 2.140625 (exact in BF16).
        let b = value_residual_blend(2.140625);
        assert_eq!(b, half::bf16::from_f32(b).to_f32());
        assert!((b - 0.894_531_25).abs() < 1e-6, "{b}");
        assert_eq!(value_residual_blend(0.0), 0.5);
    }
}
