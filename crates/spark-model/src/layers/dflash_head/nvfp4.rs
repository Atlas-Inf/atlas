// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter NVFP4: load-time quant + small-M `w4a16_gemv_batch{4,8}` dispatch.
//!
//! The drafter runs its per-step projections at M=γ rows: `w4a16_gemv_batch4`
//! covers γ ≤ 4 and `w4a16_gemv_batch8` covers γ in 5..8. Above that the
//! NVFP4 copies are never read, so the install skips quantizing them.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BlockDiffusionDraftHead, DflashQuantization};
use crate::layers::ops;
use crate::weight_map::{DenseWeight, Fp8DenseWeight, QuantizedWeight, quantize_to_nvfp4};

/// Which batched-GEMV tier the drafter NVFP4 projections dispatch through at a
/// given γ. Pure decision so `try_install_nvfp4` quantizes only what decode
/// will actually read, and so the tier table is testable without a GPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DrafterNvfp4Tier {
    Batch4,
    Batch8,
}

impl DrafterNvfp4Tier {
    fn kernel_name(self) -> &'static str {
        match self {
            Self::Batch4 => "w4a16_gemv_batch4",
            Self::Batch8 => "w4a16_gemv_batch8",
        }
    }
}

/// Pure γ→tier pick; `None` above the batched GEMV tiers' reach (γ > 8).
pub(super) fn drafter_nvfp4_tier(gamma: u32) -> Option<DrafterNvfp4Tier> {
    match gamma {
        1..=4 => Some(DrafterNvfp4Tier::Batch4),
        5..=8 => Some(DrafterNvfp4Tier::Batch8),
        _ => None,
    }
}

impl BlockDiffusionDraftHead {
    pub(super) fn try_install_nvfp4(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if std::env::var("ATLAS_NO_DFLASH_DRAFTER_NVFP4").is_ok() {
            return Ok(());
        }
        let Some(tier) = drafter_nvfp4_tier(self.gamma as u32) else {
            tracing::warn!(
                "DFlash NVFP4: γ={} exceeds the batched GEMV tiers —                  skipping drafter quantization (weights would never be read)",
                self.gamma
            );
            return Ok(());
        };
        let nvfp4_kernel = match tier {
            DrafterNvfp4Tier::Batch4 => self.kernels.w4a16_gemv_batch4,
            DrafterNvfp4Tier::Batch8 => self.kernels.w4a16_gemv_batch8,
        };
        if nvfp4_kernel.0 == 0 {
            tracing::warn!(
                "DFlash NVFP4: {} missing — skipping drafter quantization",
                tier.kernel_name()
            );
            return Ok(());
        }
        let absmax = match gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax") {
            Ok(k) => k,
            Err(e) => {
                tracing::warn!("NVFP4 absmax kernel missing: {e}");
                return Ok(());
            }
        };
        let quant = match gpu.kernel("quantize_nvfp4", "quantize_bf16_to_nvfp4") {
            Ok(k) => k,
            Err(e) => {
                tracing::warn!("NVFP4 quant kernel missing: {e}");
                return Ok(());
            }
        };
        let stream = 0u64;
        let h = self.hidden_size;
        let q_dim = self.num_q_heads * self.head_dim;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let inter = self.intermediate_size;
        tracing::info!(
            "DFlash NVFP4: quantizing {} layers × 7 GEMMs for {}",
            self.layers.len(),
            tier.kernel_name()
        );
        for layer in &mut self.layers {
            layer.q_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.q_proj,
                q_dim,
                h,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.k_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.k_proj,
                kv_dim,
                h,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.v_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.v_proj,
                kv_dim,
                h,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.o_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.o_proj,
                h,
                q_dim,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.gate_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.gate_proj,
                inter,
                h,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.up_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.up_proj,
                inter,
                h,
                gpu,
                absmax,
                quant,
                stream,
            )?);
            layer.down_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.down_proj,
                h,
                inter,
                gpu,
                absmax,
                quant,
                stream,
            )?);
        }
        self.quant = DflashQuantization::Nvfp4Weights;
        tracing::info!("DFlash NVFP4: ready (quant = Nvfp4Weights)");
        Ok(())
    }

    pub(super) fn drafter_gemm(
        &self,
        gpu: &dyn GpuBackend,
        w_bf16: &DenseWeight,
        w_fp8: &Option<Fp8DenseWeight>,
        w_nvfp4: &Option<QuantizedWeight>,
        src: DevicePtr,
        dst: DevicePtr,
        n_out: u32,
        k_in: u32,
        stream: u64,
    ) -> Result<()> {
        let g = self.gamma as u32;
        let nvfp4_kernel = drafter_nvfp4_tier(g).map(|tier| match tier {
            DrafterNvfp4Tier::Batch4 => self.kernels.w4a16_gemv_batch4,
            DrafterNvfp4Tier::Batch8 => self.kernels.w4a16_gemv_batch8,
        });
        if matches!(self.quant, DflashQuantization::Nvfp4Weights)
            && let Some(w) = w_nvfp4
            && let Some(k) = nvfp4_kernel
            && k.0 != 0
        {
            return ops::w4a16_gemv_batchm(
                gpu,
                k,
                src,
                w,
                dst,
                g,
                n_out,
                k_in,
                stream,
            );
        }
        if matches!(self.quant, DflashQuantization::Fp8Weights)
            && let Some(fp8) = w_fp8
        {
            return ops::fp8_gemm_n128_row_scaled(
                gpu,
                self.kernels.fp8_gemm_n128_row_scaled,
                src,
                fp8,
                dst,
                g,
                n_out,
                k_in,
                stream,
            );
        }
        self.drafter_dense_gemm(gpu, src, w_bf16, dst, g, n_out, k_in, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_nvfp4_tier_pick() {
        assert_eq!(drafter_nvfp4_tier(1), Some(DrafterNvfp4Tier::Batch4));
        assert_eq!(drafter_nvfp4_tier(4), Some(DrafterNvfp4Tier::Batch4));
        assert_eq!(drafter_nvfp4_tier(5), Some(DrafterNvfp4Tier::Batch8));
        assert_eq!(drafter_nvfp4_tier(8), Some(DrafterNvfp4Tier::Batch8));
        assert_eq!(drafter_nvfp4_tier(9), None);
        assert_eq!(drafter_nvfp4_tier(0), None);
    }
}
