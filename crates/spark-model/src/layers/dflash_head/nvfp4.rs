// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter NVFP4: load-time quant and scalar or tensor-core dispatch.
//!
//! The drafter runs its per-step projections at M=γ rows: `w4a16_gemv_batch4`
//! covers γ ≤ 4, `w4a16_gemv_batch8` γ in 5..8 and `w4a16_gemv_batch16` γ in
//! 9..16 (DSpine's b16 blocks). The opt-in
//! tensor-core route covers wider blocks in pieces of up to 32 rows.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BlockDiffusionDraftHead, DflashQuantization};
use crate::layers::ops;
use crate::weight_loader::dflash_preshrink::take_twin;
use crate::weight_map::{
    DenseWeight, Fp8DenseWeight, Mxfp8Weight, QuantizedWeight, quantize_to_nvfp4,
};

/// Which batched-GEMV tier the drafter NVFP4 projections dispatch through at a
/// given γ. Pure decision so `try_install_nvfp4` quantizes only what decode
/// will actually read, and so the tier table is testable without a GPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum DrafterNvfp4Tier {
    Batch4,
    Batch8,
    Batch16,
}

impl DrafterNvfp4Tier {
    fn kernel_name(self) -> &'static str {
        match self {
            Self::Batch4 => "w4a16_gemv_batch4",
            Self::Batch8 => "w4a16_gemv_batch8",
            Self::Batch16 => "w4a16_gemv_batch16",
        }
    }
}

/// Pure γ→tier pick; `None` above the batched GEMV tiers' reach (γ > 16).
pub(super) fn drafter_nvfp4_tier(gamma: u32) -> Option<DrafterNvfp4Tier> {
    match gamma {
        1..=4 => Some(DrafterNvfp4Tier::Batch4),
        5..=8 => Some(DrafterNvfp4Tier::Batch8),
        9..=16 => Some(DrafterNvfp4Tier::Batch16),
        _ => None,
    }
}

/// Free memory the drafter NVFP4 twins must leave behind (the KV pool and
/// the remaining serve allocations are sized after them).
const DRAFTER_NVFP4_FREE_FLOOR: usize = 2 << 30;

/// Device bytes of the 7 NVFP4 twins per drafter layer (packed E2M1 plus one
/// E4M3 scale per 16 weights).
pub(super) fn drafter_nvfp4_bytes(
    layers: usize,
    h: usize,
    q_dim: usize,
    kv_dim: usize,
    inter: usize,
) -> usize {
    let per_layer = 2 * q_dim * h + 2 * kv_dim * h + 3 * inter * h;
    layers * (per_layer / 2 + per_layer / 16)
}

/// Rows of one batched drafter NVFP4 launch over `rows` rows of width `k`:
/// all of them when the row-invariant skinny kernels (`ATLAS_SKINNY_MMA`)
/// take `kernel` at that width (the weight streams once), else the 16 the
/// batch tiers hold (always for the Lightning product, whose exact contract
/// is the 16-row waves).
pub(super) fn drafter_nvfp4_wave(
    kernel: spark_runtime::gpu::KernelHandle,
    rows: u32,
    k: u32,
    lightning: bool,
) -> u32 {
    let all = !lightning
        && ops::skinny_mma::active()
            .is_some_and(|s| s.takes(ops::skinny_mma::Family::W4a16, kernel, rows, k));
    if all { rows.max(1) } else { 16 }
}

impl BlockDiffusionDraftHead {
    pub(super) fn try_install_nvfp4(&mut self, gpu: &dyn GpuBackend, nvfp4_tc: bool) -> Result<()> {
        if std::env::var("ATLAS_NO_DFLASH_DRAFTER_NVFP4").is_ok() {
            return Ok(());
        }
        let scalar_tier = drafter_nvfp4_tier(self.gamma as u32).filter(|tier| {
            match tier {
                DrafterNvfp4Tier::Batch4 => self.kernels.w4a16_gemv_batch4,
                DrafterNvfp4Tier::Batch8 => self.kernels.w4a16_gemv_batch8,
                DrafterNvfp4Tier::Batch16 => self.kernels.w4a16_gemv_batch16,
            }
            .0 != 0
        });
        let route = if nvfp4_tc && super::twins::nvfp4_tc_supported(self.gamma as u32) {
            "tensor-core tiers"
        } else if let Some(tier) = scalar_tier {
            tier.kernel_name()
        } else {
            tracing::warn!(
                "DFlash NVFP4: no resolved execution tier for γ={} — skipping drafter quantization",
                self.gamma
            );
            return Ok(());
        };
        let absmax = match gpu.kernel("quantize_nvfp4", "nvfp4_global_absmax") {
            Ok(k) => k,
            Err(e) => {
                tracing::warn!("NVFP4 absmax kernel missing: {e}");
                return Ok(());
            }
        };
        let quant = match gpu.kernel(
            "quantize_nvfp4",
            crate::weight_loader::glm5_mode::weight_quantizer()?,
        ) {
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
        // The twins sit beside the BF16 weights and are built before the KV
        // budget. Where they do not fit (Ripple-65B on a 96 GB gfx1151 leaves
        // ~1.5 GB pre-KV) the drafter keeps its BF16 arms instead of failing
        // the boot.
        // After the preshrink only k/v remain to quantize, and the BF16 they
        // would replace is already gone: the twins are not optional then.
        let parked = crate::weight_loader::dflash_preshrink::parked();
        let twin_bytes = if parked {
            drafter_nvfp4_bytes(self.layers.len(), h, 0, kv_dim, 0)
        } else {
            drafter_nvfp4_bytes(self.layers.len(), h, q_dim, kv_dim, inter)
        };
        if !parked
            && let Ok(free) = gpu.free_memory()
            && free < twin_bytes + DRAFTER_NVFP4_FREE_FLOOR
        {
            tracing::warn!(
                "DFlash NVFP4: twins need {:.2} GB, {:.2} GB free (floor {:.1} GB) — skipping drafter quantization",
                twin_bytes as f64 / 1e9,
                free as f64 / 1e9,
                DRAFTER_NVFP4_FREE_FLOOR as f64 / 1e9
            );
            return Ok(());
        }
        tracing::info!(
            "DFlash NVFP4: quantizing {} layers × 7 GEMMs for {}",
            self.layers.len(),
            route
        );
        // ATLAS_DFLASH_DROP_BF16 parked the dropped projections' twins.
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.q_proj_nvfp4 = Some(match take_twin(i, "q_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(&layer.q_proj, q_dim, h, gpu, absmax, quant, stream)?,
            });
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
            layer.o_proj_nvfp4 = Some(match take_twin(i, "o_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(&layer.o_proj, h, q_dim, gpu, absmax, quant, stream)?,
            });
            layer.gate_proj_nvfp4 = Some(match take_twin(i, "gate_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(&layer.gate_proj, inter, h, gpu, absmax, quant, stream)?,
            });
            layer.up_proj_nvfp4 = Some(match take_twin(i, "up_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(&layer.up_proj, inter, h, gpu, absmax, quant, stream)?,
            });
            layer.down_proj_nvfp4 = Some(match take_twin(i, "down_proj") {
                Some(twin) => twin,
                None => quantize_to_nvfp4(&layer.down_proj, h, inter, gpu, absmax, quant, stream)?,
            });
        }
        self.quant = DflashQuantization::Nvfp4Weights;
        tracing::info!("DFlash NVFP4: ready (quant = Nvfp4Weights)");
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn drafter_gemm(
        &self,
        gpu: &dyn GpuBackend,
        w_bf16: &DenseWeight,
        w_fp8: &Option<Fp8DenseWeight>,
        w_nvfp4: &Option<QuantizedWeight>,
        w_mx: Option<&Mxfp8Weight>,
        src: DevicePtr,
        dst: DevicePtr,
        n_out: u32,
        k_in: u32,
        stream: u64,
    ) -> Result<()> {
        self.drafter_gemm_rows(
            gpu,
            w_bf16,
            w_fp8,
            w_nvfp4,
            w_mx,
            src,
            dst,
            self.gamma as u32,
            n_out,
            k_in,
            stream,
        )
    }

    /// `drafter_gemm` with an explicit row count — the B×gamma seam calls it
    /// once per projection over all staged rows so every weight read is
    /// shared instead of looping per sequence. Arms, first that applies:
    /// NVFP4 twin on the tensor-core tiers (`ATLAS_DFLASH_NVFP4_TC`, any row
    /// count in 32-row pieces); NVFP4 twin on `w4a16_gemv_batch4` (≤4 rows);
    /// the Phase G FP8 mirror; the MXFP8 twin (`ATLAS_DFLASH_MXFP8`, ≤32
    /// rows); the BF16 weight.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn drafter_gemm_rows(
        &self,
        gpu: &dyn GpuBackend,
        w_bf16: &DenseWeight,
        w_fp8: &Option<Fp8DenseWeight>,
        w_nvfp4: &Option<QuantizedWeight>,
        w_mx: Option<&Mxfp8Weight>,
        src: DevicePtr,
        dst: DevicePtr,
        m: u32,
        n_out: u32,
        k_in: u32,
        stream: u64,
    ) -> Result<()> {
        if self.twins.nvfp4_tc
            && let Some(w) = w_nvfp4
            && self.nvfp4_tc_rows(gpu, w, src, dst, m, n_out, k_in, stream)?
        {
            return Ok(());
        }
        let nvfp4_kernel = drafter_nvfp4_tier(m).map(|tier| match tier {
            DrafterNvfp4Tier::Batch4 => self.kernels.w4a16_gemv_batch4,
            DrafterNvfp4Tier::Batch8 => self.kernels.w4a16_gemv_batch8,
            DrafterNvfp4Tier::Batch16 => self.kernels.w4a16_gemv_batch16,
        });
        if matches!(self.quant, DflashQuantization::Nvfp4Weights)
            && let Some(w) = w_nvfp4
            && let Some(k) = nvfp4_kernel
            && k.0 != 0
        {
            return ops::w4a16_gemv_batchm(gpu, k, src, w, dst, m, n_out, k_in, stream);
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
                m,
                n_out,
                k_in,
                stream,
            );
        }
        if let Some(mx) = w_mx
            && self.mxfp8_rows(gpu, mx, src, dst, m, n_out, k_in, stream)?
        {
            return Ok(());
        }
        self.drafter_dense_gemm(gpu, src, w_bf16, dst, m, n_out, k_in, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drafter_nvfp4_wave_is_16_rows_off_the_skinny_kernels() {
        use spark_runtime::gpu::KernelHandle;
        // No `ATLAS_SKINNY_MMA` in unit tests: the batch tiers' 16-row waves.
        assert_eq!(drafter_nvfp4_wave(KernelHandle(7), 128, 5120, false), 16);
        assert_eq!(drafter_nvfp4_wave(KernelHandle(7), 128, 5120, true), 16);
    }

    #[test]
    fn drafter_nvfp4_tier_pick() {
        assert_eq!(drafter_nvfp4_tier(1), Some(DrafterNvfp4Tier::Batch4));
        assert_eq!(drafter_nvfp4_tier(4), Some(DrafterNvfp4Tier::Batch4));
        assert_eq!(drafter_nvfp4_tier(5), Some(DrafterNvfp4Tier::Batch8));
        assert_eq!(drafter_nvfp4_tier(8), Some(DrafterNvfp4Tier::Batch8));
        assert_eq!(drafter_nvfp4_tier(9), Some(DrafterNvfp4Tier::Batch16));
        assert_eq!(drafter_nvfp4_tier(16), Some(DrafterNvfp4Tier::Batch16));
        assert_eq!(drafter_nvfp4_tier(17), None);
        assert_eq!(drafter_nvfp4_tier(0), None);
    }

    #[test]
    fn drafter_nvfp4_twin_bytes_ripple() {
        // Ripple-65B DFlash2: 5 layers, h 5120, q 80x128, kv 8x128, inter 40960.
        assert_eq!(drafter_nvfp4_bytes(5, 5120, 10240, 1024, 40960), 2_093_875_200);
    }
}

#[cfg(test)]
#[path = "nvfp4_install_tests.rs"]
mod install_tests;
