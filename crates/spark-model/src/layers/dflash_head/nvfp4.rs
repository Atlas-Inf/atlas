// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter NVFP4: load-time quant + small-M `w4a16_gemv_batch4` dispatch.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BlockDiffusionDraftHead, DflashQuantization};
use crate::layers::ops;
use crate::weight_loader::dflash_preshrink::take_twin;
use crate::weight_map::{
    DenseWeight, Fp8DenseWeight, Mxfp8Weight, QuantizedWeight, quantize_to_nvfp4,
};

impl BlockDiffusionDraftHead {
    pub(super) fn try_install_nvfp4(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        if std::env::var("ATLAS_NO_DFLASH_DRAFTER_NVFP4").is_ok() {
            return Ok(());
        }
        if self.kernels.w4a16_gemv_batch4.0 == 0 {
            tracing::warn!("ATLAS_DFLASH_DRAFTER_NVFP4=1 but w4a16_gemv_batch4 missing");
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
            "DFlash NVFP4: quantizing {} layers × 7 GEMMs for w4a16_gemv_batch4",
            self.layers.len()
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
        if matches!(self.quant, DflashQuantization::Nvfp4Weights)
            && m <= 4
            && let Some(w) = w_nvfp4
            && self.kernels.w4a16_gemv_batch4.0 != 0
        {
            return ops::w4a16_gemv_batchm(
                gpu,
                self.kernels.w4a16_gemv_batch4,
                src,
                w,
                dst,
                m,
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
