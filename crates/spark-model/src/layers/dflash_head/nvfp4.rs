// SPDX-License-Identifier: AGPL-3.0-only

//! Drafter NVFP4: load-time quant + small-M `w4a16_gemv_batch4` dispatch.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{BlockDiffusionDraftHead, DflashQuantization};
use crate::layers::ops;
use crate::weight_map::{quantize_to_nvfp4, DenseWeight, Fp8DenseWeight, QuantizedWeight};

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
        for layer in &mut self.layers {
            layer.q_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.q_proj, q_dim, h, gpu, absmax, quant, stream,
            )?);
            layer.k_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.k_proj, kv_dim, h, gpu, absmax, quant, stream,
            )?);
            layer.v_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.v_proj, kv_dim, h, gpu, absmax, quant, stream,
            )?);
            layer.o_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.o_proj, h, q_dim, gpu, absmax, quant, stream,
            )?);
            layer.gate_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.gate_proj, inter, h, gpu, absmax, quant, stream,
            )?);
            layer.up_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.up_proj, inter, h, gpu, absmax, quant, stream,
            )?);
            layer.down_proj_nvfp4 = Some(quantize_to_nvfp4(
                &layer.down_proj, h, inter, gpu, absmax, quant, stream,
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
        if matches!(self.quant, DflashQuantization::Nvfp4Weights)
            && g <= 4
            && let Some(w) = w_nvfp4
            && self.kernels.w4a16_gemv_batch4.0 != 0
        {
            return ops::w4a16_gemv_batchm(
                gpu,
                self.kernels.w4a16_gemv_batch4,
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
        ops::dense_gemm_bf16_pipelined(
            gpu,
            self.kernels.dense_gemm_pipelined,
            src,
            w_bf16,
            dst,
            g,
            n_out,
            k_in,
            stream,
        )
    }
}
