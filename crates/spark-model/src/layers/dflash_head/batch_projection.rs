// SPDX-License-Identifier: AGPL-3.0-only

use super::BlockDiffusionDraftHead;
use anyhow::Result;

impl BlockDiffusionDraftHead {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_staged_projection(
        &self,
        batch_size: u32,
        src: spark_runtime::gpu::DevicePtr,
        weight: &crate::weight_map::DenseWeight,
        weight_fp8: &Option<crate::weight_map::Fp8DenseWeight>,
        weight_nvfp4: &Option<crate::weight_map::QuantizedWeight>,
        dst: spark_runtime::gpu::DevicePtr,
        n_out: u32,
        k_in: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let total_rows = batch_size
            .checked_mul(self.gamma as u32)
            .ok_or_else(|| anyhow::anyhow!("DFlash staged projection row overflow"))?;
        if matches!(self.quant, super::DflashQuantization::Nvfp4Weights)
            && let Some(weight) = weight_nvfp4
        {
            let kernel = match total_rows {
                1..=4 => self.kernels.w4a16_gemv_batch4,
                5..=8 => self.kernels.w4a16_gemv_batch8,
                9..=32 => self.kernels.w4a16_gemv_batch16,
                _ => spark_runtime::gpu::KernelHandle(0),
            };
            if kernel.0 != 0 {
                let mut row = 0u32;
                while row < total_rows {
                    let rows = (total_rows - row).min(16);
                    crate::layers::ops::w4a16_gemv_batchm(
                        ctx.gpu,
                        kernel,
                        src.offset(row as usize * k_in as usize * 2),
                        weight,
                        dst.offset(row as usize * n_out as usize * 2),
                        rows,
                        n_out,
                        k_in,
                        stream,
                    )?;
                    row += rows;
                }
                return Ok(());
            }
        }
        let rows = self.gamma;
        let src_row_bytes = rows
            .checked_mul(k_in as usize)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash staged projection input offset overflow"))?;
        let dst_row_bytes = rows
            .checked_mul(n_out as usize)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash staged projection output offset overflow"))?;
        for sequence in 0..batch_size as usize {
            self.drafter_gemm(
                ctx.gpu,
                weight,
                weight_fp8,
                weight_nvfp4,
                src.offset(sequence * src_row_bytes),
                dst.offset(sequence * dst_row_bytes),
                n_out,
                k_in,
                stream,
            )?;
        }
        Ok(())
    }
}
