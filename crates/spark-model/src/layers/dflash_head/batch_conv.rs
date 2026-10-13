// SPDX-License-Identifier: AGPL-3.0-only

//! Per-sequence DFlash2 grouped causal conv on the staged `[B, gamma]` rows.
//!
//! The conv is causal along the row dim, so the batch never crosses a
//! sequence boundary: every op runs on one sequence's `gamma`-row slice, and
//! the kernel-projection GEMM keeps the serial `m = gamma` shape per slice.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::BlockDiffusionDraftHead;

impl BlockDiffusionDraftHead {
    /// DFlash2 grouped causal conv `prepare` on each sequence's `[gamma]`
    /// row slice. The conv is causal along the row dim, so the batch never
    /// crosses a sequence boundary. The delta GEMM stays at the serial
    /// m=gamma shape per slice, except where the tensor-core GEMV tiers serve
    /// all B·gamma rows: there one GEMM reads `kernel_projection` once for
    /// the whole batch (20 sites x 8 MB per extra sequence on GLM-5.3) and
    /// the per-slice deltas land in the same `[sequence][gamma]` layout.
    pub(super) fn staged_conv_prepare(
        &self,
        conv: &super::Dflash2Conv,
        buf: DevicePtr,
        batch_size: u32,
        hidden: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.batch_conv_delta.is_null() && !self.batch_conv_out.is_null(),
            "DFlash batched conv scratch is null"
        );
        let row_elements = (self.gamma as u32)
            .checked_mul(hidden)
            .ok_or_else(|| anyhow::anyhow!("DFlash conv row elements overflow"))?;
        let row_bytes = row_elements as usize * 2;
        let delta_row_bytes = (self.gamma)
            .checked_mul(2 * conv.kernel_size * conv.num_groups)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash conv delta row bytes overflow"))?;
        let gemm = |src, w: &crate::weight_map::DenseWeight, dst, m, n, k| {
            self.drafter_dense_gemm(ctx.gpu, src, w, dst, m, n, k, stream)
        };
        let total_rows = batch_size * self.gamma as u32;
        let joint = matches!(
            super::small_m_gemm::small_m_arm(
                self.kernels.small_m_gemv,
                self.kernels.dense_gemv_batchm.0 != 0,
                self.kernels.dense_gemv_tc16.0 != 0 && self.kernels.dense_gemv_tc32.0 != 0,
                total_rows,
                hidden,
            ),
            super::small_m_gemm::SmallMArm::TensorCore
                | super::small_m_gemm::SmallMArm::TensorCoreSplit
        );
        if joint {
            conv.project_deltas(&gemm, buf, self.batch_conv_delta, total_rows)?;
        }
        for sequence in 0..batch_size as usize {
            let buf_seq = buf.offset(sequence * row_bytes);
            let delta_seq = self.batch_conv_delta.offset(sequence * delta_row_bytes);
            let out_seq = self.batch_conv_out.offset(sequence * row_bytes);
            if !joint {
                conv.project_deltas(&gemm, buf_seq, delta_seq, self.gamma as u32)?;
            }
            conv.apply_input(
                ctx.gpu,
                self.kernels.dflash2_conv,
                buf_seq,
                delta_seq,
                out_seq,
                self.gamma as u32,
                stream,
            )?;
        }
        // The slices are contiguous: one copy back for the whole batch.
        ctx.gpu.copy_d2d_async(
            self.batch_conv_out,
            buf,
            batch_size as usize * row_bytes,
            stream,
        )
    }

    /// The matching `finish`: output-side conv on the sublayer output slice.
    pub(super) fn staged_conv_finish(
        &self,
        conv: &super::Dflash2Conv,
        buf: DevicePtr,
        batch_size: u32,
        hidden: u32,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            !self.batch_conv_delta.is_null() && !self.batch_conv_out.is_null(),
            "DFlash batched conv scratch is null"
        );
        let row_elements = (self.gamma as u32)
            .checked_mul(hidden)
            .ok_or_else(|| anyhow::anyhow!("DFlash conv row elements overflow"))?;
        let row_bytes = row_elements as usize * 2;
        let delta_row_bytes = (self.gamma)
            .checked_mul(2 * conv.kernel_size * conv.num_groups)
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| anyhow::anyhow!("DFlash conv delta row bytes overflow"))?;
        for sequence in 0..batch_size as usize {
            let buf_seq = buf.offset(sequence * row_bytes);
            let delta_seq = self
                .batch_conv_delta
                .offset(sequence * delta_row_bytes + 2 * conv.num_groups * 2);
            let out_seq = self.batch_conv_out.offset(sequence * row_bytes);
            conv.finish(
                ctx.gpu,
                self.kernels.dflash2_conv,
                buf_seq,
                delta_seq,
                out_seq,
                self.gamma as u32,
                stream,
            )?;
        }
        ctx.gpu.copy_d2d_async(
            self.batch_conv_out,
            buf,
            batch_size as usize * row_bytes,
            stream,
        )
    }
}
