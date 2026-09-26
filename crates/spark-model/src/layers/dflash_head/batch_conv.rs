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
    /// crosses a sequence boundary; the delta GEMM stays at the serial m=gamma
    /// shape per slice.
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
        for sequence in 0..batch_size as usize {
            let buf_seq = buf.offset(sequence * row_bytes);
            let delta_seq = self.batch_conv_delta.offset(sequence * delta_row_bytes);
            let out_seq = self.batch_conv_out.offset(sequence * row_bytes);
            conv.prepare(
                ctx.gpu,
                &|src, w, dst, m, n, k| {
                    self.drafter_dense_gemm(ctx.gpu, src, w, dst, m, n, k, stream)
                },
                self.kernels.dflash2_conv,
                buf_seq,
                delta_seq,
                out_seq,
                self.gamma as u32,
                stream,
            )?;
            ctx.gpu
                .copy_d2d_async(out_seq, buf_seq, row_bytes, stream)?;
        }
        Ok(())
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
            ctx.gpu
                .copy_d2d_async(out_seq, buf_seq, row_bytes, stream)?;
        }
        Ok(())
    }
}
