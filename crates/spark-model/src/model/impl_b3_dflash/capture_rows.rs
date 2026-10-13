// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

impl TransformerModel {
    /// Copy `rows` target hidden rows, starting at forward row `row0`, into
    /// capture storage whose rows are `dst_stride` bytes apart. mHC targets
    /// (GLM-5) keep the residual in the FP32 highway and `hidden_states` holds
    /// only the next sublayer's mixed input, so they contract the highway by
    /// stream mean — the reference DFlash target hidden — instead.
    pub(super) fn dflash_capture_rows(
        &self,
        row0: usize,
        rows: usize,
        dst: spark_runtime::gpu::DevicePtr,
        dst_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        if rows == 0 {
            return Ok(());
        }
        let trace = std::env::var("ATLAS_DFLASH_CAPTURE_TRACE").as_deref() == Ok("1");
        let result = self.dflash_capture_rows_inner(row0, rows, dst, dst_stride, stream);
        if trace && result.is_ok() {
            // Debug only: synchronous readback of the first captured row.
            self.gpu.synchronize(stream)?;
            let mut raw = vec![0u8; h * 2];
            self.gpu.copy_d2h(dst, &mut raw)?;
            let norm = raw
                .chunks_exact(2)
                .map(|b| f32::from_bits((u16::from_le_bytes([b[0], b[1]]) as u32) << 16))
                .map(|v| v * v)
                .sum::<f32>()
                .sqrt();
            tracing::info!(row0, rows, dst_stride, norm, "DFlash capture trace");
        }
        result
    }

    fn dflash_capture_rows_inner(
        &self,
        row0: usize,
        rows: usize,
        dst: spark_runtime::gpu::DevicePtr,
        dst_stride: usize,
        stream: u64,
    ) -> Result<()> {
        let h = self.config.hidden_size;
        let hc = self.config.hc_mult;
        if hc > 1 {
            static KERNEL: std::sync::OnceLock<spark_runtime::gpu::KernelHandle> =
                std::sync::OnceLock::new();
            let kernel = match KERNEL.get() {
                Some(k) => *k,
                None => *KERNEL.get_or_init(|| {
                    self.gpu
                        .kernel(
                            "hyper_connection",
                            &crate::layers::ops::hc_kernel_name(
                                &self.config.model_type,
                                "hc_contract_strided",
                            ),
                        )
                        .unwrap_or(spark_runtime::gpu::KernelHandle(0))
                }),
            };
            anyhow::ensure!(kernel.0 != 0, "DFlash mHC capture kernel unavailable");
            anyhow::ensure!(
                dst_stride.is_multiple_of(2),
                "DFlash capture stride must be BF16-aligned"
            );
            return spark_runtime::kernel_args::KernelLaunch::new(self.gpu.as_ref(), kernel)
                .grid([rows as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buffers.hc_streams().offset(
                    row0 * hc * h * crate::layers::ops::hc_elem_bytes(&self.config.model_type),
                ))
                .arg_ptr(dst)
                .arg_u32(h as u32)
                .arg_u32(hc as u32)
                .arg_u32((dst_stride / 2) as u32)
                .launch(stream);
        }
        let src = self.buffers.hidden_states();
        for t in 0..rows {
            self.gpu.copy_d2d_async(
                src.offset((row0 + t) * h * 2),
                dst.offset(t * dst_stride),
                h * 2,
                stream,
            )?;
        }
        Ok(())
    }
}
