// SPDX-License-Identifier: AGPL-3.0-only

//! Literal-M1 output projection for exact Lightning verification.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl NemotronMamba2Layer {
    pub(super) fn decode_out_proj_exact(
        &self,
        gated_out: DevicePtr,
        out: DevicePtr,
        h: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(ref w) = self.out_proj_bf16 {
            ops::dense_gemv(
                ctx.gpu,
                self.dense_gemv_bf16_k,
                gated_out,
                w,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else if let Some(ref fp8w) = self.out_proj_fp8 {
            ops::w8a16_gemv(
                ctx.gpu,
                self.w8a16_gemv_k,
                gated_out,
                fp8w.weight,
                fp8w.row_scale,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        } else {
            ops::w4a16_gemv(
                ctx.gpu,
                self.w4a16_gemv_k,
                gated_out,
                &self.ssm.out_proj,
                out,
                h as u32,
                self.d_inner as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
