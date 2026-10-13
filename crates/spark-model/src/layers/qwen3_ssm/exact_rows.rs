// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_BATCH_FAST`: the GDN layer's two big projections over
//! several rows with single-row decode's arithmetic per row
//! (`model/qwen4exp_batch_fast.rs`).
//!
//! `ssm_forward` (serial decode) projects one row with `w8a16_gemv` (FP8
//! block-scaled copy), `w4a16_gemv[_sw]` (NVFP4) or `dense_gemv_bf16` (BF16),
//! in that order of preference. Each has a batched GEMV whose rows are those
//! bytes — `w8a16_gemv_batch4/16`, the `w4a16_gemv_batchN` tiers,
//! `dense_gemv_bf16_batchm` — that reads the weight once per launch. The
//! multi-sequence decode and the batched verifies take cuBLASLt / tile GEMMs
//! at some widths instead, which are not; under the switch both come here.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Which projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GdnProj {
    /// `in_proj_qkvz`: `[rows, hidden] -> [rows, qkvz]` (sequential layout).
    Qkvz,
    /// `out_proj`: `[rows, value_dim] -> [rows, hidden]`.
    Out,
}

impl Qwen3SsmLayer {
    /// `rows` rows of `input` through projection `proj` into `output` (rows
    /// contiguous at the projection's width), each row byte-identical to
    /// `ssm_forward`'s single-row GEMV. `Ok(false)`, nothing launched, when
    /// the switch is off or serial decode would take an arm this has no
    /// exact batched twin for (Q2, interleaved QKVZ, `ATLAS_GDN_FP8_DECODE`):
    /// the caller keeps its own dispatch.
    pub(super) fn exact_rows_proj(
        &self,
        proj: GdnProj,
        input: DevicePtr,
        output: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        if !ctx.levers.qwen4exp_batch_fast
            || rows == 0
            || !self.sequential_qkvz
            || self.qkvz_q2.is_some()
            || ctx.levers.gdn_fp8_decode
        {
            return Ok(false);
        }
        let h = ctx.config.hidden_size;
        let value_dim = ctx.config.linear_num_value_heads * ctx.config.linear_value_head_dim;
        let (n, k) = match proj {
            GdnProj::Qkvz => (ctx.config.ssm_qkvz_size(), h),
            GdnProj::Out => (h, value_dim),
        };
        let (in_row, out_row) = (k * 2, n * 2);
        let fp8 = match proj {
            GdnProj::Qkvz => self.qkvz_fp8w.as_ref(),
            GdnProj::Out => self.out_proj_fp8w.as_ref(),
        };
        if let Some(fp8) = fp8 {
            if fp8.scale_format != crate::weight_map::WeightQuantFormat::Fp8BlockScaled {
                return Ok(false);
            }
            // batch4 to 4 rows, batch16 past that: one template, each row
            // `w8a16_gemv`'s bytes.
            let (kernel, chunk) = if rows <= 4 || self.w8a16_gemv_batch16_k.0 == 0 {
                (self.w8a16_gemv_batch4_k, 4)
            } else {
                (self.w8a16_gemv_batch16_k, 16)
            };
            if kernel.0 == 0 {
                return Ok(false);
            }
            for first in (0..rows).step_by(chunk) {
                let m = (rows - first).min(chunk);
                ops::w8a16_gemv_batch4(
                    ctx.gpu,
                    kernel,
                    input.offset(first * in_row),
                    fp8.weight,
                    fp8.row_scale,
                    output.offset(first * out_row),
                    m as u32,
                    n as u32,
                    k as u32,
                    stream,
                )?;
            }
            return Ok(true);
        }
        let nvfp4 = match proj {
            GdnProj::Qkvz => self.qkvz_nvfp4.as_ref(),
            GdnProj::Out => (self.out_proj_dense.is_none() && !self.ssm.out_proj.weight.is_null())
                .then_some(&self.ssm.out_proj),
        };
        if let Some(w) = nvfp4 {
            if !self.w4a16_batchm.has_base() {
                return Ok(false);
            }
            let chunk = if self.w4a16_batchm.width(8).is_some() {
                8
            } else {
                4
            };
            for first in (0..rows).step_by(chunk) {
                let m = (rows - first).min(chunk) as u32;
                ops::w4a16_gemv_batchm(
                    ctx.gpu,
                    self.w4a16_batchm.kernel(m),
                    input.offset(first * in_row),
                    w,
                    output.offset(first * out_row),
                    m,
                    n as u32,
                    k as u32,
                    stream,
                )?;
            }
            return Ok(true);
        }
        let dense = match proj {
            GdnProj::Qkvz => {
                (!self.ssm.in_proj_qkvz.weight.is_null()).then_some(&self.ssm.in_proj_qkvz)
            }
            GdnProj::Out => self.out_proj_dense.as_ref(),
        };
        let Some(w) = dense else {
            return Ok(false);
        };
        if self.dense_gemv_batchm_k.0 == 0 {
            return Ok(false);
        }
        ops::dense_gemv_batchm_chunked(
            ctx.gpu,
            self.dense_gemv_batchm_k,
            input,
            w,
            output,
            rows as u32,
            n as u32,
            k as u32,
            n as u32,
            stream,
        )?;
        Ok(true)
    }
}
