// SPDX-License-Identifier: AGPL-3.0-only

//! Native MXFP8 attention projections (nvidia/Qwen3.8-Flash-Next-FP4-Local).
//!
//! That checkpoint ships q/k/v/o as MXFP8. A layer built from them holds
//! each as `QuantWeight::Mxfp8` and nothing else: the BF16 `attn.*_proj`
//! slots are NULL and no NVFP4 / FP8 twin exists. Every projection site asks
//! [`Qwen3AttentionLayer::try_mx8`] first; a site that did not would fall
//! through to a NULL BF16 slot and fault rather than read another copy.

use anyhow::{Context, Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops::{self, Mxfp8ProjKernels, Mxfp8Scratch};
use crate::weight_map::{Mxfp8Proj, QuantWeight};

/// Kernels and the prefill expansion scratch for MXFP8 projections.
#[derive(Debug, Clone, Copy)]
pub struct Mx8AttnRt {
    pub kernels: Mxfp8ProjKernels,
    pub scratch: Mxfp8Scratch,
    /// The q/k/v/o W8A8 operands (tiled scales) built when
    /// `ATLAS_QWEN4EXP_PREFILL_W8A8` was on at load; `None` on the weight
    /// the switch leaves at the BF16 expand + GEMM.
    pub w8a8: [Option<ops::MxW8a8>; 4],
}

impl Qwen3AttentionLayer {
    /// Install MXFP8 q/k/v/o. Refused when the BF16 slots still hold
    /// weights, when another decode copy exists, or when the scratch cannot
    /// take the largest expansion.
    pub fn set_mx8_projections(
        &mut self,
        [q, k, v, o]: [Mxfp8Proj; 4],
        rt: Mx8AttnRt,
    ) -> Result<()> {
        ensure!(
            self.attn.q_proj.weight.is_null()
                && self.attn.k_proj.weight.is_null()
                && self.attn.v_proj.weight.is_null()
                && self.attn.o_proj.is_null()
                && self.o_dense_bf16.is_none()
                && [
                    &self.q_weight,
                    &self.k_weight,
                    &self.v_weight,
                    &self.o_weight
                ]
                .iter()
                .all(|w| w.is_none()),
            "MXFP8 attention: the layer already holds another copy of its projections"
        );
        let largest = [q, k, v, o].iter().map(Mxfp8Proj::bf16_bytes).max();
        ensure!(
            !rt.scratch.ptr.is_null() && rt.scratch.bytes >= largest.unwrap_or(0),
            "MXFP8 attention: prefill scratch {} B is smaller than the largest expansion",
            rt.scratch.bytes
        );
        self.q_weight = Some(QuantWeight::Mxfp8(q));
        self.k_weight = Some(QuantWeight::Mxfp8(k));
        self.v_weight = Some(QuantWeight::Mxfp8(v));
        self.o_weight = Some(QuantWeight::Mxfp8(o));
        self.mx8_rt = Some(rt);
        Ok(())
    }

    /// `out[m, n] = input[m, k] @ W^T` when `weight` is MXFP8 (`Some(result)`);
    /// `None` for any other format, which the caller dispatches itself.
    /// `input` and `out` rows are contiguous at the projection's widths.
    /// `exact` selects the numerics family: decode and verify take
    /// `mxfp8_proj_exact_rows` (every row the serial GEMV's bytes at any m),
    /// prefill takes `mxfp8_proj_prefill` — expand + GEMM past 32 rows, or
    /// the W8A8 `w8a8` operand when `ATLAS_QWEN4EXP_PREFILL_W8A8` built it.
    /// Only the prefill branch reads `w8a8`; decode and verify pass `None`.
    pub(crate) fn try_mx8(
        &self,
        ctx: &ForwardContext,
        weight: Option<&QuantWeight>,
        w8a8: Option<&ops::MxW8a8>,
        input: DevicePtr,
        out: DevicePtr,
        m: usize,
        exact: bool,
        stream: u64,
    ) -> Option<Result<()>> {
        let proj = weight.and_then(QuantWeight::as_mxfp8)?;
        Some(
            self.mx8_rt
                .as_ref()
                .context("MXFP8 attention weight without its kernels")
                .and_then(|rt| {
                    if exact {
                        ops::mxfp8_proj_exact_rows(
                            ctx.gpu,
                            &rt.kernels,
                            proj,
                            input,
                            out,
                            m,
                            stream,
                        )
                    } else {
                        ops::mxfp8_proj_prefill(
                            ctx.gpu,
                            &rt.kernels,
                            proj,
                            w8a8,
                            input,
                            out,
                            m,
                            rt.scratch,
                            stream,
                        )
                    }
                }),
        )
    }

    /// The prefill guard for weights that are not GEMM-ready as stored:
    /// native MXFP8 ([`Self::try_mx8`]), then keep-packed Q2_0
    /// (`try_q2_prefill`). `None` for every other format. Checked first at
    /// each prefill projection: the arms after it read NULL slots for both.
    pub(crate) fn try_unpacked_prefill(
        &self,
        ctx: &ForwardContext,
        weight: Option<&QuantWeight>,
        input: DevicePtr,
        out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Option<Result<()>> {
        // This weight's W8A8 slot, by which stored projection it is.
        let w8a8 = self.mx8_rt.as_ref().and_then(|rt| {
            [
                &self.q_weight,
                &self.k_weight,
                &self.v_weight,
                &self.o_weight,
            ]
            .iter()
            .position(|w| {
                w.as_ref()
                    .zip(weight)
                    .is_some_and(|(a, b)| std::ptr::eq(a, b))
            })
            .and_then(|i| rt.w8a8[i].as_ref())
        });
        self.try_mx8(ctx, weight, w8a8, input, out, m as usize, false, stream)
            .or_else(|| self.try_q2_prefill(ctx, weight, input, out, m, stream))
    }
}
