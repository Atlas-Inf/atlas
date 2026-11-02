// SPDX-License-Identifier: AGPL-3.0-only

//! Native MXFP8 GDN projections (nvidia/Qwen3.8-Flash-Next-FP4-Local).
//!
//! That checkpoint ships `in_proj_qkv`, `in_proj_z` and `out_proj` as MXFP8.
//! A layer built from them holds [`Mx8Gdn`] and nothing else for those two
//! projections: `ssm.in_proj_qkvz` is NULL and every BF16 / FP8 / NVFP4 copy
//! is `None`. Each dispatch site checks `self.mx8` first, so the bytes the
//! checkpoint ships are the bytes every forward reads; a site that forgot
//! the check would launch on a NULL weight and fault, not compute quietly
//! with another copy.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::Qwen3SsmLayer;
use crate::layer::ForwardContext;
use crate::layers::ops::qwen4exp_w8a8::{self, MxPrefill};
use crate::layers::ops::{self, Mxfp8Proj, Mxfp8ProjKernels, Mxfp8Scratch};

/// The MXFP8 `[Q|K|V|Z]` input projection (rows of `in_proj_qkv` then
/// `in_proj_z`, sequential) and `out_proj`, with the kernels that read them,
/// the shared BF16 scratch prefill expands into and, under
/// `ATLAS_QWEN4EXP_PREFILL_W8A8=1`, the W8A8 GEMMs prefill runs instead.
#[derive(Debug, Clone, Copy)]
pub struct Mx8Gdn {
    pub qkvz: Mxfp8Proj,
    pub out: Mxfp8Proj,
    pub kernels: Mxfp8ProjKernels,
    pub scratch: Mxfp8Scratch,
    pub w8a8: Option<MxPrefill<2>>,
}

impl Qwen3SsmLayer {
    /// Install the MXFP8 projections. Refused unless the layer carries no
    /// other copy of either projection and its QKVZ output is sequential,
    /// which is the layout `in_proj_qkv || in_proj_z` rows produce.
    pub fn set_mx8_projections(&mut self, mx8: Mx8Gdn) -> Result<()> {
        ensure!(
            self.sequential_qkvz,
            "MXFP8 GDN: the QKVZ projection must be sequential [Q|K|V|Z]"
        );
        ensure!(
            self.ssm.in_proj_qkvz.weight.is_null()
                && self.out_proj_dense.is_none()
                && self.qkvz_nvfp4.is_none()
                && self.qkvz_fp8w.is_none()
                && self.out_proj_fp8w.is_none()
                && self.qkvz_fp8w_rowwise.is_none()
                && self.out_proj_fp8w_rowwise.is_none()
                && self.qkvz_q2.is_none(),
            "MXFP8 GDN: the layer already holds another copy of its projections"
        );
        let largest = mx8.qkvz.bf16_bytes().max(mx8.out.bf16_bytes());
        ensure!(
            !mx8.scratch.ptr.is_null() && mx8.scratch.bytes >= largest,
            "MXFP8 GDN: prefill scratch {} B is smaller than the {largest} B expansion",
            mx8.scratch.bytes
        );
        self.mx8 = Some(mx8);
        Ok(())
    }

    /// `dst[m, qkvz] = act[m, hidden] @ W_qkvz^T`, sequential `[Q|K|V|Z]`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mx8_qkvz(
        &self,
        mx8: &Mx8Gdn,
        ctx: &ForwardContext,
        act: DevicePtr,
        dst: DevicePtr,
        m: usize,
        prefill: bool,
        stream: u64,
    ) -> Result<()> {
        Self::mx8_proj(mx8, &mx8.qkvz, ctx, act, dst, m, prefill, stream)
    }

    /// `dst[m, hidden] = act[m, value_dim] @ W_out^T`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn mx8_out(
        &self,
        mx8: &Mx8Gdn,
        ctx: &ForwardContext,
        act: DevicePtr,
        dst: DevicePtr,
        m: usize,
        prefill: bool,
        stream: u64,
    ) -> Result<()> {
        Self::mx8_proj(mx8, &mx8.out, ctx, act, dst, m, prefill, stream)
    }

    /// A prefill pass takes the W8A8 GEMM where it is installed; everything
    /// else `mxfp8_proj`.
    #[allow(clippy::too_many_arguments)]
    fn mx8_proj(
        mx8: &Mx8Gdn,
        proj: &Mxfp8Proj,
        ctx: &ForwardContext,
        act: DevicePtr,
        dst: DevicePtr,
        m: usize,
        prefill: bool,
        stream: u64,
    ) -> Result<()> {
        if prefill
            && let Some(w) = &mx8.w8a8
            && w.project(ctx.gpu, ctx.buffers, proj, act, dst, m, stream)?
        {
            return Ok(());
        }
        ops::mxfp8_proj(
            ctx.gpu,
            &mx8.kernels,
            proj,
            act,
            dst,
            m,
            mx8.scratch,
            stream,
        )
    }

    /// Arm the collapse that writes rows `[row0, row0 + rows)` of `x`
    /// (`[total, hidden]`, the W8A8 projections' input) to emit their MXFP8
    /// image too (`ops::qwen4exp_w8a8::Producer`). `None` without W8A8.
    pub(crate) fn w8a8_producer(
        &self,
        ctx: &ForwardContext,
        x: DevicePtr,
        total: usize,
        row0: usize,
        rows: usize,
    ) -> Result<Option<qwen4exp_w8a8::Producer>> {
        if self.mx8.as_ref().is_none_or(|m| m.w8a8.is_none()) {
            return Ok(None);
        }
        let h = ctx.config.hidden_size;
        qwen4exp_w8a8::Producer::arm(ctx.buffers, x, total, row0, rows, h)
    }
}
