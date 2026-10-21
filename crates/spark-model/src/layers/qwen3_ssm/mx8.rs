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
use crate::layers::ops::{self, Mxfp8Proj, Mxfp8ProjKernels, Mxfp8Scratch};

/// The MXFP8 `[Q|K|V|Z]` input projection (rows of `in_proj_qkv` then
/// `in_proj_z`, sequential) and `out_proj`, with the kernels that read them
/// and the shared BF16 scratch prefill expands into.
#[derive(Debug, Clone, Copy)]
pub struct Mx8Gdn {
    pub qkvz: Mxfp8Proj,
    pub out: Mxfp8Proj,
    pub kernels: Mxfp8ProjKernels,
    pub scratch: Mxfp8Scratch,
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
    pub(super) fn mx8_qkvz(
        &self,
        mx8: &Mx8Gdn,
        ctx: &ForwardContext,
        act: DevicePtr,
        dst: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        ops::mxfp8_proj(
            ctx.gpu,
            &mx8.kernels,
            &mx8.qkvz,
            act,
            dst,
            m,
            mx8.scratch,
            stream,
        )
    }

    /// `dst[m, hidden] = act[m, value_dim] @ W_out^T`.
    pub(super) fn mx8_out(
        &self,
        mx8: &Mx8Gdn,
        ctx: &ForwardContext,
        act: DevicePtr,
        dst: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        ops::mxfp8_proj(
            ctx.gpu,
            &mx8.kernels,
            &mx8.out,
            act,
            dst,
            m,
            mx8.scratch,
            stream,
        )
    }
}
