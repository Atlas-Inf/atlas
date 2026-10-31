// SPDX-License-Identifier: AGPL-3.0-only

//! KDA q/k/v/o on the checkpoint's own MXFP8 bytes
//! (`ATLAS_GLM_FP4L_KDA_MXFP8_DECODE`, loaded by `weight_loader::glm5::kda_mx`).
//!
//! Every decode, multi-sequence decode and verify projection of a layer that
//! carries the twin goes through `ops::mxfp8_gemv_rows`: the
//! `mxfp8_gemv_tc{8,16,32}` tiers, 32 rows a launch, whose output row bits do
//! not depend on how many rows share the launch. So a 1-row decode, a k-row
//! DFlash verify and a multi-sequence batch agree bit for bit, the property
//! the canonical W4A16 tier gave the NVFP4 twins.
//!
//! Every prefill projection of a layer with a W8A8 operand
//! (`ATLAS_GLM_FP4L_KDA_W8A8_PREFILL`) is the cuBLASLt MXFP8 W8A8 GEMM:
//! the activation quantized per row and 32 values into the arena's FP8
//! scratch, then the row bucket's in-order kernel, row-invariant too.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{Glm5KdaLayer, Glm5Projection};
use crate::layer::ForwardContext;
use crate::layers::ops;

/// `ATLAS_GLM_FP4L_KDA_QKV_FUSE` (default on; `0` runs Q, K, V as three
/// launches, for A/B: the bytes are the same either way).
fn qkv_fuse_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_FP4L_KDA_QKV_FUSE").as_deref() != Ok("0"))
}

impl Glm5KdaLayer {
    /// `output[m, n] = input[m, k] · W^T` on `weight`'s MXFP8 twin; `false`,
    /// nothing launched, when it has none.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::glm5_kda) fn try_project_mxfp8(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(twin) = weight.mxfp8.as_ref() else {
            return Ok(false);
        };
        ensure!(
            self.mxfp8_gemv_k.iter().all(|k| k.0 != 0),
            "GLM KDA MXFP8 decode: the mxfp8_gemv tiers are missing from the kernel target"
        );
        let (m, n, k) = (m as usize, n as usize, k as usize);
        ops::mxfp8_gemv_rows_touched(
            ctx.gpu,
            &self.mxfp8_gemv_k,
            twin,
            input,
            output,
            m,
            n,
            k,
            n,
            stream,
        )?;
        Ok(true)
    }

    /// `output[m, n] = input[m, k] · W^T` as the W8A8 GEMM on `weight`'s
    /// MXFP8 operand, quantizing `input` into the arena's FP8 activation
    /// scratch first unless `cast` is false (the previous projection of the
    /// same rows left it there); `false`, nothing launched, without one.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::glm5_kda) fn try_project_w8a8(
        &self,
        input: DevicePtr,
        weight: &Glm5Projection,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        cast: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(w) = weight.w8a8.as_ref() else {
            return Ok(false);
        };
        let (m, n, k) = (m as usize, n as usize, k as usize);
        ensure!(
            w.n == n && w.k == k && self.mx_quant_k.0 != 0,
            "GLM KDA W8A8: operand [{}, {}] for a [{n}, {k}] projection, quant kernel {}",
            w.n,
            w.k,
            self.mx_quant_k.0
        );
        let act = ops::mx_act_scratch(ctx.buffers, m, k)?;
        if cast {
            ops::mx_quant_rows(ctx.gpu, self.mx_quant_k, input, act, 0..m, k, stream)?;
        }
        w.gemm(act, output, m, stream)?;
        Ok(true)
    }

    /// Q, K and V of `m` decode / verify rows in one `mxfp8_gemv_tc*_multi`
    /// launch (32 rows a launch) into the planes of `projected`
    /// (`plane_bytes` apart, rows `p` apart): each CTA runs the body of its
    /// projection's own tier, so every output word equals the three separate
    /// launches'. `false` without twins, or with `ATLAS_GLM_FP4L_KDA_QKV_FUSE=0`.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::glm5_kda) fn try_project_qkv_mxfp8(
        &self,
        normed: DevicePtr,
        projected: DevicePtr,
        plane_bytes: usize,
        m: u32,
        p: usize,
        h: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let w = &self.weights;
        let (Some(q), Some(k), Some(v)) = (
            w.q_proj.mxfp8.as_ref(),
            w.k_proj.mxfp8.as_ref(),
            w.v_proj.mxfp8.as_ref(),
        ) else {
            return Ok(false);
        };
        let kernels = ops::native_kernels(ctx.gpu);
        if !qkv_fuse_enabled() || q.tiled != k.tiled || q.tiled != v.tiled {
            return Ok(false);
        }
        ensure!(
            kernels.mx_multi.complete(q.tiled) && plane_bytes >= m as usize * p * 2,
            "GLM KDA fused MXFP8 QKV: the mxfp8_gemv_tc*_multi tiers are missing \
             (set ATLAS_GLM_FP4L_KDA_QKV_FUSE=0) or the planes overlap"
        );
        let parts = [
            ops::MxPart::mx(q, projected, p),
            ops::MxPart::mx(k, projected.offset(plane_bytes), p),
            ops::MxPart::mx(v, projected.offset(2 * plane_bytes), p),
        ];
        ops::mx_multi_rows(
            ctx.gpu,
            &kernels.mx_multi,
            q.tiled,
            normed,
            &parts,
            m as usize,
            h as usize,
            stream,
        )?;
        Ok(true)
    }

    /// Whether this layer's q/k/v/o read MXFP8 twins.
    pub(in crate::layers::glm5_kda) fn has_mxfp8_projections(&self) -> bool {
        self.weights.q_proj.mxfp8.is_some()
    }
}
