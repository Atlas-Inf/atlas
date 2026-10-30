// SPDX-License-Identifier: AGPL-3.0-only

//! GLM dense MLP (layers 0..2) on FP4-Local's own MXFP8 bytes
//! (`ATLAS_GLM_FP4L_DENSE_MXFP8`, loaded by `weight_loader::glm5::components`).
//!
//! The layer keeps the checkpoint's MXFP8 gate / up `[inter, h]` and down
//! `[h, inter]` (replicated, as the NVFP4 copies were) and no NVFP4 copy at
//! all. Every entry (`forward`, K2 / K3 / batch-M through
//! `try_glm_canonical_rows`, prefill) lands in [`DenseFfnLayer::try_native`]:
//!
//! - up to 64 rows: gate + up in one `mxfp8_gemv_tc*_multi` launch, SiLU-mul,
//!   down on `mxfp8_gemv_tc*` (`mxfp8_gemv_rows`): exact weights, row bits
//!   independent of the launch width;
//! - longer: `ATLAS_GLM_FP4L_FP8_PREFILL=w8a8` (default) runs the cuBLASLt
//!   MXFP8 W8A8 GEMM (`ops::MxW8a8`, NVIDIA's MXFP8 recipe: activations to
//!   E4M3 per 32 values) in row chunks that fit the FP8 scratch; `w8a16` /
//!   `bf16` expand each projection exactly to BF16 in `expert_down_out` (as
//!   `ATLAS_GLM_DENSE_PREFILL_BF16=2` dequantizes NVFP4 per call) and run the
//!   BF16 GEMM: exact weights on BF16 activations.
//!
//! `ATLAS_GLM_DENSE_TP_SPLIT=1` splits the GEMV rows over the pair
//! (`dense_ffn_tp_split.rs`): half of these projections per rank.
//!
//! Decode bytes per rank per step (3 layers x 151 M params): 467 MB at 1.031
//! B/param against NVFP4's 255 MB; resident +~225 MB.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{DenseFfnLayer, FfnActivation};
use crate::layer::ForwardContext;
use crate::layers::ops::{self, MxPart, MxW8a8, native_kernels};
use crate::weight_map::Mxfp8Weight;

/// Rows up to which the native dense MLP runs the GEMV tiers.
pub(crate) const DENSE_NATIVE_GEMV_ROWS: usize = 64;

/// A layer's native dense MLP: the checkpoint's MXFP8 projections, and their
/// W8A8 operands when prefill runs W8A8.
#[derive(Debug, Clone, Copy)]
pub struct DenseNative {
    pub gate: Mxfp8Weight,
    pub up: Mxfp8Weight,
    pub down: Mxfp8Weight,
    pub w8a8: Option<[MxW8a8; 3]>,
}

impl DenseNative {
    /// Free the twins and the W8A8 operands' scale images (the operands
    /// borrow the twins' E4M3 bytes).
    pub fn free(self, gpu: &dyn GpuBackend) -> Result<()> {
        if let Some(ops) = self.w8a8 {
            for w in ops {
                w.free(gpu)?;
            }
        }
        for t in [self.gate, self.up, self.down] {
            gpu.free(t.data)?;
            gpu.free(t.scales)?;
        }
        Ok(())
    }
}

/// Rows per W8A8 GEMM so a `[rows, k]` MXFP8 activation image fits the
/// arena's FP8 scratch: a multiple of the 128-row scale tile.
fn w8a8_chunk_rows(q_cap: usize, sf_cap: usize, k: usize) -> usize {
    let mut rows = (q_cap / k) / 128 * 128;
    while rows > 0 && ops::mx_lt_scale_bytes(rows, k) > sf_cap {
        rows -= 128;
    }
    rows
}

impl DenseFfnLayer {
    /// Install the native MXFP8 projections (the layer's NVFP4 weights are
    /// null placeholders then).
    pub fn install_native(&mut self, native: DenseNative) -> Result<()> {
        ensure!(
            self.activation == FfnActivation::SiLU && self.lora.is_none(),
            "native dense MLP supports unadapted SiLU only"
        );
        self.native = Some(native);
        Ok(())
    }

    /// Whether this layer's projections are native (`dense_ffn_native.rs`).
    pub fn has_native(&self) -> bool {
        self.native.is_some()
    }

    /// The SiLU MLP over `rows` rows of `input` into `moe_output` on the
    /// native projections; `false`, nothing launched, without them.
    pub(super) fn try_native(
        &self,
        input: DevicePtr,
        rows: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(native) = self.native else {
            return Ok(false);
        };
        let (h, inter) = (ctx.config.hidden_size, ctx.config.intermediate_size);
        let (gate, up, out) = (
            ctx.buffers.expert_gate_out(),
            ctx.buffers.expert_up_out(),
            ctx.buffers.moe_output(),
        );
        let kernels = native_kernels(ctx.gpu);
        let gemv = rows <= DENSE_NATIVE_GEMV_ROWS;
        let proj = super::tp_split::SplitProj::Mxfp8(&native);
        if gemv && self.glm_verify_rows_split(input, rows as u32, proj, ctx, stream)? {
            return Ok(true);
        }
        if gemv {
            ops::mx_multi_rows(
                ctx.gpu,
                &kernels.mx_multi,
                false,
                input,
                &[
                    MxPart::mx(&native.gate, gate, inter),
                    MxPart::mx(&native.up, up, inter),
                ],
                rows,
                h,
                stream,
            )?;
        } else {
            self.native_prefill_gemm(&native, 0, input, gate, rows, inter, h, ctx, stream)?;
            self.native_prefill_gemm(&native, 1, input, up, rows, inter, h, ctx, stream)?;
        }
        ops::silu_mul(
            ctx.gpu,
            self.act_mul,
            gate,
            up,
            gate,
            (rows * inter) as u32,
            stream,
        )?;
        if gemv {
            ops::mxfp8_gemv_rows_touched(
                ctx.gpu,
                &kernels.mx.row_major,
                &native.down,
                gate,
                out,
                rows,
                h,
                inter,
                h,
                stream,
            )?;
        } else {
            self.native_prefill_gemm(&native, 2, gate, out, rows, h, inter, ctx, stream)?;
        }
        Ok(true)
    }

    /// Prefill projection `i` (gate, up, down) of `m` rows: the W8A8 GEMM in
    /// row chunks, or the exact BF16 expansion and the BF16 GEMM.
    #[allow(clippy::too_many_arguments)]
    fn native_prefill_gemm(
        &self,
        native: &DenseNative,
        i: usize,
        input: DevicePtr,
        output: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if let Some(ops) = native.w8a8 {
            let w = &ops[i];
            let quant = native_kernels(ctx.gpu).mx_quant;
            ensure!(
                w.n == n && w.k == k && quant.0 != 0,
                "native dense W8A8: operand [{}, {}] for [{n}, {k}]",
                w.n,
                w.k
            );
            let chunk = w8a8_chunk_rows(
                ctx.buffers.fp8_act_bytes(),
                ctx.buffers.sizes().fp8_act_scale,
                k,
            );
            ensure!(
                chunk > 0,
                "native dense W8A8: no 128-row [_, {k}] tile fits the FP8 scratch"
            );
            for first in (0..m).step_by(chunk) {
                let rows = (m - first).min(chunk);
                let act = ops::mx_act_scratch(ctx.buffers, rows, k)?;
                ops::mx_quant_rows(
                    ctx.gpu,
                    quant,
                    input.offset(first * k * 2),
                    act,
                    0..rows,
                    k,
                    stream,
                )?;
                w.gemm(act, output.offset(first * n * 2), rows, stream)?;
            }
            return Ok(());
        }
        ensure!(
            ctx.dispatch.cublas_gemm,
            "native dense BF16 prefill requires ATLAS_CUBLAS_GEMM=1"
        );
        let twin = [&native.gate, &native.up, &native.down][i];
        let scratch = ctx.buffers.expert_down_out();
        ensure!(
            ctx.buffers.sizes().expert_down_out >= n * k * 2,
            "native dense BF16 expansion exceeds expert_down_out"
        );
        ops::mxfp8_expand_bf16(
            ctx.gpu,
            native_kernels(ctx.gpu).mx_expand,
            twin,
            scratch,
            n,
            k,
            stream,
        )?;
        ops::cublas_bf16_proj_dense(input, scratch, output, m as u32, n as u32, k as u32, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn w8a8_chunks_fit_the_scratch_on_the_128_row_grid() {
        // GLM down (k = 12288) in a scratch sized for 4096 rows of k = 8192.
        let (q, sf) = (4096 * 8192, 4096 * (8192 / 32));
        let rows = w8a8_chunk_rows(q, sf, 12288);
        assert!(rows > 0 && rows.is_multiple_of(128));
        assert!(rows * 12288 <= q && ops::mx_lt_scale_bytes(rows, 12288) <= sf);
        assert!((rows + 128) * 12288 > q || ops::mx_lt_scale_bytes(rows + 128, 12288) > sf);
        assert_eq!(w8a8_chunk_rows(100 * 12288, sf, 12288), 0);
    }
}
