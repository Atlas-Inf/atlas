// SPDX-License-Identifier: AGPL-3.0-only

//! FFN sub-block: `pre_feedforward_layernorm` → GeGLU MLP (clipped gate/up
//! linears, fused `gelu_tanh × up`, clipped down linear) →
//! `post_feedforward_layernorm` → residual add.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{GemmaVisionEncoder, GemmaVisionLayerWeights};
use super::launch_optional;
use crate::layers::ops;

impl GemmaVisionEncoder {
    /// Clipped linear: clamp the input in place to `[input_min, input_max]`,
    /// GEMM `[m, k] @ [n, k]ᵀ`, clamp the output to `[output_min, output_max]`.
    /// The input clamp is in-place on `input` — callers guarantee it is
    /// scratch that every linear's GEMM has already consumed.
    ///
    /// Wave 2: the clamps ride the Wave-3 `gemma_vision_clamp` stub (null
    /// handle → no-op); the GEMM runs for real on every target.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn clipped_gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: spark_runtime::gpu::DevicePtr,
        lin: &super::super::ClipLinearWeights,
        output: spark_runtime::gpu::DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        let n_in = m * k;
        launch_optional(gpu, self.k_clamp, "gemma_vision_clamp", stream, |kl| {
            kl.grid([div_ceil(n_in, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(input)
                .arg_f32(lin.input_min)
                .arg_f32(lin.input_max)
                .arg_u32(n_in)
        })?;
        ops::dense_gemm(
            gpu,
            self.k_gemm,
            input,
            &lin.weight,
            output,
            m,
            n,
            k,
            stream,
        )?;
        let n_out = m * n;
        launch_optional(gpu, self.k_clamp, "gemma_vision_clamp", stream, |kl| {
            kl.grid([div_ceil(n_out, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(output)
                .arg_f32(lin.output_min)
                .arg_f32(lin.output_max)
                .arg_u32(n_out)
        })
    }

    /// FFN sub-block over M=Σp rows. `buf_h1` holds the hidden states on
    /// entry; the sub-block ends with the post-FFN residual in `buf_h1`.
    pub(super) fn ffn_sub_block(
        &self,
        blk: &GemmaVisionLayerWeights,
        pt: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let inter = self.intermediate_size as u32;
        let n_h = pt * h;
        // Residual save for the FFN sub-block.
        gpu.copy_d2d(self.buf_h1, self.buf_h2, (n_h * 2) as usize)?;
        // pre_feedforward_layernorm, in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.pre_feedforward_layernorm,
            self.buf_h1,
            pt,
            h,
            self.norm_eps,
            stream,
        )?;
        // gate plane [Σp, inter] + up plane [Σp, inter] in buf_wide.
        let gate = self.buf_wide;
        let up = self.buf_wide.offset((pt * inter * 2) as usize);
        self.clipped_gemm(gpu, self.buf_h1, &blk.gate_proj, gate, pt, inter, h, stream)?;
        self.clipped_gemm(gpu, self.buf_h1, &blk.up_proj, up, pt, inter, h, stream)?;
        // GeGLU activation: gelu_tanh(gate) × up → gate plane (in place;
        // both inputs fully read before the elementwise write).
        let n_gi = pt * inter;
        KernelLaunch::new(gpu, self.k_gelu_mul)
            .grid([div_ceil(n_gi, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_ptr(gate)
            .arg_u32(n_gi)
            .launch(stream)?;
        // down_proj [Σp, inter] → buf_h1 [Σp, hidden] (normed input already
        // consumed by gate/up).
        self.clipped_gemm(gpu, gate, &blk.down_proj, self.buf_h1, pt, h, inter, stream)?;
        // post_feedforward_layernorm + residual add, both in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.post_feedforward_layernorm,
            self.buf_h1,
            pt,
            h,
            self.norm_eps,
            stream,
        )?;
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h)
            .launch(stream)
    }
}
