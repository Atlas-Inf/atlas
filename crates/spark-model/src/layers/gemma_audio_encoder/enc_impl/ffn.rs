// SPDX-License-Identifier: AGPL-3.0-only

//! FFN sub-block: residual save → `pre_layer_norm` → clipped `ffw_layer_1`
//! → SiLU (Wave-4C `gemma_audio_silu`) → clipped `ffw_layer_2` →
//! `post_layer_norm` → `residual + residual_weight·normed` blend (shared
//! `bf16_scaled_add`).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::div_ceil;

use super::super::{GemmaAudioEncoder, GemmaAudioFfnWeights};
use super::launch_optional;
use crate::layers::gemma_vision_encoder::ClipLinearWeights;
use crate::layers::ops;

impl GemmaAudioEncoder {
    /// Clipped linear: clamp the input in place to `[input_min, input_max]`,
    /// GEMM `[m, k] @ [n, k]ᵀ`, clamp the output to `[output_min,
    /// output_max]`. The input clamp is in-place on `input` — callers
    /// guarantee it is scratch that every linear's GEMM has already consumed.
    ///
    /// Wave 4A: the clamps ride the Wave-3 `gemma_vision::gemma_vision_clamp`
    /// stub (REUSED — same ClippableLinear class, same `(buf, lo, hi, n)`
    /// contract; null handle → no-op); the GEMM runs for real on every
    /// target. NOTE: `output_proj` is a plain `nn.Linear` in HF and does NOT
    /// go through this path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn clipped_gemm(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        lin: &ClipLinearWeights,
        output: DevicePtr,
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

    /// FFN sub-block over `rows` rows (a packed slice or a whole clip).
    /// `buf_h1` holds the hidden states on entry; the sub-block ends with the
    /// post-blend residual in `buf_h1`.
    ///
    /// HF `Gemma4AudioFeedForward.forward`: residual = x; clamp (no-op at
    /// gradient_clipping 1e10 — documented, not launched); pre_layer_norm;
    /// ffw_layer_1; silu; ffw_layer_2; clamp (no-op); post_layer_norm;
    /// × residual_weight; += residual.
    pub(super) fn ffn_sub_block(
        &self,
        blk: &GemmaAudioFfnWeights,
        rows: u32,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let inter = self.intermediate_size as u32;
        let n_h = rows * h;
        // Residual save for the FFN sub-block.
        gpu.copy_d2d(self.buf_h1, self.buf_h2, (n_h * 2) as usize)?;
        // pre_layer_norm, in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.pre_layer_norm,
            self.buf_h1,
            rows,
            h,
            self.norm_eps,
            stream,
        )?;
        // ffw_layer_1 [rows, inter] ← buf_h1 (normed input).
        self.clipped_gemm(
            gpu,
            self.buf_h1,
            &blk.ffw_layer_1,
            self.buf_ffn,
            rows,
            inter,
            h,
            stream,
        )?;
        // SiLU in place (Wave-4C gemma_audio_silu stub; the shared tree only
        // has the gated silu_mul_separate — this FFN is un-gated).
        let n_gi = rows * inter;
        launch_optional(gpu, self.k_silu, "gemma_audio_silu", stream, |kl| {
            kl.grid([div_ceil(n_gi, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_ffn)
                .arg_u32(n_gi)
        })?;
        // ffw_layer_2 [rows, h] → buf_h1 (normed input already consumed).
        self.clipped_gemm(
            gpu,
            self.buf_ffn,
            &blk.ffw_layer_2,
            self.buf_h1,
            rows,
            h,
            inter,
            stream,
        )?;
        // post_layer_norm, in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.post_layer_norm,
            self.buf_h1,
            rows,
            h,
            self.norm_eps,
            stream,
        )?;
        // Blend: buf_h2 = residual + residual_weight × normed (scaled_add).
        ops::scaled_add(
            gpu,
            self.k_scaled_add,
            self.buf_h2,
            self.buf_h1,
            self.residual_weight,
            n_h,
            stream,
        )?;
        // Restore the buf_h1 invariant.
        gpu.copy_d2d(self.buf_h2, self.buf_h1, (n_h * 2) as usize)
    }
}
