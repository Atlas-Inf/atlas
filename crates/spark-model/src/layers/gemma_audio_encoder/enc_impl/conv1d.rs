// SPDX-License-Identifier: AGPL-3.0-only

//! Light conv1d sub-block: residual save → `pre_layer_norm` → clipped
//! `linear_start` → GLU (`a·σ(b)` via the shared `sigmoid_gate_mul`) →
//! depthwise causal conv (Wave-4C `gemma_audio_conv1d`) → `conv_norm` →
//! SiLU → clipped `linear_end` → residual add.
//!
//! Mirrors HF `Gemma4AudioLightConv1d.forward`; the `gradient_clipping`
//! clamp (1e10) is a documented no-op, not launched.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{GemmaAudioEncoder, GemmaAudioLayerWeights};
use super::launch_optional;
use crate::layers::ops;

impl GemmaAudioEncoder {
    /// Run one clip's light conv on its disjoint `buf_h1` slice (row offset
    /// `t_off`, `t` rows). `buf_h2` holds the per-clip residual.
    pub(super) fn light_conv_sub_block(
        &self,
        blk: &GemmaAudioLayerWeights,
        t_off: usize,
        t: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let t32 = t as u32;
        let row = self.hidden_size * 2;
        let x = self.buf_h1.offset(t_off * row);
        let res = self.buf_h2.offset(t_off * row);
        let wide = self.buf_wide.offset(t_off * 2 * row);
        let conv_out = self.buf_mlp.offset(t_off * row);
        // Residual save for the light-conv sub-block.
        gpu.copy_d2d(x, res, t * self.hidden_size * 2)?;
        // pre_layer_norm, in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            x,
            &blk.lconv1d.pre_layer_norm,
            x,
            t32,
            h,
            self.norm_eps,
            stream,
        )?;
        // linear_start → buf_wide [t, 2×hidden].
        self.clipped_gemm(
            gpu,
            x,
            &blk.lconv1d.linear_start,
            wide,
            t32,
            2 * h,
            h,
            stream,
        )?;
        // GLU (F.glu): out = a·σ(b) with a = wide[r, 0..h], b = wide[r, h..2h].
        // `wide` is ROW-INTERLEAVED [t, 2×hidden] (linear_start's dense_gemm
        // output), so a flat (a-plane, b-plane) sigmoid_gate_mul would read the
        // gate from the wrong row (offset n_glu*2 elements lands mid-buffer).
        // Apply per row with the exact a/b pointers.
        for r in 0..t {
            let row_bytes = 2 * self.hidden_size * 2;
            let a = wide.offset(r * row_bytes);
            let b = wide.offset(r * row_bytes + self.hidden_size * 2);
            ops::sigmoid_gate_mul(
                gpu,
                self.k_sigmoid_gate,
                a,
                b,
                a,
                self.hidden_size as u32,
                stream,
            )?;
        }
        // Depthwise CAUSAL conv1d: left pad kernel−1,
        // `out[t][c] = Σ_k dw[c][k] × x[t−(kernel−1)+k][c]`.
        // Kernel contract: grid (ceil(hidden/256),1,1) — one thread per
        // channel, each looping the full time axis. `in_stride` = 2×hidden:
        // the input is the row-interleaved GLU output inside buf_wide.
        launch_optional(gpu, self.k_conv1d, "gemma_audio_conv1d", stream, |kl| {
            kl.grid([div_ceil(h, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(wide)
                .arg_ptr(blk.lconv1d.depthwise_conv1d.weight)
                .arg_ptr(conv_out)
                .arg_u32(t32)
                .arg_u32(h)
                .arg_u32(self.conv_kernel as u32)
                .arg_u32(2 * self.hidden_size as u32)
        })?;
        // conv_norm, in place.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            conv_out,
            &blk.lconv1d.conv_norm,
            conv_out,
            t32,
            h,
            self.norm_eps,
            stream,
        )?;
        // SiLU (Wave-4C stub), in place.
        let n_c = t32 * h;
        launch_optional(gpu, self.k_silu, "gemma_audio_silu", stream, |kl| {
            kl.grid([div_ceil(n_c, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(conv_out)
                .arg_u32(n_c)
        })?;
        // linear_end → x (conv staging fully consumed).
        self.clipped_gemm(gpu, conv_out, &blk.lconv1d.linear_end, x, t32, h, h, stream)?;
        // Residual add, in place.
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_c, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(res)
            .arg_u32(n_c)
            .launch(stream)
    }
}
