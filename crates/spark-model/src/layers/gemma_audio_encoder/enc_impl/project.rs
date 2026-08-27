// SPDX-License-Identifier: AGPL-3.0-only

//! `output_proj` + `embed_audio` projection (INSIDE the encoder, so the
//! downstream splice is a straight copy): Linear hidden → OUT_HIDDEN_SIZE
//! with bias (Wave-4C `gemma_audio_bias_add`) → RMSNorm(OUT_HIDDEN_SIZE,
//! `with_scale=False` → ones weight) → Linear OUT_HIDDEN_SIZE →
//! OUT_HIDDEN_SIZE.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::div_ceil;

use super::super::{GemmaAudioEncoder, OUT_HIDDEN_SIZE};
use super::launch_optional;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl GemmaAudioEncoder {
    /// Project one clip's `rows` FULL hidden rows (at full-row offset
    /// `t_full_off`) into `buf_out` at the same full offset:
    /// `output_proj` (plain Linear + bias — NOT clipped, HF uses `nn.Linear`)
    /// → `embed_audio` RMSNorm(ones) → 1536→1536 GEMM. The caller gathers
    /// the VALID rows into the packed layout afterwards.
    ///
    /// Wave-4C contract: `gemma_audio_bias_add(out, bias, rows, cols)` —
    /// `out[r·cols+c] += bias[c]` (see `enc_impl/mod.rs`).
    pub(super) fn embed_audio_project(
        &self,
        rows: usize,
        t_full_off: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let r = rows as u32;
        let cols = OUT_HIDDEN_SIZE as u32;
        let src = self.buf_h1.offset(t_full_off * self.hidden_size * 2);
        let proj = self.buf_proj.offset(t_full_off * OUT_HIDDEN_SIZE * 2);
        let dst = self.buf_out.offset(t_full_off * OUT_HIDDEN_SIZE * 2);
        if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
            // HF output_proj.weight is [1536, 1024], |W| ≈ 39.2 — if our
            // loaded norm differs, the weight tensor was mapped wrong.
            let w_n = OUT_HIDDEN_SIZE * self.hidden_size;
            let mut wb = vec![0u8; w_n * 2];
            if let Ok(()) = gpu.copy_d2h(self.output_proj_w, &mut wb) {
                let wf: Vec<f32> = wb
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let wn = wf.iter().map(|x| (x * x) as f64).sum::<f64>().sqrt();
                tracing::info!("GAE: output_proj_w |W|={wn:.4}");
            }
        }
        // output_proj: [rows, hidden] → [rows, OUT_HIDDEN_SIZE] (with bias).
        ops::dense_gemm(
            gpu,
            self.k_gemm,
            src,
            &DenseWeight {
                weight: self.output_proj_w,
            },
            proj,
            r,
            cols,
            self.hidden_size as u32,
            stream,
        )?;
        // Row-broadcast bias add: kernel contract (gemma_audio_encoder.cu)
        // is a flat grid (ceil(rows·cols/256),1,1) — one thread per element.
        launch_optional(gpu, self.k_bias_add, "gemma_audio_bias_add", stream, |kl| {
            kl.grid([div_ceil(r * cols, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(proj)
                .arg_ptr(self.output_proj_b)
                .arg_u32(r)
                .arg_u32(cols)
        })?;
        // embed_audio: unweighted RMSNorm then the 1536→1536 projection.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            proj,
            &DenseWeight {
                weight: self.norm_unit_w,
            },
            proj,
            r,
            cols,
            self.norm_eps,
            stream,
        )?;
        ops::dense_gemm(
            gpu,
            self.k_gemm,
            proj,
            &DenseWeight {
                weight: self.embed_audio_proj_w,
            },
            dst,
            r,
            cols,
            cols,
            stream,
        )
    }
}
