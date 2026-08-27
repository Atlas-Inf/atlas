// SPDX-License-Identifier: AGPL-3.0-only

//! Subsample conv projection step: mel upload → fused conv stack (Wave-4C
//! `gemma_audio_subsample_conv`) → flatten linear on the shared GEMM.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::GemmaAudioEncoder;
use super::{f32_to_bf16_bits, launch_optional};
use crate::layers::ops;
use crate::media::gemma_audio::GemmaAudioInput;
use crate::weight_map::DenseWeight;

impl GemmaAudioEncoder {
    /// Subsample ONE clip: upload its mel features (f32 → BF16 host-side)
    /// and validity mask at the packed offsets, run the fused conv stack
    /// into `buf_conv`, then the flatten `input_proj_linear` GEMM into
    /// `buf_h1` — all at `t_off` full-row offset.
    ///
    /// Wave-4C contract (see `enc_impl/mod.rs`):
    /// `gemma_audio_subsample_conv(features, mask, conv0_w, ln0_w, conv1_w,
    /// ln1_w, out, t_mel, t_out, mel)` — the fused kernel applies the mask
    /// multiplicatively (subsampled `[::2]` between the convs), the two
    /// mean-subtracting LayerNorms + ReLUs, and writes the flattened
    /// `[t_out × flatten]` rows (mel-major within row).
    pub(super) fn subsample_stage(
        &self,
        clip: &GemmaAudioInput,
        m_off: usize,
        t_off: usize,
        t_out: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        // Mel upload: scale-free log-mel values, f32 → BF16.
        let mut bf16 = vec![0u16; clip.n_frames * clip.n_mels];
        for (e, &v) in clip.features.iter().enumerate() {
            bf16[e] = f32_to_bf16_bits(v);
        }
        // SAFETY: `bf16` is a live `vec![u16; n_frames*n_mels]`; byte length
        // derived from the same Vec; every element written by the loop; u16/u8
        // have no invalid bit patterns. Read-only, dies first.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(bf16.as_ptr() as *const u8, bf16.len() * 2) };
        gpu.copy_h2d_async(bytes, self.buf_mel.offset(m_off * clip.n_mels * 2), stream)?;
        // Validity mask upload (u8, `[n_frames]`).
        gpu.copy_h2d_async(&clip.mask, self.buf_mask_mel.offset(m_off), stream)?;
        if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
            // Compare our mel against the HF feature extractor: the 440 Hz
            // fixture's HF mel[0] |x| ≈ 23.79, first8 ≈ [-6.91, 0.12, ...].
            let mut mbuf = vec![0u8; clip.n_mels * 2];
            let _ = gpu.copy_d2h(self.buf_mel.offset(m_off * clip.n_mels * 2), &mut mbuf);
            let mf: Vec<f32> = mbuf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let mn = mf.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!("GAE: mel[0] |x|={mn:.4} first8={:?}", &mf[..8]);
        }
        // Two-stage conv stack: mel → conv1+LN1+ReLU → buf_conv1 → conv2+LN2
        // → flattened [t_out, flatten]. Kernel contracts (gemma_audio_encoder.cu):
        //   conv1: (input, mask, out1, w1, ln1_w, T, T1, eps)
        //          grid (T1,1,1), block (1024,1,1) — T = mel frames, T1 = ceil(T/2)
        //   conv2: (in1, output, w2, ln2_w, T1, T2, eps)
        //          grid (T2,1,1), block (1024,1,1) — T2 = output rows (t_out)
        // The fused single kernel was OOB (read 3 conv1 slices, staged 2).
        let t1 = clip.n_frames.div_ceil(2);
        let m_off_bytes = m_off * clip.n_mels * 2;
        launch_optional(
            gpu,
            self.k_subsample_conv1,
            "gemma_audio_subsample_conv1",
            stream,
            |kl| {
                kl.grid([t1 as u32, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.buf_mel.offset(m_off_bytes))
                    .arg_ptr(self.buf_mask_mel.offset(m_off))
                    .arg_ptr(self.buf_conv1)
                    .arg_ptr(self.subsample_conv0_w)
                    .arg_ptr(self.subsample_ln0_w)
                    .arg_u32(clip.n_frames as u32)
                    .arg_u32(t1 as u32)
                    .arg_f32(self.norm_eps)
            },
        )?;
        launch_optional(
            gpu,
            self.k_subsample_conv2,
            "gemma_audio_subsample_conv2",
            stream,
            |kl| {
                kl.grid([t_out as u32, 1, 1])
                    .block([1024, 1, 1])
                    .arg_ptr(self.buf_conv1)
                    .arg_ptr(self.buf_conv.offset(t_off * self.flatten_dim * 2))
                    .arg_ptr(self.subsample_conv1_w)
                    .arg_ptr(self.subsample_ln1_w)
                    .arg_u32(t1 as u32)
                    .arg_u32(t_out as u32)
                    .arg_f32(self.norm_eps)
            },
        )?;
        // Flatten linear: [t_out, flatten] @ [hidden, flatten]ᵀ → buf_h1
        // [t_out, hidden] (shared GEMM, runs for real).
        ops::dense_gemm(
            gpu,
            self.k_gemm,
            self.buf_conv.offset(t_off * self.flatten_dim * 2),
            &DenseWeight {
                weight: self.subsample_proj_w,
            },
            self.buf_h1.offset(t_off * self.hidden_size * 2),
            t_out as u32,
            self.hidden_size as u32,
            self.flatten_dim as u32,
            stream,
        )
    }
}
