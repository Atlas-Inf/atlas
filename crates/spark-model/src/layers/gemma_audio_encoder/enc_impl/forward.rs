// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level `GemmaAudioEncoder::forward_batched`: drives the full
//! clip → soft-token pipeline (subsample conv → 12 conformer layers →
//! output_proj → embed_audio) over N clips, packing the final 1536-wide
//! features into `buf_out` in clip order and returning per-clip valid-token
//! counts.
//!
//! Wave 4A: orchestration contract only — per-clip geometry (mel upload,
//! subsampled masks, blocked attention masks) is precomputed on the host;
//! the shared kernels (GEMM / RMSNorm / GLU / scaled-add) run for real; the
//! gemma-specific kernels (subsample conv, q/k scales, chunked attention,
//! light conv, silu, bias add) are Wave-4C stubs that no-op until their PTX
//! lands.

use anyhow::{Result, ensure};
use spark_runtime::gpu::GpuBackend;

use super::super::{GemmaAudioEncoder, OUT_HIDDEN_SIZE};
use crate::media::gemma_audio::{
    GemmaAudioInput, build_blocked_attn_mask, subsample_conv_len, subsample_mask,
};

impl GemmaAudioEncoder {
    /// Batched forward over N clips. Returns per-clip VALID token counts
    /// (clip order, padding dropped); the projected features are packed into
    /// [`Self::buf_out`] `[Σvalid, OUT_HIDDEN_SIZE]` BF16, clip-order, with
    /// [`Self::total_soft_tokens`] set to Σvalid.
    ///
    /// The tower runs over FULL rows (the conv output length
    /// `subsample_conv_len(n_frames)` per clip — padding rows included, like
    /// HF); only the rows the subsampled mask marks valid survive into
    /// `buf_out` (HF strips `audio_features[audio_mask]` before the text
    /// splice). M-agnostic stages (FFNs, norms, qkv/post GEMMs) run ONCE
    /// over the packed batch; per-clip stages (subsample conv, chunked
    /// attention, light conv, blocked masks) loop per clip over its disjoint
    /// slices — the Gemma vision `forward_batched` contract.
    ///
    /// IN-BOUNDS INVARIANT: the packed path requires Σrows ≤ `t_max` (all
    /// row capacities are one-clip); beyond that a per-clip fallback loops
    /// and REFUSES a batch whose Σvalid would overrun `buf_out` (fail fast).
    pub fn forward_batched(
        &self,
        clips: &[GemmaAudioInput],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<usize>> {
        let n = clips.len();
        let chunk = self.chunk_size;
        let ctx = self.context_size;
        // Per-clip geometry: valid counts (returned), full conv rows (worked
        // on), packed row / mel / mask-byte offsets.
        let mut t_valid = Vec::with_capacity(n);
        let mut t_full = Vec::with_capacity(n);
        let mut t_off = Vec::with_capacity(n);
        let mut m_off = Vec::with_capacity(n);
        let mut a_off = Vec::with_capacity(n);
        let (mut rows_total, mut mel_total, mut mask_bytes) = (0usize, 0usize, 0usize);
        for c in clips {
            ensure!(
                c.n_mels == self.mel_bins,
                "gemma audio: {n_mels} mel bins, expected {mel} (mel_bins)",
                n_mels = c.n_mels,
                mel = self.mel_bins
            );
            ensure!(c.n_frames >= 1, "gemma audio: clip with no frames");
            ensure!(
                c.mask.len() == c.n_frames,
                "gemma audio: {m} mask bytes for {f} frames",
                m = c.mask.len(),
                f = c.n_frames
            );
            ensure!(
                c.features.len() == c.n_frames * c.n_mels,
                "gemma audio: {} feature floats for {}×{} frames×mels",
                c.features.len(),
                c.n_frames,
                c.n_mels
            );
            let t_out = subsample_conv_len(c.n_frames);
            ensure!(
                t_out <= self.t_max,
                "gemma audio: {t_out} rows from {} frames exceeds the {}-row clip cap",
                c.n_frames,
                self.t_max
            );
            let valid = subsample_mask(&c.mask);
            let count = valid.iter().filter(|&&v| v == 1).count();
            ensure!(
                count >= 1,
                "gemma audio: clip with zero valid tokens after 4× subsampling"
            );
            t_off.push(rows_total);
            m_off.push(mel_total);
            a_off.push(mask_bytes);
            t_valid.push(count);
            t_full.push(t_out);
            rows_total += t_out;
            mel_total += c.n_frames;
            mask_bytes += t_out.div_ceil(chunk) * chunk * ctx;
        }

        if rows_total > self.t_max {
            return self.forward_oversized_fallback(clips, &t_valid, &t_full, gpu, stream);
        }

        // 1. Per-clip host prep: blocked attention masks (the mel + frame
        //    masks upload inside `subsample_stage`).
        for (i, c) in clips.iter().enumerate() {
            let valid = subsample_mask(&c.mask);
            let (mask_bytes, _nb) =
                build_blocked_attn_mask(&valid, chunk, self.max_past, self.max_future);
            gpu.copy_h2d(&mask_bytes, self.buf_mask_attn.offset(a_off[i]))?;
        }

        // 2. Subsample conv projection per clip → packed buf_h1 rows.
        for (i, c) in clips.iter().enumerate() {
            self.subsample_stage(c, m_off[i], t_off[i], t_full[i], gpu, stream)?;
            if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
                gpu.synchronize(stream)?;
                tracing::info!("GAE: subsample clip {i} ok");
                // Pre-conformer stage (post input_proj_linear): HF oracle
                // |x| ≈ 253.05 for the 440 Hz fixture — isolates subsample
                // vs conformer drift.
                let mut buf = vec![0u8; self.hidden_size * 2];
                let _ = gpu.copy_d2h(
                    self.buf_h1.offset(t_off[i] * self.hidden_size * 2),
                    &mut buf,
                );
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!("GAE: subsample_out[0] |x|={n:.4} first5={:?}", &v[..5]);
            }
        }

        // 3. Conformer layers over the packed batch (per-clip attention +
        //    light conv inside).
        for (idx, blk) in self.layers.iter().enumerate() {
            self.conformer_layer_batched(
                idx, blk, rows_total, &t_full, &t_off, &a_off, gpu, stream,
            )?;
            if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
                gpu.synchronize(stream)?;
                tracing::info!("GAE: conformer layer {idx} ok");
                if true {
                    // every layer for calibration
                    let mut buf = vec![0u8; self.hidden_size * 2];
                    let _ = gpu.copy_d2h(self.buf_h1, &mut buf);
                    let v: Vec<f32> = buf
                        .chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect();
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    tracing::info!("GAE: L{idx} out[0] |x|={n:.4} first5={:?}", &v[..5]);
                }
            }
        }

        // 4. output_proj + embed_audio per clip → buf_out at FULL offsets,
        //    then gather the VALID rows into the packed layout (dst ≤ src
        //    always: full offsets dominate valid offsets, so ascending d2d
        //    copies never clobber a not-yet-copied source).
        let row = OUT_HIDDEN_SIZE * 2;
        let mut v_off = 0usize;
        for (i, c) in clips.iter().enumerate() {
            self.embed_audio_project(t_full[i], t_off[i], gpu, stream)?;
            if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
                gpu.synchronize(stream).ok();
                // output_proj+bias stage (pre-embed_audio): HF |x| ≈ 300.4
                // for the 440 Hz fixture — isolates conformer vs embed drift.
                let mut buf = vec![0u8; OUT_HIDDEN_SIZE * 2];
                let _ = gpu.copy_d2h(
                    self.buf_proj.offset(t_off[i] * OUT_HIDDEN_SIZE * 2),
                    &mut buf,
                );
                let v: Vec<f32> = buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                tracing::info!("GAE: output_proj[0] |x|={n:.4} first5={:?}", &v[..5]);
            }
            let valid = subsample_mask(&c.mask);
            for (j, &v) in valid.iter().enumerate() {
                if v == 1 {
                    gpu.copy_d2d(
                        self.buf_out.offset((t_off[i] + j) * row),
                        self.buf_out.offset(v_off * row),
                        row,
                    )?;
                    v_off += 1;
                }
            }
        }

        let total: usize = t_valid.iter().sum();
        self.total_soft_tokens
            .store(total, std::sync::atomic::Ordering::Relaxed);
        if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
            gpu.synchronize(stream).ok();
            // Compare the FIRST packed buf_out row against the HF oracle
            // (embed_audio[0] |x| ≈ 300.4 for the 440 Hz 2 s sine fixture).
            let mut buf = vec![0u8; OUT_HIDDEN_SIZE * 2];
            let _ = gpu.copy_d2h(self.buf_out, &mut buf);
            let v: Vec<f32> = buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            tracing::info!(
                "GAE: buf_out[0] |x|={n:.4} first5={:?} valid_rows={total}",
                &v[..5]
            );
        }
        Ok(t_valid)
    }

    /// Fallback for Σrows > t_max: encode each clip ALONE (full single-clip
    /// kernel sequence at offset 0) writing its valid rows into the packed
    /// `buf_out` at the running valid offset. Refuses a batch whose Σvalid
    /// exceeds the `buf_out` row capacity rather than overflowing it.
    #[allow(clippy::too_many_arguments)]
    fn forward_oversized_fallback(
        &self,
        clips: &[GemmaAudioInput],
        t_valid: &[usize],
        t_full: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<Vec<usize>> {
        let s_total: usize = t_valid.iter().sum();
        ensure!(
            s_total <= self.t_max,
            "gemma audio: Σvalid {s_total} > buf_out rows {} — oversized batch refused \
             (the splice wave sizes buf_out per request)",
            self.t_max
        );
        let row = OUT_HIDDEN_SIZE * 2;
        let mut v_off = 0usize;
        for (i, c) in clips.iter().enumerate() {
            let valid = subsample_mask(&c.mask);
            let (mask_bytes, _nb) =
                build_blocked_attn_mask(&valid, self.chunk_size, self.max_past, self.max_future);
            gpu.copy_h2d(&mask_bytes, self.buf_mask_attn)?;
            self.subsample_stage(c, 0, 0, t_full[i], gpu, stream)?;
            for (idx, blk) in self.layers.iter().enumerate() {
                self.conformer_layer_batched(
                    idx,
                    blk,
                    t_full[i],
                    &[t_full[i]],
                    &[0],
                    &[0],
                    gpu,
                    stream,
                )?;
            }
            self.embed_audio_project(t_full[i], 0, gpu, stream)?;
            for (j, &v) in valid.iter().enumerate() {
                if v == 1 {
                    // Source rows sit at offset 0 (per-clip forward); pack the
                    // valid ones into the running global row offset.
                    gpu.copy_d2d(
                        self.buf_out.offset(j * row),
                        self.buf_out.offset(v_off * row),
                        row,
                    )?;
                    v_off += 1;
                }
            }
        }
        self.total_soft_tokens
            .store(s_total, std::sync::atomic::Ordering::Relaxed);
        Ok(t_valid.to_vec())
    }
}
