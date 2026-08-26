// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B embed splice (Wave 2C, audio in Wave 4B): overwrite gemma
//! image/video/audio slot-token positions with the matching rows of the
//! gemma vision/audio towers' packed `buf_out` — a straight per-row copy,
//! because each tower's `embed_*` projection lives INSIDE the encoder.
//! Extracted to its own file so both the chunked (`prefill_b/embed_chunk.rs`)
//! and whole-sequence (`prefill_c.rs`) embed paths share one implementation
//! under the ≤500 LoC cap.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::{Result, bail};
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;

impl TransformerModel {
    /// Overwrite gemma media slot-token rows of `hidden` with rows from the
    /// towers' packed `buf_out`, consumed in encounter order:
    /// - image (258880) / video (258884) slots draw from the SINGLE vision
    ///   encoder `buf_out` (Wave 2C semantics, unchanged);
    /// - audio slots (258881) draw from the audio encoder's OWN `buf_out`
    ///   `[Σvalid, 1536]` with an independent row counter (Wave 4B).
    ///
    /// Boundary tokens (boi/boa 256000 + eoi/eoa) are real vocab embeddings
    /// (the tokenizer expander emits boundary + N×slot + boundary), so only
    /// the slot ids in the middle reach this splice. The Qwen pad-id splice
    /// is untouched; this runs only when gemma patches are pending, and a
    /// model never has both encoders (vision and audio, that is).
    ///
    /// Gating mirrors the Wave-2C contract per kind: a slot whose encoder is
    /// absent from the model errors loudly (PCND — an audio slot reaching a
    /// text-only serve is a configuration bug); a slot whose encoder exists
    /// but was not armed by `prepare_*` falls back to its vocab embedding,
    /// exactly like the unarmed image-slot case.
    pub(in crate::model) fn splice_gemma_media_rows(
        &self,
        tokens: &[u32],
        hidden: DevicePtr,
        hidden_bytes_per_row: usize,
        stream: u64,
    ) -> Result<()> {
        let v_pending = *self.gemma_vision_embed_patches.lock();
        let a_pending = *self.gemma_audio_embed_patches.lock();
        // Per-kind arm: a kind splices only when its prepare call staged rows
        // AND its encoder is installed (unarmed/absent kinds fall back to the
        // vocab embedding, mirroring the Wave-2C `pending == 0 || encoder
        // none` early-return — now applied independently to each kind).
        let v_splice = v_pending > 0 && self.gemma_vision_encoder.is_some();
        let a_splice = a_pending > 0 && self.gemma_audio_encoder.is_some();
        if !v_splice && !a_splice {
            return Ok(());
        }
        // PCND: an installed encoder implies its config — the slot ids live
        // there, so fail fast rather than guess a default. Resolved only for
        // the kinds actually splicing (a vision-only serve never touches the
        // audio config, and vice versa).
        let gv = if v_splice {
            Some(self.config.gemma_vision.as_ref().ok_or_else(|| {
                anyhow::anyhow!(
                    "gemma vision encoder installed but config.gemma_vision is None — \
                         cannot resolve image/video slot token ids"
                )
            })?)
        } else {
            None
        };
        let gve = if v_splice {
            Some(self.gemma_vision_encoder.as_ref().unwrap())
        } else {
            None
        };
        let audio_id = self.config.gemma_audio.as_ref().map(|a| a.audio_token_id);
        let v_row_base = *self.gemma_vision_row_base.lock();
        let v_row_bytes = crate::layers::gemma_vision_encoder::OUT_HIDDEN_SIZE * 2;
        let a_row_base = *self.gemma_audio_row_base.lock();
        let a_row_bytes = crate::layers::gemma_audio_encoder::OUT_HIDDEN_SIZE * 2;
        let mut v_row = 0usize;
        let mut a_row = 0usize;
        let mut v_copied = 0usize;
        let mut a_copied = 0usize;
        for (i, &tok) in tokens.iter().enumerate() {
            if let (Some(gv), Some(gve)) = (gv, gve)
                && (tok == gv.image_token_id || tok == gv.video_token_id)
            {
                let src = gve.buf_out().offset((v_row_base + v_row) * v_row_bytes);
                let dst = hidden.offset(i * hidden_bytes_per_row);
                self.gpu.copy_d2d_async(src, dst, v_row_bytes, stream)?;
                v_row += 1;
                v_copied += 1;
            } else if Some(tok) == audio_id {
                match &self.gemma_audio_encoder {
                    Some(gae) if a_pending > 0 => {
                        let src = gae.buf_out().offset((a_row_base + a_row) * a_row_bytes);
                        let dst = hidden.offset(i * hidden_bytes_per_row);
                        self.gpu.copy_d2d_async(src, dst, a_row_bytes, stream)?;
                        a_row += 1;
                        a_copied += 1;
                    }
                    Some(_) => {
                        // Unarmed audio encoder → vocab embedding stays,
                        // mirroring the unarmed image-slot fallback.
                    }
                    None => bail!(
                        "gemma audio slot token {tok} in the token stream but no audio \
                         encoder is wired; audio media is rejected upstream (Wave 1 gate)"
                    ),
                }
            }
        }
        if std::env::var("ATLAS_DUMP_EMBED").ok().as_deref() == Some("1") {
            self.gpu.synchronize(stream).ok();
            // Read the FIRST copied vision row back and report its norm — a
            // zero buf_out (pool/embed_vision stub gap) means the model sees
            // blank vision slots and "image is missing".
            if v_copied > 0 {
                // Read the FIRST vision buf_out row from the encoder (not
                // `hidden`, whose row 0 is the BOS token) and report its norm.
                let mut buf = vec![0u8; v_row_bytes];
                let _ = self.gpu.copy_d2h(
                    gve.as_ref().unwrap().buf_out().offset(v_row_base * v_row_bytes),
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
                tracing::info!(
                    "ATLAS_GVE_SPLICE: copied {v_copied} vision rows; first buf_out row |x|={n:.4} first5={:?}",
                    &v[..5]
                );
            } else {
                tracing::info!("ATLAS_GVE_SPLICE: 0 vision slot tokens found in chunk");
            }
            // Audio splice diagnostic: report how many audio rows were copied
            // and the norm of the first audio row in buf_out (≈115.92 for the
            // 440 Hz fixture) — a mismatch means the model sees blank/garbage
            // audio slots.
            if let (Some(gae), Some(_)) = (&self.gemma_audio_encoder, audio_id) {
                tracing::info!(
                    "ATLAS_GAE_SPLICE: copied {a_copied} audio rows (pending {a_pending}, \
                     row_base {a_row_base})"
                );
                if a_copied > 0 {
                    let mut buf = vec![0u8; a_row_bytes];
                    let _ = self
                        .gpu
                        .copy_d2h(gae.buf_out().offset(a_row_base * a_row_bytes), &mut buf);
                    let v: Vec<f32> = buf
                        .chunks_exact(2)
                        .map(|c| {
                            let bits = u16::from_le_bytes([c[0], c[1]]);
                            f32::from_bits((bits as u32) << 16)
                        })
                        .collect();
                    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                    tracing::info!(
                        "ATLAS_GAE_SPLICE: first audio buf_out row |x|={n:.4} first5={:?}",
                        &v[..5]
                    );
                }
            }
        }
        Ok(())
    }
}
