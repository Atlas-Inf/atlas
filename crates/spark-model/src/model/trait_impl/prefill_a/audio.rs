// SPDX-License-Identifier: AGPL-3.0-only

//! Prefill phase A — gemma-4 E2B audio-embed dispatch (Wave 4B).

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;

impl TransformerModel {
    /// Gemma-4 E2B: encode every audio clip in ONE `forward_batched` call and
    /// stage the packed `[Σvalid, OUT_HIDDEN_SIZE]` BF16 features (clip-order
    /// in the encoder's `buf_out`) + the per-clip valid-token counts for the
    /// splice. No-op without an audio encoder (text-only gemma / other
    /// families).
    ///
    /// Mirror of [`Self::prepare_gemma_media_embed_dispatch`]: the audio
    /// tower's `embed_audio` projection lives INSIDE the encoder, so the
    /// downstream splice is a straight per-row `buf_out` copy — only the row
    /// count and per-clip counts are staged.
    pub(in crate::model) fn prepare_gemma_audio_embed_dispatch(
        &self,
        audios: &[crate::media::gemma_audio::GemmaAudioInput],
    ) -> Result<()> {
        let gae = match &self.gemma_audio_encoder {
            Some(gae) => gae,
            None => return Ok(()),
        };
        // Empty media list: nothing to encode. forward_batched with zero
        // clips would launch conformer kernels on a 0-row grid (CUDA
        // INVALID_VALUE grid=[0,1,1]) — the scheduler calls this with &[]
        // whenever a request carries no audio, so short-circuit here.
        if audios.is_empty() {
            return Ok(());
        }
        let stream = self.gpu.default_stream();
        let per_clip = gae.forward_batched(audios, self.gpu.as_ref(), stream)?;
        let total: usize = per_clip.iter().sum();
        *self.gemma_audio_embed_patches.lock() = total;
        *self.gemma_audio_soft_counts.lock() = per_clip.clone();
        tracing::info!(
            "Gemma audio encoder: {} clips, {} soft tokens encoded",
            audios.len(),
            total
        );
        Ok(())
    }
}
