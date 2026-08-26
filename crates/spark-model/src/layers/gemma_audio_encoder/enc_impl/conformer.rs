// SPDX-License-Identifier: AGPL-3.0-only

//! The conformer layer driver: FFN1 (M-agnostic) → attention block
//! (per-clip chunked attention) → light conv (per clip) → FFN2 (M-agnostic)
//! → `norm_out`. `buf_h1` holds the hidden states on entry and exit.
//!
//! Mirrors HF `Gemma4AudioLayer.forward`; the `gradient_clipping` clamps
//! (config 1e10 — beyond BF16 range) are documented no-ops and are NOT
//! launched.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use super::super::{GemmaAudioEncoder, GemmaAudioLayerWeights};
use crate::layers::ops;

impl GemmaAudioEncoder {
    /// One conformer layer over the packed batch (`rows` = Σ full rows at
    /// `t_off[i]`): the FFN/norm stages run once over Σ; the chunked
    /// attention and light conv loop per clip over disjoint slices (their
    /// blocked masks at byte offsets `a_off[i]`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn conformer_layer_batched(
        &self,
        layer_idx: usize,
        blk: &GemmaAudioLayerWeights,
        rows: usize,
        t_i: &[usize],
        t_off: &[usize],
        a_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let rows32 = rows as u32;
        // 1. FFN1 over the packed batch.
        self.ffn_sub_block(&blk.feed_forward1, rows32, gpu, stream)?;
        if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
            gpu.synchronize(stream)?;
            tracing::info!("GAE: L{layer_idx} ffn1 ok");
        }
        // 2. Attention block (norm → chunked attn → norm → residual).
        self.attn_sub_block(
            blk,
            rows,
            t_i,
            t_off,
            a_off,
            self.relative_k[layer_idx],
            self.spd_bufs[layer_idx],
            gpu,
            stream,
        )?;
        if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
            gpu.synchronize(stream)?;
            tracing::info!("GAE: L{layer_idx} attn ok");
        }
        // 3. Light conv per clip (the depthwise conv is causal per clip).
        for (i, &t) in t_i.iter().enumerate() {
            self.light_conv_sub_block(blk, t_off[i], t, gpu, stream)?;
        }
        if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
            gpu.synchronize(stream)?;
            tracing::info!("GAE: L{layer_idx} lconv ok");
        }
        // 4. FFN2 over the packed batch.
        self.ffn_sub_block(&blk.feed_forward2, rows32, gpu, stream)?;
        if std::env::var("ATLAS_AUDIO_TIMING").is_ok() {
            gpu.synchronize(stream)?;
            tracing::info!("GAE: L{layer_idx} ffn2 ok");
        }
        // 5. norm_out closes the layer WITHOUT a residual add.
        ops::rms_norm(
            gpu,
            self.k_rms_norm,
            self.buf_h1,
            &blk.norm_out,
            self.buf_h1,
            rows32,
            h,
            self.norm_eps,
            stream,
        )
    }
}
