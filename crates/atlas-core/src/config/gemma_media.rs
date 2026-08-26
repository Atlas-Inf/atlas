// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Gemma-4 E2B vision and
//! audio tower configurations.

/// Vision encoder configuration for Gemma-4 E2B (`vision_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaVisionConfig {
    /// ViT hidden dimension.
    pub hidden_size: usize,
    /// ViT MLP intermediate size.
    pub intermediate_size: usize,
    /// Number of ViT transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Spatial patch size in pixels.
    pub patch_size: usize,
    /// Patch pooling kernel size.
    pub pooling_kernel_size: usize,
    /// Positional embedding table size.
    pub position_embedding_size: usize,
    /// Clip linear projections (use_clipped_linears).
    pub use_clipped_linears: bool,
    /// Top-level `image_token_id` that splices vision embeddings into text.
    pub image_token_id: u32,
    /// RoPE theta for the vision tower (`vision_config.rope_parameters.rope_theta`).
    pub rope_theta: f32,
    /// Patch-pooled patches per image: `max_soft_tokens` × pooling_kernel_size².
    pub max_patches: usize,
    /// Processor token budget: soft patches per image/frame before pooling.
    pub max_soft_tokens: usize,
    /// Position table shape (image/frame slots, position slots, hidden).
    pub position_table_shape: (usize, usize, usize),
    /// RMS norm epsilon for the vision tower.
    pub norm_eps: f32,
    /// Frames per video clip.
    pub video_frames: usize,
    /// Soft tokens per video frame.
    pub video_soft_tokens_per_frame: usize,
    /// Top-level `video_token_id` that splices video embeddings into text.
    pub video_token_id: u32,
    /// Top-level `<|image|>` begin-of-image token.
    pub boi_token_id: u32,
    /// Top-level `<image|>` end-of-image token.
    pub eoi_token_id: u32,
}

/// Audio encoder configuration for Gemma-4 E2B (`audio_config`).
#[derive(Debug, Clone, PartialEq)]
pub struct GemmaAudioConfig {
    /// Audio encoder hidden dimension.
    pub hidden_size: usize,
    /// Number of audio transformer blocks.
    pub num_hidden_layers: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Subsampling conv channel counts, outermost to innermost.
    pub subsampling_conv_channels: Vec<usize>,
    /// Subsampling conv kernel size.
    pub conv_kernel_size: usize,
    /// Attention chunk size.
    pub attention_chunk_size: usize,
    /// Left context of the chunked attention window.
    pub attention_context_left: usize,
    /// Right context of the chunked attention window.
    pub attention_context_right: usize,
    /// Output projection dimension (into text hidden_size).
    pub output_proj_dims: usize,
    /// Residual blend weight of the audio branch.
    pub residual_weight: f64,
    /// Clip linear projections (use_clipped_linears).
    pub use_clipped_linears: bool,
    /// Top-level `audio_token_id` that splices audio embeddings into text.
    pub audio_token_id: u32,
    /// Mel filterbank bin count.
    pub mel_bins: usize,
    /// STFT frame length in samples.
    pub frame_length: usize,
    /// STFT hop length in samples.
    pub hop_length: usize,
    /// FFT size for the STFT.
    pub fft_size: usize,
    /// Floor for the mel spectrogram.
    pub mel_floor: f64,
    /// Mel scale (`"htk"` or `"slaney"`).
    pub mel_scale: String,
    /// Audio token budget per clip (`audio_seq_length`).
    pub token_cap: usize,
    /// RMS norm epsilon for the audio tower.
    pub norm_eps: f32,
    /// Audio encoder activation.
    pub activation: String,
    /// Top-level `<|audio|>` begin-of-audio token.
    pub boa_token_id: u32,
    /// Top-level `<audio|>` end-of-audio token.
    pub eoa_token_id: u32,
}
