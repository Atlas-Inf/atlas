// SPDX-License-Identifier: AGPL-3.0-only

//! [`VisionConfig`] — the vision encoder configuration. Split out of
//! `config.rs` for file-size budget.

/// Vision encoder configuration for Qwen3-VL models.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// True only for the native GLM-5.3 vision tower.  Its RMSNorm/SwiGLU/
    /// convolutional merger is not interchangeable with the Qwen ViT path.
    pub is_glm5_next: bool,
    /// Input channel count for the visual patch embed (GLM uses 3).
    pub in_channels: usize,
    /// Number of ViT transformer blocks (depth=27).
    pub depth: usize,
    /// ViT hidden dimension (1152).
    pub hidden_size: usize,
    /// Number of attention heads (16).
    pub num_heads: usize,
    /// Spatial patch size in pixels (16).
    pub patch_size: usize,
    /// Temporal patch size: still images are replicated this many times (2).
    pub temporal_patch_size: usize,
    /// 2×2 spatial merge: this many patch-lengths merged into one token (2).
    pub spatial_merge_size: usize,
    /// ViT MLP intermediate size (4304).
    pub intermediate_size: usize,
    /// Projection output dimension = LLM hidden_size (2048).
    pub out_hidden_size: usize,
    /// GLM merger gated projection width (10240); zero for Qwen towers.
    pub projection_intermediate_size: usize,
    /// GLM visual RMSNorm epsilon. Qwen uses the model-specific LayerNorm
    /// kernels and leaves this at the default.
    pub rms_norm_eps: f64,
    /// GLM clamped SwiGLU limit. Zero means the Qwen path has no such limit.
    pub swiglu_limit: f32,
    /// Layer indices after which deepstack mergers are applied ([8, 16, 24]).
    pub deepstack_visual_indexes: Vec<usize>,
    /// Placeholder token ID that marks where vision embeddings get spliced
    /// into the text embedding stream. Qwen3-VL uses 151655; Qwen3.6 uses
    /// 248056. When 0 the runtime falls back to the legacy Qwen3-VL value.
    pub image_pad_token_id: u32,
    /// Placeholder token ID for VIDEO frames, the temporal sibling of
    /// [`Self::image_pad_token_id`]. Qwen3.6/3.8 use 248057. A distinct token
    /// is what lets the position builder tell a video item from an image one
    /// in the token stream, which matters because their MRoPE treatment
    /// differs: an image holds T constant across its whole pad run, a video
    /// advances T once per temporal group. When 0 the runtime falls back to
    /// the family default.
    pub video_pad_token_id: u32,
    /// GLM image sequence boundary token IDs. GLM's chat template emits one
    /// image marker triple and the serving processor expands it to one token
    /// per merged patch. Zero for model families whose template owns a
    /// different vision layout.
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    /// GLM video sequence boundary token IDs. A video expands to one image
    /// marker per temporal group, with timestamp tokens between groups.
    pub video_start_token_id: u32,
    pub video_end_token_id: u32,
    /// Resolved vision AREA bound in pixels: the operator's
    /// `--vision-max-pixels`, else the checkpoint's `preprocessor_config.json`,
    /// else `None`.
    ///
    /// ★ THE SINGLE SOURCE OF TRUTH, and it exists because there used to be
    /// two. The CPU preprocessor clamped every image to 1280px on the long
    /// side while the GPU encoder allocated its buffers for 6400 patches —
    /// exactly 1280×1280 — with nothing in the code connecting them. They
    /// agreed only by coincidence, so raising one on 2026-08-14 made every
    /// image above 1280px fail an H2D copy with `CUDA_ERROR_INVALID_VALUE`
    /// from deep inside the scheduler.
    ///
    /// Both now derive from this field, resolved once at config load, before
    /// the encoder is constructed. `None` keeps the historical behaviour on
    /// both sides.
    pub max_pixels: Option<usize>,
}

impl VisionConfig {
    /// Dimension of the merger input (spatial_merge_size² × hidden_size).
    pub fn merger_input_size(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size * self.hidden_size
    }
}
