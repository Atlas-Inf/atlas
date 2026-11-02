// SPDX-License-Identifier: AGPL-3.0-only

//! [`VisionConfig`] — the vision encoder configuration. Split out of
//! `config.rs` for file-size budget.

/// Vision encoder configuration for Qwen3-VL models.
#[derive(Debug, Clone)]
pub struct VisionConfig {
    /// True only for the native GLM-5.3 vision tower.  Its RMSNorm/SwiGLU/
    /// convolutional merger is not interchangeable with the Qwen ViT path.
    pub is_glm5_next: bool,
    /// True only when the checkpoint's video processor is HF's Qwen3VL
    /// (`video_processor_type: Qwen3VLVideoProcessor`, or `processor_class:
    /// Qwen3VLProcessor`). Its prompt nests per-temporal-group timestamp
    /// text and vision start/end markers inside the video's outer pair, and
    /// its MRoPE gives every group its own t=1 plane. Resolved at serve
    /// time from the checkpoint's preprocessor config, not config.json.
    pub is_qwen3vl_video: bool,
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
    /// GLM-5.3's processor token bounds, resolved by the server next to
    /// [`Self::max_pixels`]; other towers ignore them.
    pub token_budgets: VisionTokenBudgets,
    /// The checkpoint's image `min_pixels` (`size.shortest_edge`): HF's Qwen
    /// `smart_resize` scales a smaller still UP to it. `None` when the
    /// checkpoint declares none.
    /// Enntity/atlas 0eef4031.
    pub min_pixels: Option<usize>,
    /// The checkpoint's video processor bounds, as `t x h x w` pixel
    /// volumes over a clip's sampled frames
    /// (`video_preprocessor_config.json` or `processor_config.json`
    /// `[video_processor]`). `None` sizes each frame like a still.
    pub video_pixels: Option<PixelBounds>,

    /// Omni (`nemotron_h_omni`) RADIO-tower parameters. `Some` marks this
    /// vision config as the RADIO preprocessing/encoder family rather
    /// than the Qwen-VL one.
    pub omni: Option<OmniVisionConfig>,
}

/// `(min, max)` vision tokens per still and per video, as GLM-5.3's nested
/// `processor_config.json` declares them (`image_processor` /
/// `video_processor` → `min_image_tokens` / `max_image_tokens`). A vision
/// token is one temporal group of one `merge × merge` patch block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionTokenBudgets {
    pub image: (usize, usize),
    pub video: (usize, usize),
}

impl Default for VisionTokenBudgets {
    /// The processor classes' own defaults, which the shipped checkpoint
    /// repeats: 16..8000 tokens per still, 16..240000 per video.
    fn default() -> Self {
        Self {
            image: (16, 8_000),
            video: (16, 240_000),
        }
    }
}

/// A processor's `size = {shortest_edge, longest_edge}`: pixel counts, not
/// edge lengths, despite the names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PixelBounds {
    pub min_pixels: usize,
    pub max_pixels: usize,
}

impl VisionConfig {
    /// Dimension of the merger input (spatial_merge_size² × hidden_size).
    pub fn merger_input_size(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size * self.hidden_size
    }
}

/// RADIO-tower parameters for `nemotron_h_omni`. Mirrors the keys the
/// checkpoint's own `image_processing_nemotron_h_omni.py` and
/// `modeling_radio.py` read.
#[derive(Debug, Clone, Default)]
pub struct OmniVisionConfig {
    /// CLIP normalization stats (the checkpoint's `input_conditioner`
    /// buffers — never hard-coded).
    pub norm_mean: [f32; 3],
    pub norm_std: [f32; 3],
    /// Patch/token budget bounds for `_compute_target_patches`.
    pub max_num_patches: usize,
    pub min_num_patches: usize,
    /// Pixel-shuffle factor (0.5 → 2×2 merge).
    pub downsample_ratio: f64,
    /// `<image>`(18)/`<img>`(19)/`</img>`(20) wrapper token ids.
    pub image_token_id: u32,
    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    /// `force_image_size` from the config (0 = unset).
    pub force_image_size: usize,
    /// cls+register prefix tokens the embedder prepends / strip count (10).
    pub prefix_tokens: usize,
    pub num_summary_tokens: usize,
    /// ViT positional-embed table side (max_img_size / patch_size).
    pub pos_grid_side: usize,
    /// mlp1 RMSNorm eps.
    pub projector_norm_eps: f64,
    /// ViT layer-norm eps.
    pub layer_norm_eps: f64,
}
