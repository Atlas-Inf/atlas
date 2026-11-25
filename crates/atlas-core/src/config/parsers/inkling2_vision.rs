// SPDX-License-Identifier: AGPL-3.0-only

//! Inkling 2's `vision_config` (`model_type: inkling2_vision`).
//!
//! A CLIP-style ViT (CLS token, pre-LayerNorm, layer scale) over
//! `native_patch_size` pixel patches, with a learned position table resampled
//! to each image's grid and axial 2D RoPE on Q/K. The adapter concatenates
//! each `adapter_merge_size`² block of patch states and projects it to the
//! text width with two bias-free linears around a GELU. Images are not
//! resized: the processor pads each to a grid of
//! `native_patch_size * adapter_merge_size` pixel tiles, one token per tile.
//! `temporal_patch_size` describes video; a still image's patch is one frame.

use anyhow::{Result, ensure};
use serde_json::Value;

use crate::config::VisionConfig;

/// The processor's placeholder for one image token (`processor_config.json`
/// `image_token`, `<|unused_200054|>`); `config.json` carries no id.
pub const INKLING2_IMAGE_TOKEN: &str = "<|unused_200054|>";
pub const INKLING2_IMAGE_TOKEN_ID: u32 = 200_054;

pub(crate) fn parse_inkling2_vision(
    raw: &Value,
    text_hidden: usize,
) -> Result<Option<VisionConfig>> {
    let Some(vc) = raw.get("vision_config") else {
        return Ok(None);
    };
    let n = |key: &str| -> Result<usize> {
        vc.get(key)
            .and_then(Value::as_u64)
            .map(|v| v as usize)
            .ok_or_else(|| {
                anyhow::anyhow!("inkling2 vision_config.{key} missing or not an unsigned integer")
            })
    };
    let f = |key: &str| vc.get(key).and_then(Value::as_f64);
    let (hidden, heads) = (n("hidden_size")?, n("num_attention_heads")?);
    ensure!(
        heads > 0 && hidden.is_multiple_of(heads) && (hidden / heads).is_multiple_of(4),
        "inkling2 vision head width {hidden}/{heads} must be a multiple of 4 (axial RoPE)"
    );
    ensure!(
        n("text_hidden_size")? == text_hidden,
        "inkling2 vision text_hidden_size must equal the decoder's {text_hidden}"
    );
    ensure!(
        vc.get("projector_bias").and_then(Value::as_bool) == Some(false),
        "inkling2 vision projector_bias is not implemented"
    );
    Ok(Some(VisionConfig {
        is_glm5_next: false,
        is_qwen3vl_video: false,
        is_inkling2: true,
        in_channels: 3,
        depth: n("num_hidden_layers")?,
        hidden_size: hidden,
        num_heads: heads,
        patch_size: n("native_patch_size")?,
        temporal_patch_size: 1,
        spatial_merge_size: n("adapter_merge_size")?,
        intermediate_size: (hidden as f64 * f("mlp_ratio").unwrap_or(4.0)) as usize,
        out_hidden_size: text_hidden,
        projection_intermediate_size: n("projection_hidden_dim")?,
        rms_norm_eps: f("layer_norm_eps").unwrap_or(1e-5),
        swiglu_limit: 0.0,
        deepstack_visual_indexes: Vec::new(),
        image_pad_token_id: INKLING2_IMAGE_TOKEN_ID,
        video_pad_token_id: 0,
        image_start_token_id: 0,
        image_end_token_id: 0,
        video_start_token_id: 0,
        video_end_token_id: 0,
        max_pixels: None,
        token_budgets: Default::default(),
        min_pixels: None,
        video_pixels: None,
        omni: None,
    }))
}
