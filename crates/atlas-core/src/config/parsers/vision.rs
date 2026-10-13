// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `config.rs` for file-size budget. Parser for a model family.

#![allow(unused_imports)]

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{ModelConfig, VisionConfig};

pub(crate) fn parse_vision_config(raw: &serde_json::Value) -> Option<VisionConfig> {
    let vc = raw.get("vision_config")?;
    let get_usize = |key: &str| -> usize {
        vc.get(key).and_then(serde_json::Value::as_u64).unwrap_or(0) as usize
    };
    let deepstack_visual_indexes = vc
        .get("deepstack_visual_indexes")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    // Some checkpoints declare the image placeholder token at the TOP
    // level (Qwen3.6: `image_token_id`). Older VL configs embed it under
    // `vision_config`. Read both; fall back to 0 which downstream treats
    // as "use the Qwen3-VL default 151655".
    let image_pad_token_id = raw
        .get("image_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("image_token_id").and_then(serde_json::Value::as_u64))
        .unwrap_or(0) as u32;
    // Same two-location dance as the image token, and the same fallback.
    let video_pad_token_id = raw
        .get("video_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("video_token_id").and_then(serde_json::Value::as_u64))
        .unwrap_or(0) as u32;
    Some(VisionConfig {
        is_glm5_next: false,
        in_channels: 3,
        depth: get_usize("depth"),
        hidden_size: get_usize("hidden_size"),
        num_heads: get_usize("num_heads"),
        patch_size: get_usize("patch_size"),
        temporal_patch_size: get_usize("temporal_patch_size"),
        spatial_merge_size: get_usize("spatial_merge_size"),
        intermediate_size: get_usize("intermediate_size"),
        out_hidden_size: get_usize("out_hidden_size"),
        projection_intermediate_size: 0,
        rms_norm_eps: 1e-6,
        swiglu_limit: 0.0,
        deepstack_visual_indexes,
        image_pad_token_id,
        video_pad_token_id,
        image_start_token_id: 0,
        image_end_token_id: 0,
        video_start_token_id: 0,
        video_end_token_id: 0,
        // Not in config.json — it comes from preprocessor_config.json (or the
        // operator's flag), which this parser does not see. Resolved and
        // installed by the server right after config load, before the encoder
        // is built. `None` here means "not yet resolved", never "unbounded".
        max_pixels: None,
    })
}

/// Parse the NVIDIA GLM-5.3 vision tower.  This is deliberately separate from
/// [`parse_vision_config`]: the generic Qwen-shaped config must never silently
/// select GELU/LayerNorm blocks for a GLM checkpoint.
pub(crate) fn parse_glm5_vision_config(
    raw: &serde_json::Value,
    text_config: &serde_json::Value,
) -> Result<Option<VisionConfig>> {
    let Some(vc) = raw
        .get("vision_config")
        .or_else(|| text_config.get("vision_config"))
    else {
        return Ok(None);
    };
    let get_usize = |key: &str| -> Result<usize> {
        let value = vc
            .get(key)
            .and_then(serde_json::Value::as_u64)
            .with_context(|| format!("glm5_next vision_config missing integer `{key}`"))?;
        ensure!(
            value > 0,
            "glm5_next vision_config `{key}` must be non-zero"
        );
        Ok(value as usize)
    };
    let depth = get_usize("depth")?;
    let hidden_size = get_usize("hidden_size")?;
    let num_heads = get_usize("num_heads")?;
    let patch_size = get_usize("patch_size")?;
    let temporal_patch_size = get_usize("temporal_patch_size")?;
    let spatial_merge_size = get_usize("spatial_merge_size")?;
    let intermediate_size = get_usize("intermediate_size")?;
    let out_hidden_size = get_usize("out_hidden_size")?;
    let projection_intermediate_size = get_usize("projection_intermediate_size")?;
    let in_channels = vc
        .get("in_channels")
        .and_then(serde_json::Value::as_u64)
        .unwrap_or(3) as usize;
    let rms_norm_eps = vc
        .get("rms_norm_eps")
        .or_else(|| text_config.get("rms_norm_eps"))
        .and_then(serde_json::Value::as_f64)
        .context("glm5_next vision_config missing rms_norm_eps")?;
    let swiglu_limit = vc
        .get("swiglu_limit")
        .or_else(|| text_config.get("swiglu_limit"))
        .and_then(serde_json::Value::as_f64)
        .context("glm5_next vision_config missing swiglu_limit")? as f32;
    ensure!(
        rms_norm_eps.is_finite() && rms_norm_eps > 0.0,
        "glm5_next vision rms_norm_eps must be positive"
    );
    ensure!(
        swiglu_limit.is_finite() && swiglu_limit > 0.0,
        "glm5_next vision swiglu_limit must be positive"
    );
    ensure!(
        depth == 24,
        "glm5_next native vision requires depth=24, got {depth}"
    );
    ensure!(
        hidden_size == 1024,
        "glm5_next native vision requires hidden_size=1024, got {hidden_size}"
    );
    ensure!(
        num_heads == 16,
        "glm5_next native vision requires num_heads=16, got {num_heads}"
    );
    ensure!(
        patch_size == 14,
        "glm5_next native vision requires patch_size=14, got {patch_size}"
    );
    ensure!(
        temporal_patch_size == 2,
        "glm5_next native vision requires temporal_patch_size=2, got {temporal_patch_size}"
    );
    ensure!(
        spatial_merge_size == 2,
        "glm5_next native vision requires spatial_merge_size=2, got {spatial_merge_size}"
    );
    ensure!(
        intermediate_size == 4096,
        "glm5_next native vision requires intermediate_size=4096, got {intermediate_size}"
    );
    ensure!(
        out_hidden_size == 4096,
        "glm5_next native vision requires out_hidden_size=4096, got {out_hidden_size}"
    );
    ensure!(
        projection_intermediate_size == 10240,
        "glm5_next native vision requires projection_intermediate_size=10240, got {projection_intermediate_size}"
    );
    ensure!(
        in_channels == 3,
        "glm5_next native vision requires in_channels=3, got {in_channels}"
    );
    let image_pad_token_id = raw
        .get("image_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("image_token_id").and_then(serde_json::Value::as_u64))
        .or_else(|| {
            text_config
                .get("image_token_id")
                .and_then(serde_json::Value::as_u64)
        })
        .context("glm5_next config missing image_token_id")? as u32;
    ensure!(
        image_pad_token_id != 0,
        "glm5_next image_token_id must be non-zero"
    );
    let video_pad_token_id = raw
        .get("video_token_id")
        .and_then(serde_json::Value::as_u64)
        .or_else(|| vc.get("video_token_id").and_then(serde_json::Value::as_u64))
        .or_else(|| {
            text_config
                .get("video_token_id")
                .and_then(serde_json::Value::as_u64)
        })
        .unwrap_or(0) as u32;
    // GLM-5 stores these at the top level alongside image_token_id and
    // video_token_id. Keep the canonical IDs as a compatibility default for
    // older exported configs that omitted the redundant fields; the pad IDs
    // remain required above because they identify the encoder rows.
    let token_id = |key: &str, default: u32| {
        raw.get(key)
            .and_then(serde_json::Value::as_u64)
            .or_else(|| vc.get(key).and_then(serde_json::Value::as_u64))
            .or_else(|| text_config.get(key).and_then(serde_json::Value::as_u64))
            .unwrap_or(default as u64) as u32
    };
    let image_start_token_id = token_id("image_start_token_id", 154_830);
    let image_end_token_id = token_id("image_end_token_id", 154_831);
    let video_start_token_id = token_id("video_start_token_id", 154_832);
    let video_end_token_id = token_id("video_end_token_id", 154_833);
    let deepstack_visual_indexes = vc
        .get("deepstack_visual_indexes")
        .and_then(serde_json::Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(serde_json::Value::as_u64)
                .map(|v| v as usize)
                .collect()
        })
        .unwrap_or_default();
    Ok(Some(VisionConfig {
        is_glm5_next: true,
        in_channels,
        depth,
        hidden_size,
        num_heads,
        patch_size,
        temporal_patch_size,
        spatial_merge_size,
        intermediate_size,
        out_hidden_size,
        projection_intermediate_size,
        rms_norm_eps,
        swiglu_limit,
        deepstack_visual_indexes,
        image_pad_token_id,
        video_pad_token_id,
        image_start_token_id,
        image_end_token_id,
        video_start_token_id,
        video_end_token_id,
        max_pixels: None,
    }))
}
