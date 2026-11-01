// SPDX-License-Identifier: AGPL-3.0-only

//! A tiny GLM visual tower on any (mock) backend, for tests that need a real
//! encoder: two blocks with the checkpoint's head width and patch geometry,
//! tiny hidden sizes, capped at 16x16 input patches (64 merged rows).

use atlas_core::config::VisionConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::{GlmVisionBlockWeights, GlmVisionEncoder, GlmVisionMergerWeights, GlmVisionWeights};
use crate::VisionItem;

pub(crate) const PATCH: usize = 14;
pub(crate) const TEMPORAL: usize = 2;
pub(crate) const PATCH_DIM: usize = 3 * TEMPORAL * PATCH * PATCH;
pub(crate) const HEADS: usize = 2;

pub(crate) fn config() -> VisionConfig {
    VisionConfig {
        is_qwen3vl_video: false,
        is_glm5_next: true,
        in_channels: 3,
        depth: 2,
        hidden_size: 64 * HEADS,
        num_heads: HEADS,
        patch_size: PATCH,
        temporal_patch_size: TEMPORAL,
        spatial_merge_size: 2,
        intermediate_size: 96,
        out_hidden_size: 32,
        projection_intermediate_size: 48,
        rms_norm_eps: 1e-5,
        swiglu_limit: 10.0,
        deepstack_visual_indexes: Vec::new(),
        image_pad_token_id: 0,
        video_pad_token_id: 0,
        image_start_token_id: 0,
        image_end_token_id: 0,
        video_start_token_id: 0,
        video_end_token_id: 0,
        max_pixels: Some(256 * TEMPORAL * PATCH * PATCH),
        token_budgets: Default::default(),
        min_pixels: None,
        video_pixels: None,
        omni: None,
    }
}

/// Placeholder weights: the mock backend never reads them.
pub(crate) fn weights() -> GlmVisionWeights {
    let config = config();
    let w = DevicePtr(0x1000);
    let block = GlmVisionBlockWeights {
        norm1_w: w,
        qkv_w: w,
        qkv_b: w,
        q_norm_w: w,
        k_norm_w: w,
        proj_w: w,
        proj_b: w,
        norm2_w: w,
        gate_up_w: w,
        gate_up_b: w,
        down_w: w,
        down_b: w,
    };
    GlmVisionWeights {
        patch_embed_w: w,
        patch_embed_b: w,
        blocks: vec![block; config.depth],
        post_layernorm_w: w,
        downsample_w: w,
        downsample_b: w,
        merger: GlmVisionMergerWeights {
            proj_w: w,
            post_norm_w: w,
            post_norm_b: w,
            gate_up_w: w,
            down_w: w,
        },
    }
}

/// The tower itself.
pub(crate) fn encoder(gpu: &dyn GpuBackend) -> GlmVisionEncoder {
    let encoder = GlmVisionEncoder::new(weights(), &config(), gpu).unwrap();
    assert_eq!(encoder.p_max, 256);
    encoder
}

/// A `grid` x `grid`-patch still image (zero pixels).
pub(crate) fn image(grid: usize) -> VisionItem {
    VisionItem::image(vec![0.0; grid * grid * PATCH_DIM], grid, grid)
}
