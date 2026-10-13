// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM-5.3 visual tower state.
//!
//! GLM's visual weights are dense BF16 even when the language model is NVFP4.
//! This module keeps that tower separate from the Qwen ViT implementation:
//! the two encoders do not share block, merger, or attention semantics.

use anyhow::{Result, ensure};
use atlas_core::config::VisionConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::vision_preprocess::derive_glm_max_patches;

mod forward;
#[cfg(test)]
#[path = "glm/forward_tests.rs"]
mod forward_tests;
mod ops;

/// Queries per flash-attention block (four warps of 16 rows).
const FLASH_QUERY_ROWS: u32 = 64;
/// Head width the flash-attention kernel is compiled for.
const FLASH_HEAD_DIM: usize = 64;

/// Merged rows `buf_out` holds for one request. Every item is encoded as its
/// own attention sequence (at most `p_max` input patches), but the merged rows
/// of all items are packed back to back and spliced from this one buffer, so
/// a request may carry several full-size images: 16384 rows is eleven
/// 1024x1024 images at 1369 rows each (128 MiB at `out_hidden_size` 4096).
pub(crate) const GLM_MAX_OUTPUT_ROWS: usize = 16_384;

#[derive(Clone, Copy)]
pub(crate) struct GlmVisionBlockWeights {
    pub norm1_w: DevicePtr,
    pub qkv_w: DevicePtr,
    pub qkv_b: DevicePtr,
    pub q_norm_w: DevicePtr,
    pub k_norm_w: DevicePtr,
    pub proj_w: DevicePtr,
    pub proj_b: DevicePtr,
    pub norm2_w: DevicePtr,
    pub gate_up_w: DevicePtr,
    pub gate_up_b: DevicePtr,
    pub down_w: DevicePtr,
    pub down_b: DevicePtr,
}

#[derive(Clone, Copy)]
pub(crate) struct GlmVisionMergerWeights {
    pub proj_w: DevicePtr,
    pub post_norm_w: DevicePtr,
    pub post_norm_b: DevicePtr,
    pub gate_up_w: DevicePtr,
    pub down_w: DevicePtr,
}

pub(crate) struct GlmVisionWeights {
    pub patch_embed_w: DevicePtr,
    pub patch_embed_b: DevicePtr,
    pub blocks: Vec<GlmVisionBlockWeights>,
    pub post_layernorm_w: DevicePtr,
    pub downsample_w: DevicePtr,
    pub downsample_b: DevicePtr,
    pub merger: GlmVisionMergerWeights,
}

pub(crate) struct GlmVisionEncoder {
    pub buf_out: DevicePtr,
    pub out_hidden_size: usize,
    pub p_max: usize,
    /// Row capacity of `buf_out`; see [`GLM_MAX_OUTPUT_ROWS`].
    pub out_rows: usize,
    patch_embed_w: DevicePtr,
    patch_embed_b: DevicePtr,
    blocks: Vec<GlmVisionBlockWeights>,
    post_layernorm_w: DevicePtr,
    downsample_w: DevicePtr,
    downsample_b: DevicePtr,
    merger: GlmVisionMergerWeights,
    hidden_size: usize,
    num_heads: usize,
    head_dim: usize,
    patch_dim: usize,
    intermediate_size: usize,
    projection_intermediate_size: usize,
    rms_norm_eps: f32,
    swiglu_limit: f32,
    k_gemm: KernelHandle,
    k_add_bias: KernelHandle,
    k_rms_norm: KernelHandle,
    k_qk_norm_rope: KernelHandle,
    k_attention: KernelHandle,
    k_swiglu: KernelHandle,
    k_add: KernelHandle,
    k_layer_norm: KernelHandle,
    k_gelu: KernelHandle,
    k_merge_reorder: KernelHandle,
    k_f32_bf16: KernelHandle,
    buf_f32: DevicePtr,
    buf_pixels: DevicePtr,
    buf_h1: DevicePtr,
    buf_norm: DevicePtr,
    buf_attn: DevicePtr,
    buf_wide: DevicePtr,
    buf_act: DevicePtr,
    buf_conv: DevicePtr,
    buf_merger: DevicePtr,
    buf_rope_cos: DevicePtr,
    buf_rope_sin: DevicePtr,
}

impl GlmVisionEncoder {
    pub(crate) fn new(
        weights: GlmVisionWeights,
        config: &VisionConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        ensure!(
            config.is_glm5_next,
            "GLM vision constructor received a non-GLM config"
        );
        ensure!(
            config.hidden_size.is_multiple_of(config.num_heads),
            "GLM vision hidden_size must divide num_heads"
        );
        ensure!(
            weights.blocks.len() == config.depth,
            "GLM vision block count mismatch"
        );
        let (p_max, asked_for) = derive_glm_max_patches(
            config.max_pixels,
            config.patch_size,
            config.temporal_patch_size,
        );
        if let Some(wanted) = asked_for {
            tracing::warn!(
                "GLM vision capacity clamped to {p_max} patches from requested {wanted}"
            );
        }
        let head_dim = config.hidden_size / config.num_heads;
        ensure!(
            head_dim == FLASH_HEAD_DIM,
            "GLM vision attention supports head_dim {FLASH_HEAD_DIM}, got {head_dim}"
        );
        let patch_dim =
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
        let wide = (config.hidden_size * 3)
            .max(config.intermediate_size * 2)
            .max(config.projection_intermediate_size * 2);
        let buf_f32 = gpu.alloc(p_max * patch_dim * 4)?;
        let buf_pixels = gpu.alloc(p_max * patch_dim * 2)?;
        let buf_h1 = gpu.alloc(p_max * config.hidden_size * 2)?;
        let buf_norm = gpu.alloc(p_max * config.hidden_size * 2)?;
        let buf_attn = gpu.alloc(p_max * config.hidden_size * 2)?;
        let buf_wide = gpu.alloc(p_max * wide * 2)?;
        let buf_act = gpu.alloc(
            p_max
                * config
                    .intermediate_size
                    .max(config.projection_intermediate_size)
                * 2,
        )?;
        let buf_conv = gpu.alloc(p_max * config.out_hidden_size * 2)?;
        let buf_merger = gpu.alloc(p_max * config.out_hidden_size * 2)?;
        let buf_out = gpu.alloc(GLM_MAX_OUTPUT_ROWS * config.out_hidden_size * 2)?;
        let buf_rope_cos = gpu.alloc(p_max * head_dim * 2)?;
        let buf_rope_sin = gpu.alloc(p_max * head_dim * 2)?;
        Ok(Self {
            buf_out,
            out_hidden_size: config.out_hidden_size,
            p_max,
            out_rows: GLM_MAX_OUTPUT_ROWS,
            patch_embed_w: weights.patch_embed_w,
            patch_embed_b: weights.patch_embed_b,
            blocks: weights.blocks,
            post_layernorm_w: weights.post_layernorm_w,
            downsample_w: weights.downsample_w,
            downsample_b: weights.downsample_b,
            merger: weights.merger,
            hidden_size: config.hidden_size,
            num_heads: config.num_heads,
            head_dim,
            patch_dim,
            intermediate_size: config.intermediate_size,
            projection_intermediate_size: config.projection_intermediate_size,
            rms_norm_eps: config.rms_norm_eps as f32,
            swiglu_limit: config.swiglu_limit,
            k_gemm: gpu.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            k_add_bias: gpu.kernel("glm_vision_encoder", "glm_vision_add_bias")?,
            k_rms_norm: gpu.kernel("glm_vision_encoder", "glm_vision_rms_norm")?,
            k_qk_norm_rope: gpu.kernel("glm_vision_encoder", "glm_vision_qk_norm_rope")?,
            k_attention: gpu.kernel("glm_vision_encoder", "glm_vision_flash_attention")?,
            k_swiglu: gpu.kernel("glm_vision_encoder", "glm_vision_swiglu_clamp")?,
            k_add: gpu.kernel("glm_vision_encoder", "glm_vision_add")?,
            k_layer_norm: gpu.kernel("glm_vision_encoder", "glm_vision_layer_norm")?,
            k_gelu: gpu.kernel("glm_vision_encoder", "glm_vision_gelu")?,
            k_merge_reorder: gpu.kernel("glm_vision_encoder", "glm_vision_merge_reorder")?,
            k_f32_bf16: gpu.kernel("glm_vision_encoder", "glm_vision_f32_to_bf16")?,
            buf_f32,
            buf_pixels,
            buf_h1,
            buf_norm,
            buf_attn,
            buf_wide,
            buf_act,
            buf_conv,
            buf_merger,
            buf_rope_cos,
            buf_rope_sin,
        })
    }
}
