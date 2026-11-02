// SPDX-License-Identifier: AGPL-3.0-only

//! Native GLM-5.3 visual tower state.
//!
//! GLM's visual weights are dense BF16 even when the language model is NVFP4.
//! This module keeps that tower separate from the Qwen ViT implementation:
//! the two encoders do not share block, merger, or attention semantics.

use anyhow::{Result, ensure};
use atlas_core::config::VisionConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::vision_preprocess::{
    derive_glm_max_patches, glm_max_item_tokens, glm_max_sequence_patches,
};

mod forward;
#[cfg(test)]
#[path = "glm/forward_tests.rs"]
mod forward_tests;
#[cfg(all(test, feature = "cuda"))]
#[path = "glm/gpu_tests.rs"]
mod gpu_tests;
mod ops;
#[cfg(test)]
#[path = "glm/test_tower.rs"]
pub(crate) mod test_tower;

/// Queries per flash-attention block (four warps of 16 rows).
const FLASH_QUERY_ROWS: u32 = 64;
/// Head width the flash-attention kernel is compiled for.
const FLASH_HEAD_DIM: usize = 64;

/// Patch rows per launch of the token-local stages: the patch embed, the
/// block MLPs and (as `ROW_CHUNK / merge²` merged rows) the merger. Attention
/// needs a sequence's whole Q/K/V, so the residual stream, its norm and QKV
/// span `p_max` rows; everything around attention runs in row chunks, which
/// bounds its scratch without touching any row's arithmetic (the GEMM tiles
/// are fixed, so a row's result does not depend on M). The reference chunks
/// its projections and merger by the same 4096 rows.
const ROW_CHUNK: usize = 4096;

/// Merged rows per merger launch: `row_chunk` patch rows of 2x2 blocks (the
/// parser requires `spatial_merge_size` 2).
fn merged_chunk(row_chunk: usize) -> usize {
    (row_chunk / 4).max(1)
}

/// Merged rows `buf_out` holds beyond the largest single item. Every item is
/// its own attention sequence, but the rows of all items an encode covers are
/// packed back to back and spliced from this one buffer: a prefill chunk's
/// window holds the items it touches, so a maximal video plus two 8192-token
/// chunks of neighbouring rows fit.
pub(crate) const GLM_NEIGHBOUR_ROWS: usize = 16_384;

/// Merged rows of the bounded tower ([`GlmCapacity::bounded`]): 16384 rows is
/// eleven 1024x1024 images at 1369 rows each (128 MiB at `out_hidden_size`
/// 4096).
const GLM_BOUNDED_ROWS: usize = 16_384;

/// What a tower holds, on every rank of a world alike: the ranks admit vision
/// requests against it and receive rows into a buffer of this size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct GlmCapacity {
    /// Pre-merge patches of the largest attention sequence.
    p_max: usize,
    /// Row capacity of `buf_out`.
    out_rows: usize,
    /// Patch rows per token-local launch; see [`ROW_CHUNK`].
    row_chunk: usize,
}

impl GlmCapacity {
    /// Everything the serving processor can emit: one still at its token
    /// budget, or one temporal group of a video at its budget, as SparkGLM
    /// 0.2.0 serves them.
    pub(crate) fn budgets(config: &VisionConfig) -> Self {
        let p_max = glm_max_sequence_patches(config);
        Self {
            p_max,
            out_rows: glm_max_item_tokens(config) + GLM_NEIGHBOUR_ROWS,
            row_chunk: ROW_CHUNK.min(p_max),
        }
    }

    /// The pair's tower: `max_pixels` under a fixed ceiling and a fixed row
    /// window, every stage one launch. An item past it is refused at encode,
    /// so a pair serves a smaller still than the budgets admit.
    pub(crate) fn bounded(config: &VisionConfig) -> Self {
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
        Self {
            p_max,
            out_rows: GLM_BOUNDED_ROWS,
            row_chunk: p_max,
        }
    }

    /// The tower of `config`'s world: on many ranks, the budgets.
    pub(crate) fn of_world(
        config: &atlas_core::config::ModelConfig,
        vision: &VisionConfig,
    ) -> Self {
        if super::glm_budget_tower(config) {
            Self::budgets(vision)
        } else {
            Self::bounded(vision)
        }
    }
}

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

impl GlmVisionWeights {
    /// The weights of a tower that never encodes (see [`GlmVisionEncoder::new`]).
    fn absent() -> Self {
        let n = DevicePtr::NULL;
        Self {
            patch_embed_w: n,
            patch_embed_b: n,
            blocks: Vec::new(),
            post_layernorm_w: n,
            downsample_w: n,
            downsample_b: n,
            merger: GlmVisionMergerWeights {
                proj_w: n,
                post_norm_w: n,
                post_norm_b: n,
                gate_up_w: n,
                down_w: n,
            },
        }
    }
}

pub(crate) struct GlmVisionEncoder {
    pub buf_out: DevicePtr,
    /// False on a rank that only receives rows (built without weights).
    encodes: bool,
    pub out_hidden_size: usize,
    /// Pre-merge patches of the largest attention sequence (one still, or one
    /// temporal group of a video) the processor can emit.
    pub p_max: usize,
    /// Row capacity of `buf_out`; see [`GLM_NEIGHBOUR_ROWS`].
    pub out_rows: usize,
    /// Patch rows per token-local launch; see [`ROW_CHUNK`].
    row_chunk: usize,
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
    /// Norm output, attention output and the MLP's in-place down projection.
    buf_norm: DevicePtr,
    /// QKV of the whole sequence; the attention projection's output once QKV
    /// is consumed; then row-chunk scratch for the MLP and merger.
    buf_wide: DevicePtr,
    buf_act: DevicePtr,
    buf_conv: DevicePtr,
    buf_merger: DevicePtr,
    buf_rope_cos: DevicePtr,
    buf_rope_sin: DevicePtr,
}

impl GlmVisionEncoder {
    /// The tower with `weights`, or without (`None`): a multi-rank worker
    /// receives every encoded row from rank 0 into `buf_out`, so it holds that
    /// buffer and no weights or ViT scratch, and refuses to encode.
    pub(crate) fn new(
        weights: Option<GlmVisionWeights>,
        config: &VisionConfig,
        capacity: GlmCapacity,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let encodes = weights.is_some();
        let weights = weights.unwrap_or_else(GlmVisionWeights::absent);
        let mut scratch_bytes = 0usize;
        let mut alloc = |bytes: usize| {
            if encodes {
                scratch_bytes += bytes;
                gpu.alloc(bytes)
            } else {
                Ok(DevicePtr::NULL)
            }
        };
        ensure!(
            config.is_glm5_next,
            "GLM vision constructor received a non-GLM config"
        );
        ensure!(
            config.hidden_size.is_multiple_of(config.num_heads),
            "GLM vision hidden_size must divide num_heads"
        );
        ensure!(
            !encodes || weights.blocks.len() == config.depth,
            "GLM vision block count mismatch"
        );
        let GlmCapacity {
            p_max,
            out_rows,
            row_chunk,
        } = capacity;
        let merged_chunk = merged_chunk(row_chunk);
        let head_dim = config.hidden_size / config.num_heads;
        ensure!(
            head_dim == FLASH_HEAD_DIM,
            "GLM vision attention supports head_dim {FLASH_HEAD_DIM}, got {head_dim}"
        );
        let patch_dim =
            config.in_channels * config.temporal_patch_size * config.patch_size * config.patch_size;
        let (hidden, out_hidden) = (config.hidden_size, config.out_hidden_size);
        let wide = (p_max * 3 * hidden)
            .max(row_chunk * 2 * config.intermediate_size)
            .max(merged_chunk * 4 * hidden)
            .max(merged_chunk * 2 * config.projection_intermediate_size);
        let act = (row_chunk * config.intermediate_size)
            .max(merged_chunk * config.projection_intermediate_size);
        let buf_f32 = alloc(row_chunk * patch_dim * 4)?;
        let buf_pixels = alloc(row_chunk * patch_dim * 2)?;
        let buf_h1 = alloc(p_max * hidden * 2)?;
        let buf_norm = alloc(p_max * hidden * 2)?;
        let buf_wide = alloc(wide * 2)?;
        let buf_act = alloc(act * 2)?;
        let buf_conv = alloc(merged_chunk * out_hidden * 2)?;
        let buf_merger = alloc(merged_chunk * out_hidden * 2)?;
        let buf_out = gpu.alloc(out_rows * out_hidden * 2)?;
        let buf_rope_cos = alloc(p_max * head_dim * 2)?;
        let buf_rope_sin = alloc(p_max * head_dim * 2)?;
        let gib = |bytes: usize| bytes as f64 / (1u64 << 30) as f64;
        tracing::info!(
            "GLM vision tower: {p_max} patches per sequence, {out_rows} merged rows per encode \
             ({:.2} GiB scratch + {:.2} GiB rows)",
            gib(scratch_bytes),
            gib(out_rows * out_hidden * 2),
        );
        Ok(Self {
            buf_out,
            encodes,
            out_hidden_size: config.out_hidden_size,
            p_max,
            out_rows,
            row_chunk,
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
            buf_wide,
            buf_act,
            buf_conv,
            buf_merger,
            buf_rope_cos,
            buf_rope_sin,
        })
    }
}
