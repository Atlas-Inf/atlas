// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 visual weight loader.  The tower is dense BF16 and is never routed
//! through the language model's NVFP4/dequantized weight helpers.

use anyhow::{Context, Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::{
    GlmVisionBlockWeights, GlmVisionMergerWeights, GlmVisionWeights, VisionEncoder,
};

fn dense_bf16(store: &WeightStore, name: &str) -> Result<(DevicePtr, Vec<usize>)> {
    let tensor = store
        .get(name)
        .with_context(|| format!("missing GLM vision tensor '{name}'"))?;
    ensure!(
        tensor.dtype == WeightDtype::BF16,
        "GLM vision tensor '{name}' must remain dense BF16, found {:?}",
        tensor.dtype
    );
    Ok((tensor.ptr, tensor.shape.clone()))
}

fn dense_ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> {
    dense_bf16(store, name).map(|(ptr, _)| ptr)
}

fn concat_weights(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    direct: &str,
    parts: &[String],
) -> Result<DevicePtr> {
    if store.contains(&format!("{direct}.weight")) {
        return dense_ptr(store, &format!("{direct}.weight"));
    }
    let mut loaded = Vec::with_capacity(parts.len());
    let mut cols = None;
    let mut rows = 0usize;
    for part in parts {
        let (ptr, shape) = dense_bf16(store, &format!("{part}.weight"))?;
        ensure!(
            shape.len() == 2,
            "GLM vision tensor '{part}.weight' must be rank 2"
        );
        let part_cols = shape[1];
        if let Some(expected) = cols {
            ensure!(
                part_cols == expected,
                "GLM vision concatenated weights have inconsistent K"
            );
        } else {
            cols = Some(part_cols);
        }
        rows += shape[0];
        loaded.push((ptr, shape[0]));
    }
    let cols = cols.context("empty GLM vision concatenation")?;
    let dst = gpu.alloc(rows * cols * 2)?;
    let mut row = 0usize;
    for (ptr, part_rows) in loaded {
        gpu.copy_d2d(ptr, dst.offset(row * cols * 2), part_rows * cols * 2)?;
        row += part_rows;
    }
    Ok(dst)
}

fn concat_biases(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    direct: &str,
    parts: &[String],
) -> Result<DevicePtr> {
    if store.contains(&format!("{direct}.bias")) {
        return dense_ptr(store, &format!("{direct}.bias"));
    }
    let mut loaded = Vec::with_capacity(parts.len());
    let mut total = 0usize;
    for part in parts {
        let (ptr, shape) = dense_bf16(store, &format!("{part}.bias"))?;
        ensure!(
            shape.len() == 1,
            "GLM vision bias '{part}.bias' must be rank 1"
        );
        total += shape[0];
        loaded.push((ptr, shape[0]));
    }
    let dst = gpu.alloc(total * 2)?;
    let mut offset = 0usize;
    for (ptr, len) in loaded {
        gpu.copy_d2d(ptr, dst.offset(offset * 2), len * 2)?;
        offset += len;
    }
    Ok(dst)
}

fn names(base: &str, suffixes: &[&str]) -> Vec<String> {
    suffixes
        .iter()
        .map(|suffix| format!("{base}.{suffix}"))
        .collect()
}

pub(crate) fn load(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<VisionEncoder>> {
    let Some(vcfg) = config.vision.as_ref() else {
        return Ok(None);
    };
    ensure!(
        vcfg.is_glm5_next,
        "GLM-5 loader received a non-GLM vision config"
    );
    let vp = if store.contains("model.visual.patch_embed.proj.weight") {
        "model.visual"
    } else if store.contains("model.language_model.visual.patch_embed.proj.weight") {
        "model.language_model.visual"
    } else {
        anyhow::bail!("GLM vision_config is present but model.visual patch weights are missing")
    };
    let patch_embed_w = dense_ptr(store, &format!("{vp}.patch_embed.proj.weight"))?;
    let patch_embed_b = dense_ptr(store, &format!("{vp}.patch_embed.proj.bias"))?;
    let mut blocks = Vec::with_capacity(vcfg.depth);
    for index in 0..vcfg.depth {
        let base = format!("{vp}.blocks.{index}");
        let qkv_parts = names(&format!("{base}.attn"), &["q", "k", "v"]);
        let gate_parts = names(&format!("{base}.mlp"), &["gate_proj", "up_proj"]);
        blocks.push(GlmVisionBlockWeights {
            norm1_w: dense_ptr(store, &format!("{base}.norm1.weight"))?,
            qkv_w: concat_weights(store, gpu, &format!("{base}.attn.qkv"), &qkv_parts)?,
            qkv_b: concat_biases(store, gpu, &format!("{base}.attn.qkv"), &qkv_parts)?,
            q_norm_w: dense_ptr(store, &format!("{base}.attn.q_norm.weight"))?,
            k_norm_w: dense_ptr(store, &format!("{base}.attn.k_norm.weight"))?,
            proj_w: dense_ptr(store, &format!("{base}.attn.proj.weight"))?,
            proj_b: dense_ptr(store, &format!("{base}.attn.proj.bias"))?,
            norm2_w: dense_ptr(store, &format!("{base}.norm2.weight"))?,
            gate_up_w: concat_weights(
                store,
                gpu,
                &format!("{base}.mlp.gate_up_proj"),
                &gate_parts,
            )?,
            gate_up_b: concat_biases(store, gpu, &format!("{base}.mlp.gate_up_proj"), &gate_parts)?,
            down_w: dense_ptr(store, &format!("{base}.mlp.down_proj.weight"))?,
            down_b: dense_ptr(store, &format!("{base}.mlp.down_proj.bias"))?,
        });
    }
    let merger_base = format!("{vp}.merger");
    let merger_gate_parts = names(&merger_base, &["gate_proj", "up_proj"]);
    let merger = GlmVisionMergerWeights {
        proj_w: dense_ptr(store, &format!("{merger_base}.proj.weight"))?,
        post_norm_w: dense_ptr(store, &format!("{merger_base}.post_projection_norm.weight"))?,
        post_norm_b: dense_ptr(store, &format!("{merger_base}.post_projection_norm.bias"))?,
        gate_up_w: concat_weights(
            store,
            gpu,
            &format!("{merger_base}.gate_up_proj"),
            &merger_gate_parts,
        )?,
        down_w: dense_ptr(store, &format!("{merger_base}.down_proj.weight"))?,
    };
    let weights = GlmVisionWeights {
        patch_embed_w,
        patch_embed_b,
        blocks,
        post_layernorm_w: dense_ptr(store, &format!("{vp}.post_layernorm.weight"))?,
        downsample_w: dense_ptr(store, &format!("{vp}.downsample.weight"))?,
        downsample_b: dense_ptr(store, &format!("{vp}.downsample.bias"))?,
        merger,
    };
    Ok(Some(VisionEncoder::new_glm(weights, vcfg, gpu)?))
}
