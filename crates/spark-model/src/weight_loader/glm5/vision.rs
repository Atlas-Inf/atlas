// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3 visual weight loader.  The tower computes in BF16. Its linears load
//! as stored on the NVFP4 release (BF16) and as their BF16 expansion on
//! FP4-Local (MXFP8 attention: exact; W4A16 NVFP4 MLP and merger: within one
//! BF16 rounding, see `qwen35::vision_dense_auto`), whose quantized sources
//! are then released. Norms and biases are BF16 in both.

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

/// The BF16 weight of the vision linear `stem` and its logical `[rows, cols]`
/// (a packed NVFP4 `[rows, cols/2]` U8 counts its unpacked columns).
pub(super) fn linear(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    stem: &str,
) -> Result<(DevicePtr, [usize; 2])> {
    let name = format!("{stem}.weight");
    let t = store
        .get(&name)
        .with_context(|| format!("missing GLM vision tensor '{name}'"))?;
    let [rows, cols] = t.shape[..] else {
        anyhow::bail!(
            "GLM vision tensor '{name}' must be rank 2, got {:?}",
            t.shape
        );
    };
    let cols = if t.dtype == WeightDtype::UInt8 {
        cols * 2
    } else {
        cols
    };
    let w = crate::weight_loader::qwen35::vision_dense_auto(store, stem, gpu)
        .with_context(|| format!("GLM vision linear {stem}"))?;
    Ok((w.weight, [rows, cols]))
}

fn linear_ptr(store: &WeightStore, gpu: &dyn GpuBackend, stem: &str) -> Result<DevicePtr> {
    linear(store, gpu, stem).map(|(ptr, _)| ptr)
}

pub(super) fn concat_weights(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    direct: &str,
    parts: &[String],
) -> Result<DevicePtr> {
    if store.contains(&format!("{direct}.weight")) {
        return linear_ptr(store, gpu, direct);
    }
    // `(ptr, rows, expanded)`: an expanded part is a BF16 copy of an
    // FP4-Local part, freed once copied, or on any error.
    let mut loaded: Vec<(DevicePtr, usize, bool)> = Vec::with_capacity(parts.len());
    let built = (|| -> Result<DevicePtr> {
        let mut cols = None;
        let mut rows = 0usize;
        for part in parts {
            let (ptr, [part_rows, part_cols]) = linear(store, gpu, part)?;
            let expanded = ptr != store.get(&format!("{part}.weight"))?.ptr;
            loaded.push((ptr, part_rows, expanded));
            if let Some(expected) = cols {
                ensure!(
                    part_cols == expected,
                    "GLM vision concatenated weights have inconsistent K"
                );
            } else {
                cols = Some(part_cols);
            }
            rows += part_rows;
        }
        let cols = cols.context("empty GLM vision concatenation")?;
        let dst = gpu.alloc(rows * cols * 2)?;
        let mut row = 0usize;
        for &(ptr, part_rows, _) in &loaded {
            if let Err(e) = gpu.copy_d2d(ptr, dst.offset(row * cols * 2), part_rows * cols * 2) {
                gpu.free(dst)?;
                return Err(e);
            }
            row += part_rows;
        }
        Ok(dst)
    })();
    for (ptr, _, expanded) in loaded {
        if expanded {
            gpu.free(ptr)?;
        }
    }
    built
}

/// Every vision linear stem the loader may have expanded.
fn linear_stems(vp: &str, depth: usize) -> Vec<String> {
    let block = (0..depth).flat_map(|i| {
        ["attn.qkv", "attn.q", "attn.k", "attn.v", "attn.proj"]
            .into_iter()
            .chain([
                "mlp.gate_up_proj",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            ])
            .map(move |p| format!("{vp}.blocks.{i}.{p}"))
    });
    let merger = ["proj", "gate_up_proj", "gate_proj", "up_proj", "down_proj"]
        .into_iter()
        .map(|p| format!("{vp}.merger.{p}"));
    block.chain(merger).collect()
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
            proj_w: linear_ptr(store, gpu, &format!("{base}.attn.proj"))?,
            proj_b: dense_ptr(store, &format!("{base}.attn.proj.bias"))?,
            norm2_w: dense_ptr(store, &format!("{base}.norm2.weight"))?,
            gate_up_w: concat_weights(
                store,
                gpu,
                &format!("{base}.mlp.gate_up_proj"),
                &gate_parts,
            )?,
            gate_up_b: concat_biases(store, gpu, &format!("{base}.mlp.gate_up_proj"), &gate_parts)?,
            down_w: linear_ptr(store, gpu, &format!("{base}.mlp.down_proj"))?,
            down_b: dense_ptr(store, &format!("{base}.mlp.down_proj.bias"))?,
        });
    }
    let merger_base = format!("{vp}.merger");
    let merger_gate_parts = names(&merger_base, &["gate_proj", "up_proj"]);
    let merger = GlmVisionMergerWeights {
        proj_w: linear_ptr(store, gpu, &format!("{merger_base}.proj"))?,
        post_norm_w: dense_ptr(store, &format!("{merger_base}.post_projection_norm.weight"))?,
        post_norm_b: dense_ptr(store, &format!("{merger_base}.post_projection_norm.bias"))?,
        gate_up_w: concat_weights(
            store,
            gpu,
            &format!("{merger_base}.gate_up_proj"),
            &merger_gate_parts,
        )?,
        down_w: linear_ptr(store, gpu, &format!("{merger_base}.down_proj"))?,
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
    let encoder = VisionEncoder::new_glm(weights, vcfg, gpu)?;
    // The tower holds BF16 copies of FP4-Local's quantized linears now.
    let mut released = 0;
    for stem in linear_stems(vp, vcfg.depth) {
        released += super::fp4l::release_expanded(store, gpu, &stem, true)?;
    }
    if released > 0 {
        tracing::info!("GLM vision: released {released} bytes of MXFP8/NVFP4 sources");
    }
    Ok(Some(encoder))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_loader::glm5::fp4l::tests::{FP4L_HEADERS, build, fp4l_config, uploaded};
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::weights::WeightTensor;
    use std::collections::HashMap;

    /// Every quantized vision linear of FP4-Local is one the loader expands
    /// and then releases.
    #[test]
    fn every_fp4local_quantized_vit_linear_is_a_released_stem() {
        let gpu = MockGpuBackend::new();
        let store = build(&gpu, uploaded(FP4L_HEADERS));
        let depth = fp4l_config().vision.as_ref().unwrap().depth;
        let stems: std::collections::HashSet<String> =
            linear_stems("model.visual", depth).into_iter().collect();
        let mut quantized = 0;
        for name in store.names() {
            let Some(stem) = name.strip_suffix(".weight") else {
                continue;
            };
            let dtype = store.get(name).unwrap().dtype;
            if name.starts_with("model.visual.")
                && matches!(dtype, WeightDtype::FP8E4M3 | WeightDtype::UInt8)
            {
                assert!(stems.contains(stem), "{stem}");
                quantized += 1;
            }
        }
        // 24 blocks x (qkv, proj, gate, up, down) + merger (proj, gate, up, down).
        assert_eq!(quantized, 24 * 5 + 4);
    }

    fn tensor(
        gpu: &MockGpuBackend,
        dtype: WeightDtype,
        shape: Vec<usize>,
        fill: u8,
    ) -> WeightTensor {
        let bytes = shape.iter().product::<usize>() * dtype.byte_size();
        let ptr = gpu.alloc(bytes.max(1)).unwrap();
        gpu.copy_h2d(&vec![fill; bytes], ptr).unwrap();
        WeightTensor { ptr, shape, dtype }
    }

    /// BF16 as stored; MXFP8 and packed W4A16 NVFP4 expanded to BF16 with
    /// their logical `[rows, cols]`.
    /// A concatenation whose later part fails frees the earlier parts'
    /// expansions instead of leaking them; a good one frees them after the copy.
    #[test]
    fn concat_frees_part_expansions_on_success_and_on_failure() {
        for (second, ok) in [("m2", true), ("bad", false)] {
            let gpu = MockGpuBackend::new();
            let mut map = HashMap::new();
            for stem in ["m", "m2"] {
                map.insert(
                    format!("{stem}.weight"),
                    tensor(&gpu, WeightDtype::FP8E4M3, vec![8, 64], 0),
                );
                map.insert(
                    format!("{stem}.weight_scale"),
                    tensor(&gpu, WeightDtype::UInt8, vec![8, 2], 127),
                );
            }
            // K 32 against the first part's 64.
            map.insert(
                "bad.weight".to_string(),
                tensor(&gpu, WeightDtype::BF16, vec![8, 32], 0),
            );
            let store = WeightStore::from_map(map);
            let allocs = gpu.alloc_count();
            let parts = ["m".to_string(), second.to_string()];
            let fused = concat_weights(&store, &gpu, "absent", &parts);
            assert_eq!(fused.is_ok(), ok, "{second}");
            // Only the fused weight (on success) outlives the call.
            assert_eq!(gpu.alloc_count(), allocs + usize::from(ok), "{second}");
        }
    }

    #[test]
    fn vit_linears_load_bf16_as_stored_and_expand_quantized_ones() {
        let gpu = MockGpuBackend::new();
        let mut map = HashMap::new();
        map.insert(
            "b.weight".to_string(),
            tensor(&gpu, WeightDtype::BF16, vec![8, 64], 0),
        );
        map.insert(
            "m.weight".to_string(),
            tensor(&gpu, WeightDtype::FP8E4M3, vec![8, 64], 0),
        );
        map.insert(
            "m.weight_scale".to_string(),
            tensor(&gpu, WeightDtype::UInt8, vec![8, 2], 127),
        );
        map.insert(
            "n.weight".to_string(),
            tensor(&gpu, WeightDtype::UInt8, vec![8, 32], 0),
        );
        map.insert(
            "n.weight_scale".to_string(),
            tensor(&gpu, WeightDtype::FP8E4M3, vec![8, 4], 0x38),
        );
        let s2 = tensor(&gpu, WeightDtype::FP32, vec![], 0);
        gpu.copy_h2d(&1.0f32.to_le_bytes(), s2.ptr).unwrap();
        map.insert("n.weight_scale_2".to_string(), s2);
        let store = WeightStore::from_map(map);
        let (b, b_shape) = linear(&store, &gpu, "b").unwrap();
        assert_eq!((b, b_shape), (store.get("b.weight").unwrap().ptr, [8, 64]));
        assert_eq!(gpu.launch_count(), 0);
        for stem in ["m", "n"] {
            let (ptr, shape) = linear(&store, &gpu, stem).unwrap();
            assert_ne!(ptr, store.get(&format!("{stem}.weight")).unwrap().ptr);
            assert_eq!(shape, [8, 64], "{stem}");
        }
        assert_eq!(gpu.launch_count(), 2);
    }
}
