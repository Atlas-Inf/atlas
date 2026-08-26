// SPDX-License-Identifier: AGPL-3.0-only

//! Gemma-4 E2B vision-tower weight loader (Wave 2B).
//!
//! Maps the checkpoint's `model.vision_tower.*` + `model.embed_vision.*`
//! tensors into [`GemmaVisionWeights`] and builds the [`GemmaVisionEncoder`]
//! via its verified constructor. Tensor map (verified E2B header; all under
//! `model.vision_tower.` unless noted):
//!
//! | Checkpoint tensor | Field |
//! |---|---|
//! | `patch_embedder.input_proj.weight` | `input_proj` |
//! | `patch_embedder.position_embedding_table` [2,10240,768] | `position_table` |
//! | `encoder.layers.{i}.input_layernorm.weight` | `input_layernorm` |
//! | `.{i}.self_attn.q_norm.weight` / `k_norm.weight` [64] | `q_norm` / `k_norm` |
//! | `.{i}.self_attn.{q,k,v,o}_proj.linear.weight` + 4 clip scalars each | `{q,k,v,o}_proj` |
//! | `.{i}.{post_attention,pre_feedforward,post_feedforward}_layernorm.weight` | the 3 norms |
//! | `.{i}.mlp.{gate,up,down}_proj.linear.weight` + 4 clip scalars each | `{gate,up,down}_proj` |
//! | `model.embed_vision.embedding_projection.weight` [1536,768] | `embed_vision_projection` |
//!
//! Clip bounds are 0-d scalars; the Gemma family stores them BF16
//! (`layer_scalar` in the text tower), HF may store FP32 — [`clip_scalar`]
//! handles both dtypes, erroring on anything else (PCND).

use anyhow::{Result, ensure};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use crate::layers::{
    ClipLinearWeights, GemmaVisionEncoder, GemmaVisionLayerWeights, GemmaVisionWeights,
};
use crate::weight_map::dense;

/// Vision-tower weight prefix (top-level, NOT under `model.language_model.`).
const VISION_PREFIX: &str = "model.vision_tower";
const EMBED_VISION_PREFIX: &str = "model.embed_vision";

/// Load the Gemma-4 E2B vision tower. `Ok(None)` for text-only checkpoints
/// (`config.gemma_vision` unset — 26B/31B ship no `vision_config`); otherwise
/// loads every tensor BF16 (`dense`), parses clip bounds via [`clip_scalar`],
/// and hands [`GemmaVisionWeights`] to [`GemmaVisionEncoder::new`].
pub(super) fn load_gemma_vision_encoder_impl(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<GemmaVisionEncoder>> {
    let vcfg = match &config.gemma_vision {
        Some(v) => v.clone(),
        None => return Ok(None),
    };

    let input_proj = dense(
        store,
        &format!("{VISION_PREFIX}.patch_embedder.input_proj.weight"),
    )?;
    let position_table = dense(
        store,
        &format!("{VISION_PREFIX}.patch_embedder.position_embedding_table"),
    )?;

    let mut layers = Vec::with_capacity(vcfg.num_hidden_layers);
    for i in 0..vcfg.num_hidden_layers {
        layers.push(load_vision_layer(store, gpu, i)?);
    }

    let embed_vision_projection = dense(
        store,
        &format!("{EMBED_VISION_PREFIX}.embedding_projection.weight"),
    )?;

    let weights = GemmaVisionWeights {
        input_proj,
        position_table,
        layers,
        embed_vision_projection,
    };
    let enc = GemmaVisionEncoder::new(&weights, &vcfg, gpu)?;
    tracing::info!(
        "Gemma-4 E2B: vision tower loaded — {} layers, hidden={}, heads={}, position_size={}",
        vcfg.num_hidden_layers,
        vcfg.hidden_size,
        vcfg.num_attention_heads,
        vcfg.position_embedding_size,
    );
    Ok(Some(enc))
}

/// Load one ViT block: 4 norms + QK-norm + 7 clipped linears.
fn load_vision_layer(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    i: usize,
) -> Result<GemmaVisionLayerWeights> {
    let lp = format!("{VISION_PREFIX}.encoder.layers.{i}");
    let attn = format!("{lp}.self_attn");
    let mlp = format!("{lp}.mlp");
    Ok(GemmaVisionLayerWeights {
        input_layernorm: dense(store, &format!("{lp}.input_layernorm.weight"))?,
        q_norm: dense(store, &format!("{attn}.q_norm.weight"))?,
        k_norm: dense(store, &format!("{attn}.k_norm.weight"))?,
        q_proj: clip_linear(store, gpu, &format!("{attn}.q_proj"))?,
        k_proj: clip_linear(store, gpu, &format!("{attn}.k_proj"))?,
        v_proj: clip_linear(store, gpu, &format!("{attn}.v_proj"))?,
        o_proj: clip_linear(store, gpu, &format!("{attn}.o_proj"))?,
        post_attention_layernorm: dense(store, &format!("{lp}.post_attention_layernorm.weight"))?,
        pre_feedforward_layernorm: dense(store, &format!("{lp}.pre_feedforward_layernorm.weight"))?,
        gate_proj: clip_linear(store, gpu, &format!("{mlp}.gate_proj"))?,
        up_proj: clip_linear(store, gpu, &format!("{mlp}.up_proj"))?,
        down_proj: clip_linear(store, gpu, &format!("{mlp}.down_proj"))?,
        post_feedforward_layernorm: dense(
            store,
            &format!("{lp}.post_feedforward_layernorm.weight"),
        )?,
    })
}

/// Load one `ClippableLinear`: BF16 `linear.weight` + its 4 clip scalars.
fn clip_linear(
    store: &WeightStore,
    gpu: &dyn GpuBackend,
    prefix: &str,
) -> Result<ClipLinearWeights> {
    Ok(ClipLinearWeights {
        weight: dense(store, &format!("{prefix}.linear.weight"))?,
        input_min: clip_scalar(store, gpu, &format!("{prefix}.input_min"))?,
        input_max: clip_scalar(store, gpu, &format!("{prefix}.input_max"))?,
        output_min: clip_scalar(store, gpu, &format!("{prefix}.output_min"))?,
        output_max: clip_scalar(store, gpu, &format!("{prefix}.output_max"))?,
    })
}

/// Read a 0-d / 1-element scalar clip bound: BF16 (the Gemma family
/// convention, e.g. `layer_scalar`) or FP32; anything else is a named error.
fn clip_scalar(store: &WeightStore, gpu: &dyn GpuBackend, name: &str) -> Result<f32> {
    let w = store.get(name)?;
    ensure!(
        w.num_elements() == 1,
        "gemma vision: expected scalar {name}, got shape {:?}",
        w.shape
    );
    match w.dtype {
        WeightDtype::FP32 => {
            let mut buf = [0u8; 4];
            gpu.copy_d2h(w.ptr, &mut buf)?;
            gpu.synchronize(gpu.default_stream())?;
            Ok(f32::from_le_bytes(buf))
        }
        WeightDtype::BF16 => {
            let mut buf = [0u8; 2];
            gpu.copy_d2h(w.ptr, &mut buf)?;
            gpu.synchronize(gpu.default_stream())?;
            Ok(f32::from_bits((u16::from_le_bytes(buf) as u32) << 16))
        }
        other => anyhow::bail!(
            "gemma vision: clip scalar {name} has unsupported dtype {other:?} \
             (expected BF16 or FP32)"
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::{EMBED_VISION_PREFIX, VISION_PREFIX, load_gemma_vision_encoder_impl};
    use atlas_core::config::parse_config;
    use spark_runtime::gpu::mock::MockGpuBackend;
    use spark_runtime::gpu::{DevicePtr, GpuBackend};
    use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

    /// Small synthetic vision tower (hidden 16, 2 layers); geometry
    /// invariants hold (heads×head_dim==hidden, p_max==s_max×pks²).
    const E2B_VISION_JSON: &str = r#"{
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 1536,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "intermediate_size": 6144,
            "vocab_size": 262144,
            "max_position_embeddings": 131072,
            "rms_norm_eps": 1e-6
        },
        "vision_config": {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "head_dim": 4,
            "patch_size": 2,
            "pooling_kernel_size": 2,
            "position_embedding_size": 16,
            "use_clipped_linears": true,
            "max_soft_tokens": 4,
            "rms_norm_eps": 1e-6,
            "rope_parameters": {"rope_theta": 100.0}
        },
        "image_token_id": 262144,
        "video_token_id": 262145,
        "boi_token_id": 262146,
        "eoi_token_id": 262147
    }"#;

    /// Text-only Gemma-4 (no `vision_config`) — the 26B/31B shape.
    const TEXT_ONLY_JSON: &str = r#"{
        "model_type": "gemma4",
        "text_config": {
            "hidden_size": 5376,
            "num_hidden_layers": 12,
            "num_attention_heads": 32,
            "num_key_value_heads": 16,
            "head_dim": 256,
            "intermediate_size": 21504,
            "vocab_size": 262144,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6
        }
    }"#;

    /// `GemmaVisionEncoder` lacks `Debug` — `Option::expect` won't compile.
    fn expect_loaded(
        r: anyhow::Result<Option<crate::layers::GemmaVisionEncoder>>,
    ) -> crate::layers::GemmaVisionEncoder {
        match r {
            Ok(Some(enc)) => enc,
            Ok(None) => panic!("expected Some(GemmaVisionEncoder), got None"),
            Err(e) => panic!("vision tower load failed: {e:#}"),
        }
    }

    /// Fake pointer tensor (never read from the mock — pointer-extracted).
    fn tensor(base: u64, shape: &[usize]) -> WeightTensor {
        WeightTensor {
            ptr: DevicePtr(base),
            shape: shape.to_vec(),
            dtype: WeightDtype::BF16,
        }
    }

    /// Clip-bound scalar backed by a REAL mock allocation so `copy_d2h` can
    /// read it back; `bf16` picks the storage dtype (both are legal).
    fn clip_scalar(gpu: &MockGpuBackend, v: f32, bf16: bool) -> WeightTensor {
        if bf16 {
            let bits = ((v.to_bits() >> 16) as u16).to_le_bytes();
            let ptr = gpu.alloc(2).unwrap();
            gpu.copy_h2d(&bits, ptr).unwrap();
            WeightTensor {
                ptr,
                shape: vec![],
                dtype: WeightDtype::BF16,
            }
        } else {
            let ptr = gpu.alloc(4).unwrap();
            gpu.copy_h2d(&v.to_le_bytes(), ptr).unwrap();
            WeightTensor {
                ptr,
                shape: vec![],
                dtype: WeightDtype::FP32,
            }
        }
    }

    /// Full synthetic vision-tower store; values are BF16-exact (½ multiples).
    fn build_store(gpu: &MockGpuBackend) -> WeightStore {
        let mut w = std::collections::HashMap::new();
        // The position table must be a REAL allocation ([2,16,16]×2 bytes) —
        // the constructor downloads it via copy_d2h.
        let pos_bytes = 2 * 16 * 16 * 2;
        let pos_ptr = gpu.alloc(pos_bytes).unwrap();
        gpu.copy_h2d(&vec![0u8; pos_bytes], pos_ptr).unwrap();
        w.insert(
            format!("{VISION_PREFIX}.patch_embedder.position_embedding_table"),
            WeightTensor {
                ptr: pos_ptr,
                shape: vec![2, 16, 16],
                dtype: WeightDtype::BF16,
            },
        );
        w.insert(
            format!("{VISION_PREFIX}.patch_embedder.input_proj.weight"),
            tensor(0x1000, &[16, 16]),
        );
        for i in 0..2 {
            let lp = format!("{VISION_PREFIX}.encoder.layers.{i}");
            let attn = format!("{lp}.self_attn");
            let mlp = format!("{lp}.mlp");
            w.insert(
                format!("{lp}.input_layernorm.weight"),
                tensor(0x2000 + i * 0x2000, &[16]),
            );
            w.insert(
                format!("{attn}.q_norm.weight"),
                tensor(0x2100 + i * 0x2000, &[4]),
            );
            w.insert(
                format!("{attn}.k_norm.weight"),
                tensor(0x2200 + i * 0x2000, &[4]),
            );
            w.insert(
                format!("{lp}.post_attention_layernorm.weight"),
                tensor(0x2300 + i * 0x2000, &[16]),
            );
            w.insert(
                format!("{lp}.pre_feedforward_layernorm.weight"),
                tensor(0x2400 + i * 0x2000, &[16]),
            );
            w.insert(
                format!("{lp}.post_feedforward_layernorm.weight"),
                tensor(0x2500 + i * 0x2000, &[16]),
            );
            // Attention clipped linears; q/k stored BF16, v/o FP32 — both
            // storage dtypes must parse to the right f32.
            for (j, proj) in ["q_proj", "k_proj", "v_proj", "o_proj"].iter().enumerate() {
                let p = format!("{attn}.{proj}");
                w.insert(
                    format!("{p}.linear.weight"),
                    tensor(0x3000 + i * 0x2000 + j as u64 * 0x100, &[16, 16]),
                );
                let bf16 = j < 2;
                w.insert(
                    format!("{p}.input_min"),
                    clip_scalar(gpu, -1.0 - j as f32, bf16),
                );
                w.insert(
                    format!("{p}.input_max"),
                    clip_scalar(gpu, 2.0 + j as f32, bf16),
                );
                w.insert(
                    format!("{p}.output_min"),
                    clip_scalar(gpu, -3.0 - j as f32, bf16),
                );
                w.insert(
                    format!("{p}.output_max"),
                    clip_scalar(gpu, 4.0 + j as f32, bf16),
                );
            }
            for (j, proj) in ["gate_proj", "up_proj", "down_proj"].iter().enumerate() {
                let p = format!("{mlp}.{proj}");
                let shape = if j == 2 { vec![16, 32] } else { vec![32, 16] };
                w.insert(
                    format!("{p}.linear.weight"),
                    tensor(0x3400 + i * 0x2000 + j as u64 * 0x100, &shape),
                );
                w.insert(
                    format!("{p}.input_min"),
                    clip_scalar(gpu, -0.5 - j as f32, true),
                );
                w.insert(
                    format!("{p}.input_max"),
                    clip_scalar(gpu, 1.5 + j as f32, true),
                );
                w.insert(
                    format!("{p}.output_min"),
                    clip_scalar(gpu, -2.5 - j as f32, true),
                );
                w.insert(
                    format!("{p}.output_max"),
                    clip_scalar(gpu, 3.5 + j as f32, true),
                );
            }
        }
        w.insert(
            format!("{EMBED_VISION_PREFIX}.embedding_projection.weight"),
            tensor(0x6000, &[1536, 16]),
        );
        WeightStore::from_map(w)
    }

    /// Every checkpoint tensor must land in the exact field it names.
    #[test]
    fn vision_loader_maps_all_fields() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(E2B_VISION_JSON).unwrap();
        let store = build_store(&gpu);
        let enc = expect_loaded(load_gemma_vision_encoder_impl(&store, &cfg, &gpu));

        assert_eq!(enc.input_proj_w.0, 0x1000, "input_proj");
        assert_eq!(enc.embed_vision_proj_w.0, 0x6000, "embed_vision projection");

        // Layer 0: norms + QK-norm + attn/mlp linears.
        let l0 = &enc.layers[0];
        assert_eq!(l0.input_layernorm.weight.0, 0x2000, "L0 input_layernorm");
        assert_eq!(l0.q_norm.weight.0, 0x2100, "L0 q_norm");
        assert_eq!(l0.k_norm.weight.0, 0x2200, "L0 k_norm");
        assert_eq!(
            l0.post_attention_layernorm.weight.0, 0x2300,
            "L0 post_attn_norm"
        );
        assert_eq!(
            l0.pre_feedforward_layernorm.weight.0, 0x2400,
            "L0 pre_ffn_norm"
        );
        assert_eq!(
            l0.post_feedforward_layernorm.weight.0, 0x2500,
            "L0 post_ffn_norm"
        );
        assert_eq!(l0.q_proj.weight.weight.0, 0x3000, "L0 q_proj");
        assert_eq!(l0.k_proj.weight.weight.0, 0x3100, "L0 k_proj");
        assert_eq!(l0.v_proj.weight.weight.0, 0x3200, "L0 v_proj");
        assert_eq!(l0.o_proj.weight.weight.0, 0x3300, "L0 o_proj");
        assert_eq!(l0.gate_proj.weight.weight.0, 0x3400, "L0 gate_proj");
        assert_eq!(l0.up_proj.weight.weight.0, 0x3500, "L0 up_proj");
        assert_eq!(l0.down_proj.weight.weight.0, 0x3600, "L0 down_proj");

        // Layer 1 spot-checks (offset per layer).
        let l1 = &enc.layers[1];
        assert_eq!(l1.input_layernorm.weight.0, 0x4000, "L1 input_layernorm");
        assert_eq!(l1.q_norm.weight.0, 0x4100, "L1 q_norm");
        assert_eq!(l1.k_norm.weight.0, 0x4200, "L1 k_norm");
        assert_eq!(l1.q_proj.weight.weight.0, 0x5000, "L1 q_proj");
        assert_eq!(l1.o_proj.weight.weight.0, 0x5300, "L1 o_proj");
        assert_eq!(l1.gate_proj.weight.weight.0, 0x5400, "L1 gate_proj");
        assert_eq!(l1.down_proj.weight.weight.0, 0x5600, "L1 down_proj");
        assert_eq!(enc.layers.len(), 2, "layer count");
    }

    /// Clip bounds parse to the exact f32 from BOTH storage dtypes (q/k
    /// BF16, v/o FP32 attn linears; all MLP linears BF16).
    #[test]
    fn vision_loader_parses_clip_bounds() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(E2B_VISION_JSON).unwrap();
        let store = build_store(&gpu);
        let enc = expect_loaded(load_gemma_vision_encoder_impl(&store, &cfg, &gpu));

        // Attention, layer 0: q_proj j=0 (BF16), v_proj j=2 (FP32).
        let q = &enc.layers[0].q_proj;
        assert_eq!(q.input_min, -1.0, "q_proj input_min");
        assert_eq!(q.input_max, 2.0, "q_proj input_max");
        assert_eq!(q.output_min, -3.0, "q_proj output_min");
        assert_eq!(q.output_max, 4.0, "q_proj output_max");
        let v = &enc.layers[0].v_proj;
        assert_eq!(v.input_min, -3.0, "v_proj input_min");
        assert_eq!(v.input_max, 4.0, "v_proj input_max");
        assert_eq!(v.output_min, -5.0, "v_proj output_min");
        assert_eq!(v.output_max, 6.0, "v_proj output_max");
        let o = &enc.layers[1].o_proj;
        assert_eq!(o.input_min, -4.0, "L1 o_proj input_min");
        assert_eq!(o.output_max, 7.0, "L1 o_proj output_max");

        // MLP, layer 1: down_proj j=2 (BF16).
        let d = &enc.layers[1].down_proj;
        assert_eq!(d.input_min, -2.5, "L1 down_proj input_min");
        assert_eq!(d.input_max, 3.5, "L1 down_proj input_max");
        assert_eq!(d.output_min, -4.5, "L1 down_proj output_min");
        assert_eq!(d.output_max, 5.5, "L1 down_proj output_max");
    }

    /// Text-only config returns `Ok(None)` against an empty store — never
    /// errors, never loads.
    #[test]
    fn text_only_config_returns_none() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(TEXT_ONLY_JSON).unwrap();
        assert!(cfg.gemma_vision.is_none());
        let store = WeightStore::empty();
        let enc = load_gemma_vision_encoder_impl(&store, &cfg, &gpu).unwrap();
        assert!(enc.is_none());
    }

    /// A missing critical tensor fails with a NAMED error, never a silent
    /// partial tower — drop `embed_vision.embedding_projection.weight`.
    #[test]
    fn missing_critical_tensor_fails_named() {
        let gpu = MockGpuBackend::new();
        let cfg = parse_config(E2B_VISION_JSON).unwrap();
        let missing = format!("{EMBED_VISION_PREFIX}.embedding_projection.weight");
        let full = build_store(&gpu);
        // WeightStore has no removal API — rebuild the map minus the key.
        let mut map = full
            .names()
            .map(|n| {
                let t = full.get(n).unwrap();
                (
                    n.to_string(),
                    WeightTensor {
                        ptr: t.ptr,
                        shape: t.shape.clone(),
                        dtype: t.dtype,
                    },
                )
            })
            .collect::<std::collections::HashMap<_, _>>();
        map.remove(&missing);
        let reduced = WeightStore::from_map(map);
        // `unwrap_err()` needs `T: Debug` on the Ok payload — match instead.
        let err = match load_gemma_vision_encoder_impl(&reduced, &cfg, &gpu) {
            Ok(_) => panic!("expected an error for missing {missing}"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&missing),
            "error should name the missing tensor, got: {msg}"
        );
    }
}
