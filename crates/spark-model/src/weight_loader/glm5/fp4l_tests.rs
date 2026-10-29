// SPDX-License-Identifier: AGPL-3.0-only

//! FP4-Local dispatch against the REAL namespaces of both NVIDIA releases.
//!
//! `test_data/glm5_{fp4local,nvfp4}_headers.json` are every tensor header of
//! nvidia/GLM-5.3-Flash-FP4-Local @ 76348ddb and nvidia/GLM-5.3-Flash-NVFP4
//! @ 423acf37 except routed experts other than 0 and 144
//! (`scripts/glm5_header_extract.py`). The store is rebuilt the way the fast
//! loader leaves it: each CSF-compressed routed-expert scale becomes the plain
//! E4M3 `weight_scale` its decoder produces. Every tensor gets a one-byte mock
//! allocation, so ownership (reclaim) is real and shapes are the checkpoint's.

use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::WeightTensor;
use std::collections::HashMap;

pub(in crate::weight_loader::glm5) const FP4L_HEADERS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/glm5_fp4local_headers.json"
));
pub(in crate::weight_loader::glm5) const NVFP4_HEADERS: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/glm5_nvfp4_headers.json"
));

pub(in crate::weight_loader::glm5) fn fp4l_config() -> ModelConfig {
    atlas_core::config::parse_config(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/glm5_next_fp4local_config.json"
    )))
    .unwrap()
}

pub(in crate::weight_loader::glm5) fn nvfp4_config() -> ModelConfig {
    atlas_core::config::parse_config(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test_data/glm5_next_nvidia_nvfp4_config.json"
    )))
    .unwrap()
}

/// `(name, dtype, shape)` as the store holds them after upload.
pub(in crate::weight_loader::glm5) fn uploaded(
    headers: &str,
) -> Vec<(String, WeightDtype, Vec<usize>)> {
    let raw: serde_json::Map<String, serde_json::Value> = serde_json::from_str(headers).unwrap();
    let shape_of = |v: &serde_json::Value| -> Vec<usize> {
        v[1].as_array()
            .unwrap()
            .iter()
            .map(|d| d.as_u64().unwrap() as usize)
            .collect()
    };
    let mut out = Vec::new();
    for (name, v) in &raw {
        if let Some((base, part)) = name.split_once(".weight_scale.csf32_") {
            // One decoded scale per CSF group: E4M3 [rows, k/16] beside the
            // packed U8 [rows, k/2] weight.
            if part == "codes" {
                let w = shape_of(&raw[&format!("{base}.weight")]);
                out.push((
                    format!("{base}.weight_scale"),
                    WeightDtype::FP8E4M3,
                    vec![w[0], w[1] / 8],
                ));
            }
            continue;
        }
        out.push((
            name.clone(),
            WeightDtype::from_safetensors_str(v[0].as_str().unwrap()).unwrap(),
            shape_of(v),
        ));
    }
    out
}

/// A store of one-byte mock allocations carrying the checkpoint's dtypes and
/// shapes.
pub(in crate::weight_loader::glm5) fn build(
    gpu: &MockGpuBackend,
    tensors: Vec<(String, WeightDtype, Vec<usize>)>,
) -> WeightStore {
    let map: HashMap<String, WeightTensor> = tensors
        .into_iter()
        .map(|(name, dtype, shape)| {
            let ptr = gpu.alloc(1).unwrap();
            (name, WeightTensor { ptr, shape, dtype })
        })
        .collect();
    WeightStore::from_map(map)
}

fn dtype(store: &WeightStore, name: &str) -> WeightDtype {
    store.get(name).unwrap().dtype
}

#[test]
fn fp4local_config_keeps_the_glm53_shape() {
    let (new, old) = (fp4l_config(), nvfp4_config());
    assert_eq!(new.model_type, "glm5_next");
    assert_eq!(new.weight_prefix, old.weight_prefix);
    assert_eq!(new.layer_types, old.layer_types);
    for (a, b) in [
        (new.linear_num_key_heads, old.linear_num_key_heads),
        (new.linear_key_head_dim, old.linear_key_head_dim),
        (new.linear_conv_kernel_dim, old.linear_conv_kernel_dim),
        (new.num_experts, old.num_experts),
        (new.vocab_size, old.vocab_size),
        (new.q_lora_rank, old.q_lora_rank),
        (new.kv_lora_rank, old.kv_lora_rank),
    ] {
        assert_eq!(a, b);
    }
    assert_eq!(new.mlp_only_layers, [0, 1, 2]);
    let qc = new.quantization_config.as_ref().unwrap();
    assert_eq!(qc.quant_algo, "MIXED_PRECISION");
}

/// The global NVFP4 variant (routed experts, expert TP, the shared FP8
/// cache's guard) still resolves to ModelOpt Standard on both releases:
/// FP4-Local's FP8_PB_WO / MXFP8 / W4A16_NVFP4 entries do not outvote NVFP4.
#[test]
fn both_releases_resolve_to_standard_nvfp4() {
    let gpu = MockGpuBackend::new();
    for (headers, config) in [
        (FP4L_HEADERS, fp4l_config()),
        (NVFP4_HEADERS, nvfp4_config()),
    ] {
        let store = build(&gpu, uploaded(headers));
        assert_eq!(
            crate::weight_map::detect_nvfp4_variant(&store, &config),
            crate::weight_map::Nvfp4Variant::Standard
        );
    }
}

/// Every stem the loader expands is, on FP4-Local, exactly the format the
/// module table says, so it reaches the arm that expands it: MXFP8 for KDA,
/// dense MLP and lm_head, the 4-D FP8_PB_WO grid for MLA and shared experts,
/// BF16 for kv_b and the indexer. On the NVFP4 release every target stem is
/// BF16 (or, for the dense MLP, packed NVFP4 the loader uses natively).
#[test]
fn expanded_stems_have_the_formats_each_arm_expects() {
    let gpu = MockGpuBackend::new();
    let config = fp4l_config();
    let new = build(&gpu, uploaded(FP4L_HEADERS));
    let old = build(&gpu, uploaded(NVFP4_HEADERS));
    let (mut mxfp8, mut fp8_block, mut bf16) = (0, 0, 0);
    for layer in 0..config.num_hidden_layers {
        let lp = config.layer_prefix(layer);
        let kind = config.layer_type(layer);
        let dense = config.mlp_only_layers.contains(&layer);
        for stem in decoder_expanded_stems(&lp, kind, dense) {
            let w = format!("{stem}.weight");
            match dtype(&old, &w) {
                WeightDtype::BF16 => {}
                WeightDtype::UInt8 => assert!(dense, "{w}: only the dense MLP is packed"),
                other => panic!("{w}: NVFP4 release ships {other:?}"),
            }
            if let Some((n, k)) = crate::weight_map::mxfp8_geometry(&new, &stem).unwrap() {
                assert_eq!([n, k], new.get(&w).unwrap().shape[..]);
                mxfp8 += 1;
                continue;
            }
            match dtype(&new, &w) {
                WeightDtype::BF16 => bf16 += 1,
                WeightDtype::FP8E4M3 => {
                    let s = new.get(&format!("{stem}.weight_scale")).unwrap();
                    let (sn, sk) = crate::weight_map::block_scale_grid(&s.shape).unwrap();
                    let t = &new.get(&w).unwrap().shape;
                    assert_eq!(
                        (s.dtype, t[0], t[1]),
                        (WeightDtype::FP32, sn * 128, sk * 128),
                        "{stem}"
                    );
                    fp8_block += 1;
                }
                other => panic!("{w}: FP4-Local ships {other:?}"),
            }
        }
    }
    // 34 KDA x 4 + 3 dense x 3; 11 MLA x 4 + 42 shared x 3; 11 x (kv_b + 3 indexer).
    assert_eq!(
        (mxfp8, fp8_block, bf16),
        (34 * 4 + 3 * 3, 11 * 4 + 42 * 3, 11 * 4)
    );
    assert!(
        crate::weight_map::mxfp8_geometry(&new, "lm_head")
            .unwrap()
            .is_some()
    );
    assert_eq!(dtype(&old, "lm_head.weight"), WeightDtype::BF16);
}

/// After a layer is built, its FP8 sources are freed and nothing else:
/// routed experts, BF16 projections (the derived weight itself on the NVFP4
/// release), norms and hc stay.
#[test]
fn release_frees_each_layers_fp8_sources_and_nothing_else() {
    let gpu = MockGpuBackend::new();
    let config = fp4l_config();
    for (headers, fp4l) in [(FP4L_HEADERS, true), (NVFP4_HEADERS, false)] {
        let store = build(&gpu, uploaded(headers));
        let (mut released, mut bytes) = (0usize, 0usize);
        for layer in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(layer);
            let dense = config.mlp_only_layers.contains(&layer);
            for stem in decoder_expanded_stems(&lp, config.layer_type(layer), dense) {
                let fp8 = dtype(&store, &format!("{stem}.weight")) == WeightDtype::FP8E4M3;
                let want = if fp8 {
                    store.get(&format!("{stem}.weight")).unwrap().byte_size()
                        + store
                            .get(&format!("{stem}.weight_scale"))
                            .unwrap()
                            .byte_size()
                } else {
                    0
                };
                assert_eq!(release_expanded(&store, &gpu, &stem, false).unwrap(), want);
                // Idempotent: a second release frees nothing.
                assert_eq!(release_expanded(&store, &gpu, &stem, false).unwrap(), 0);
                released += usize::from(fp8);
                bytes += want;
            }
        }
        let reclaimed = store.reclaimed_count();
        if fp4l {
            assert_eq!(released, 34 * 4 + 3 * 3 + 11 * 4 + 42 * 3);
            assert_eq!(reclaimed, 2 * released);
            // ~4.6 GB KDA + ~1.0 GB MLA + ~1.3 GB shared + 0.4 GB dense.
            assert!((7_000_000_000..8_000_000_000).contains(&bytes), "{bytes}");
        } else {
            assert_eq!((released, reclaimed), (0, 0));
        }
        for name in store.names() {
            if name.contains(".mlp.experts.") || name.ends_with("norm.weight") {
                assert!(!store.was_reclaimed(name), "{name}");
            }
        }
    }
}

#[test]
fn packed_nvfp4_is_released_only_when_the_caller_expanded_it() {
    let gpu = MockGpuBackend::new();
    let store = build(&gpu, uploaded(FP4L_HEADERS));
    let expert = "model.language_model.layers.3.mlp.experts.0.gate_proj";
    assert_eq!(release_expanded(&store, &gpu, expert, false).unwrap(), 0);
    assert!(!store.was_reclaimed(&format!("{expert}.weight")));
    let vit = "model.visual.blocks.0.mlp.gate_proj";
    // [4096, 512] U8 + [4096, 64] E4M3 + one F32.
    assert_eq!(
        release_expanded(&store, &gpu, vit, true).unwrap(),
        4096 * 512 + 4096 * 64 + 4
    );
    assert!(store.was_reclaimed(&format!("{vit}.weight_scale_2")));
}

fn f32_store(gpu: &MockGpuBackend, name: &str, values: &[f32]) -> WeightStore {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let ptr = gpu.alloc(bytes.len()).unwrap();
    gpu.copy_h2d(&bytes, ptr).unwrap();
    WeightStore::from_map(HashMap::from([(
        name.to_string(),
        WeightTensor {
            ptr,
            shape: vec![values.len()],
            dtype: WeightDtype::FP32,
        },
    )]))
}

/// F32 norms round to nearest even (the truncating `dense_f32_safe` would
/// drop the low half); BF16 norms are the store's own tensor.
#[test]
fn f32_norms_round_to_nearest_even_bf16() {
    let gpu = MockGpuBackend::new();
    // 1 + 2^-8 + 2^-12: truncation gives 0x3F80 (1.0), RNE 0x3F81.
    let store = f32_store(&gpu, "n.weight", &[1.0 + 1.0 / 256.0 + 1.0 / 4096.0, -2.0]);
    let w = norm_bf16(&store, "n.weight", &gpu).unwrap();
    let mut got = [0u8; 4];
    gpu.copy_d2h(w.weight, &mut got).unwrap();
    assert_eq!(got, [0x81, 0x3F, 0x00, 0xC0]);

    let gpu = MockGpuBackend::new();
    let store = build(&gpu, uploaded(FP4L_HEADERS));
    let bf16 = "model.language_model.layers.0.input_layernorm.weight";
    assert_eq!(
        norm_bf16(&store, bf16, &gpu).unwrap().weight,
        store.get(bf16).unwrap().ptr
    );
    let fp8 = "model.language_model.layers.0.self_attn.q_proj.weight";
    assert!(norm_bf16(&store, fp8, &gpu).is_err());
}

#[test]
fn null_store_pointers_never_alias_a_mock_allocation() {
    // Guard for the fixture itself: every tensor owns a distinct allocation.
    let gpu = MockGpuBackend::new();
    let store = build(&gpu, uploaded(FP4L_HEADERS));
    let mut seen = std::collections::HashSet::new();
    for name in store.names() {
        let ptr = store.get(name).unwrap().ptr;
        assert_ne!(ptr, DevicePtr::NULL);
        assert!(seen.insert(ptr.0), "{name}");
    }
}

/// `lm_head.weight` of `dtype` `[64, 64]` plus an optional scale.
fn head_store(
    gpu: &MockGpuBackend,
    dtype: WeightDtype,
    scale: Option<(WeightDtype, Vec<usize>)>,
) -> WeightStore {
    let mut map = HashMap::new();
    map.insert(
        "lm_head.weight".to_string(),
        WeightTensor {
            ptr: gpu.alloc(64 * 64 * 2).unwrap(),
            shape: vec![64, 64],
            dtype,
        },
    );
    if let Some((dtype, shape)) = scale {
        let n: usize = shape.iter().product();
        let ptr = gpu.alloc(n * dtype.byte_size()).unwrap();
        gpu.copy_h2d(&vec![127u8; n * dtype.byte_size()], ptr)
            .unwrap();
        map.insert(
            "lm_head.weight_scale".to_string(),
            WeightTensor { ptr, shape, dtype },
        );
    }
    WeightStore::from_map(map)
}

/// The head was read as BF16 whatever its dtype; now BF16 is the store's
/// tensor, MXFP8 its exact expansion (source kept for the split head's
/// shard), and anything else is refused.
#[test]
fn lm_head_is_bf16_or_expanded_mxfp8_and_nothing_else() {
    let gpu = MockGpuBackend::new();
    let store = head_store(&gpu, WeightDtype::BF16, None);
    let head = lm_head_bf16(&store, &gpu).unwrap();
    assert_eq!(head.ptr, store.get("lm_head.weight").unwrap().ptr);

    let gpu = MockGpuBackend::new();
    let store = head_store(
        &gpu,
        WeightDtype::FP8E4M3,
        Some((WeightDtype::UInt8, vec![64, 2])),
    );
    let head = lm_head_bf16(&store, &gpu).unwrap();
    assert_ne!(head.ptr, store.get("lm_head.weight").unwrap().ptr);
    assert_eq!(
        (head.dtype, head.shape.as_slice()),
        (WeightDtype::BF16, &[64, 64][..])
    );
    assert_eq!(gpu.launch_count(), 1, "one exact MXFP8 -> BF16 expansion");
    assert!(!store.was_reclaimed("lm_head.weight"));

    for (dtype, scale) in [
        (
            WeightDtype::UInt8,
            Some((WeightDtype::FP8E4M3, vec![64, 4])),
        ),
        (WeightDtype::FP8E4M3, Some((WeightDtype::FP32, vec![1]))),
        (WeightDtype::FP8E4M3, None),
        (WeightDtype::FP32, None),
    ] {
        let gpu = MockGpuBackend::new();
        let store = head_store(&gpu, dtype, scale.clone());
        assert!(lm_head_bf16(&store, &gpu).is_err(), "{dtype:?} {scale:?}");
        assert_eq!(gpu.launch_count(), 0);
    }
}

/// `ATLAS_GLM_HC_BF16` reads `hc_*_fn` through the store's BF16 pointer
/// (`components::load_hc`); FP4-Local keeps `fn` BF16 and only widens
/// base/scale to F32, which `load_hc_f32` reads natively. So the BF16 mHC
/// kernels keep their input on both releases.
#[test]
fn hc_fn_stays_bf16_and_base_scale_are_bf16_or_f32() {
    let gpu = MockGpuBackend::new();
    let config = fp4l_config();
    for headers in [FP4L_HEADERS, NVFP4_HEADERS] {
        let store = build(&gpu, uploaded(headers));
        for layer in 0..config.num_hidden_layers {
            let lp = config.layer_prefix(layer);
            for site in ["attn", "ffn"] {
                assert_eq!(
                    dtype(&store, &format!("{lp}.hc_{site}_fn")),
                    WeightDtype::BF16
                );
                for part in ["base", "scale"] {
                    let d = dtype(&store, &format!("{lp}.hc_{site}_{part}"));
                    assert!(matches!(d, WeightDtype::BF16 | WeightDtype::FP32), "{d:?}");
                }
            }
        }
    }
}

#[path = "fp4l_test_gpu.rs"]
mod gpu;
