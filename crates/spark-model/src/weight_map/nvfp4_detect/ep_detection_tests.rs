// SPDX-License-Identifier: AGPL-3.0-only

//! Mixed-precision variant detection against a synthetic weight store.
//!
//! Split out of `nvfp4_detect.rs` for the file-size cap. The seam is a real
//! one: nothing here is reachable from production — the tests build stores and
//! ask `detect_nvfp4_variant` what it makes of them. Declared by the parent as
//! `#[cfg(test)] #[path = "nvfp4_detect/ep_detection_tests.rs"] mod
//! ep_detection_tests;`, so this file IS the module body.

use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::weights::WeightStore;

/// A store holding only the FP8 attention marker at a given layer, which is
/// what the detector sniffs for. Names are all the detector reads.
fn store_with(names: &[String]) -> WeightStore {
    use std::collections::HashMap;
    let map: HashMap<String, spark_runtime::weights::WeightTensor> = names
        .iter()
        .map(|n| {
            (
                n.clone(),
                spark_runtime::weights::WeightTensor {
                    ptr: spark_runtime::gpu::DevicePtr::NULL,
                    shape: vec![1],
                    dtype: spark_runtime::weights::WeightDtype::FP8E4M3,
                },
            )
        })
        .collect();
    WeightStore::from_map(map)
}

/// Detection must not depend on which EP rank is asking.
///
/// ★ CHARACTERISATION, not a regression test — it passes with the bug too,
/// and that is worth stating rather than hiding. The old expression indexed
/// the second FP8 prefix by `local_expert_range().0`, a global EXPERT index
/// used as a LAYER index, so rank 1 of 2 probed layer 47 where rank 0
/// probed layer 0. That is genuinely wrong, but UNREACHABLE: the global
/// `.weight_scale_inv` fallback below the per-prefix probes catches the
/// marker wherever it sits, so both ranks answer the same either way.
///
/// This pins the property we want to keep — rank-independence — so that if
/// someone tightens or removes that fallback, the latent bug surfaces here
/// instead of in a two-rank EP deployment.
#[test]
fn variant_detection_is_identical_across_ep_ranks() {
    // Only a deep-layer FP8 attention marker: present at the layer rank 1
    // used to probe, absent at layer 0. Under the bug the ranks disagree.
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    // Reach the tensor-name sniffing path: a present `quantization_config`
    // short-circuits detection before any prefix is built, so leaving it
    // set makes this test assert the early return, not the bug.
    cfg.quantization_config = None;
    let deep = cfg.num_hidden_layers.saturating_sub(1);
    let store = store_with(&[format!(
        "model.language_model.layers.{deep}.self_attn.q_proj.weight_scale_inv"
    )]);

    cfg.ep_world_size = 2;
    cfg.ep_rank = 0;
    let rank0 = detect_nvfp4_variant(&store, &cfg);
    cfg.ep_rank = 1;
    let rank1 = detect_nvfp4_variant(&store, &cfg);

    assert_eq!(
        rank0, rank1,
        "EP rank changed the detected variant for one checkpoint: \
         rank0={rank0:?} rank1={rank1:?}. Detection reads a file every rank \
         sees identically, so it must not depend on the expert split."
    );
}

/// The layer-0 spelling still detects FP8 — the fix must not break the
/// case the buggy expression happened to get right.
#[test]
fn the_alternate_layer0_spelling_still_detects_fp8() {
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    cfg.quantization_config = None;
    let store = store_with(&[
        "model.language_model.layers.0.self_attn.q_proj.weight_scale_inv".to_string(),
    ]);
    assert_eq!(
        detect_nvfp4_variant(&store, &cfg),
        Nvfp4Variant::Fp8Dequanted
    );
}

/// nvidia/Qwen3.8-Flash-Next-NVFP4 shape: `quant_algo=MIXED_PRECISION`
/// with a `quantized_layers` map declaring NVFP4 experts, while the MTP
/// block ships `weight_scale_inv` (FP8_PB_WO). Without the map arm the
/// global `.weight_scale_inv` fallback mis-detects the checkpoint as
/// Fp8Dequanted and routes every NVFP4 expert read into the FP8 path.
#[test]
fn mixed_precision_map_resolves_standard_despite_mtp_scale_inv() {
    use atlas_core::config::{QuantLayerSpec, QuantizationConfig};
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let mut map = std::collections::BTreeMap::new();
    for l in 0..cfg.num_hidden_layers {
        map.insert(
            format!("model.language_model.layers.{l}.mlp.experts"),
            QuantLayerSpec {
                quant_algo: "NVFP4".into(),
                group_size: 16,
            },
        );
    }
    map.insert(
        "mtp.layers.0.mlp.experts".to_string(),
        QuantLayerSpec {
            quant_algo: "FP8_PB_WO".into(),
            group_size: 128,
        },
    );
    map.insert(
        "model.language_model.layers.1.ple.ple_embedding.ngram_embedding".to_string(),
        QuantLayerSpec {
            quant_algo: "FP8".into(),
            group_size: 0,
        },
    );
    cfg.quantization_config = Some(QuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "MIXED_PRECISION".into(),
        format: String::new(),
        ignore_modules: vec![],
        weight_block_size: vec![],
        group_size: 16,
        quantized_layers: map,
        exl3: None,
    });
    // The store shape that fools the fallback: main experts carry the
    // standard ModelOpt triple; ONLY mtp.* carries weight_scale_inv.
    let store = store_with(&[
        "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale".to_string(),
        "mtp.layers.0.mlp.experts.0.gate_proj.weight_scale_inv".to_string(),
    ]);
    assert_eq!(
        detect_nvfp4_variant(&store, &cfg),
        Nvfp4Variant::Standard,
        "MIXED_PRECISION map must route NVFP4-expert checkpoints to Standard \
         even when mtp.* ships FP8 block scales"
    );
}

/// nvidia/Qwen3.8-27B-NVFP4 shape: `quant_algo=MIXED_PRECISION` whose map is
/// genuinely mixed on the main path — FP8 linear_attn/self_attn projections
/// (208 entries) plus NVFP4 MLP gate/up/down (193 entries). An FP8-first
/// vote routes the checkpoint to `Fp8Dequanted`, whose loader then refuses
/// the uint8-packed NVFP4 MLP (`Expected FP8E4M3 ... got UInt8`) — this is
/// the exact failure the PR benchmark gates hit on decode-floor. FP8
/// projections inside a Standard checkpoint are already served per-key by
/// `quantized_any`'s `has_fp8_dense` path, so NVFP4 must win the global vote.
#[test]
fn mixed_precision_fp8_attn_nvfp4_mlp_resolves_standard() {
    use atlas_core::config::{QuantLayerSpec, QuantizationConfig};
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let mut map = std::collections::BTreeMap::new();
    for l in 0..cfg.num_hidden_layers {
        for proj in ["out_proj", "in_proj_qkv", "in_proj_z"] {
            map.insert(
                format!("model.language_model.layers.{l}.linear_attn.{proj}"),
                QuantLayerSpec {
                    quant_algo: "FP8".into(),
                    group_size: 128,
                },
            );
        }
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            map.insert(
                format!("model.language_model.layers.{l}.mlp.{proj}"),
                QuantLayerSpec {
                    quant_algo: "NVFP4".into(),
                    group_size: 16,
                },
            );
        }
    }
    cfg.quantization_config = Some(QuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "MIXED_PRECISION".into(),
        format: String::new(),
        ignore_modules: vec![],
        weight_block_size: vec![],
        group_size: 16,
        quantized_layers: map,
        exl3: None,
    });
    let store = store_with(&[]);
    assert_eq!(
        detect_nvfp4_variant(&store, &cfg),
        Nvfp4Variant::Standard,
        "MIXED_PRECISION map with FP8 attn + NVFP4 MLP must resolve Standard: \
         the FP8 arm cannot read uint8-packed FP4, while FP8 keys inside a \
         Standard checkpoint are dequanted per-key"
    );
}

/// nvidia/Qwen3.6-35B-A3B-NVFP4 shape (GitHub #88): `quant_algo=MIXED_PRECISION`
/// whose map labels FP8 attention (130) plus `W4A16_NVFP4` experts and
/// shared_expert (161) — NOT the bare `"NVFP4"` label. An exact-match vote
/// leaves `saw_nvfp4` false and mis-routes the checkpoint to `Fp8Dequanted`,
/// whose loader dies on `Expected FP8E4M3 ... got UInt8` reading the packed
/// FP4 shared-expert weights. Nemotron-3.5-Lightning's pack uses the same
/// `W4A16_NVFP4` label ×5935.
#[test]
fn mixed_precision_w4a16_nvfp4_labels_resolve_standard() {
    use atlas_core::config::{QuantLayerSpec, QuantizationConfig};
    let mut cfg = ModelConfig::qwen3_next_80b_nvfp4();
    let mut map = std::collections::BTreeMap::new();
    for l in 0..cfg.num_hidden_layers {
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            map.insert(
                format!("model.language_model.layers.{l}.self_attn.{proj}"),
                QuantLayerSpec {
                    quant_algo: "FP8".into(),
                    group_size: 128,
                },
            );
        }
        map.insert(
            format!("model.language_model.layers.{l}.mlp.experts"),
            QuantLayerSpec {
                quant_algo: "W4A16_NVFP4".into(),
                group_size: 16,
            },
        );
        for proj in ["gate_proj", "up_proj", "down_proj"] {
            map.insert(
                format!("model.language_model.layers.{l}.mlp.shared_expert.{proj}"),
                QuantLayerSpec {
                    quant_algo: "W4A16_NVFP4".into(),
                    group_size: 16,
                },
            );
        }
    }
    cfg.quantization_config = Some(QuantizationConfig {
        quant_method: "modelopt".into(),
        quant_algo: "MIXED_PRECISION".into(),
        format: String::new(),
        ignore_modules: vec![],
        weight_block_size: vec![],
        group_size: 16,
        quantized_layers: map,
        exl3: None,
    });
    let store = store_with(&[]);
    assert_eq!(
        detect_nvfp4_variant(&store, &cfg),
        Nvfp4Variant::Standard,
        "MIXED_PRECISION map with FP8 attn + W4A16_NVFP4 experts/shared_expert \
         must resolve Standard — the FP8 arm cannot read uint8-packed FP4"
    );
}
