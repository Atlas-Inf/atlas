// SPDX-License-Identifier: AGPL-3.0-only

//! Checkpoint-plan tests: a synthetic `WeightStore` carrying the EXACT
//! Ling-3.2-air layout (43 layers, 512 experts x 12 members per MoE layer,
//! ViT + projector skips, model-level tensors) drives the single-pass
//! classifier; a handful of small stores drive the failure paths.
//! All pure logic — no GPU.

use super::*;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
use std::collections::HashMap;

fn config() -> ModelConfig {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.num_hidden_layers = 42;
    config.mtp_num_hidden_layers = 1;
    config.num_experts = 512;
    config
}

fn tensor(name: &str, dtype: WeightDtype) -> (String, WeightTensor) {
    (
        name.to_string(),
        WeightTensor {
            ptr: DevicePtr::NULL,
            shape: Vec::new(),
            dtype,
        },
    )
}

/// The exact Ling layout: KDA at {0..42} \ {5,11,17,23,29,35,41}, gated MLA
/// there, dense MLP on 0..1, 512E MoE on 2..=42 (MTP at 42), model-level
/// tensors, and a token sample of the skipped vision/projector namespaces.
fn ling_store() -> WeightStore {
    let mut map = HashMap::new();
    for (name, dtype) in [
        tensor(names::WORD_EMBEDDINGS, WeightDtype::BF16),
        tensor(names::FINAL_NORM, WeightDtype::BF16),
        tensor(names::LM_HEAD, WeightDtype::BF16),
        tensor("model.visual.blocks.0.attn.qkv.weight", WeightDtype::BF16),
        tensor("linear_proj.0.weight", WeightDtype::BF16),
    ] {
        map.insert(name, dtype);
    }
    let mla: &[usize] = &[5, 11, 17, 23, 29, 35, 41];
    for idx in 0..=42 {
        let lp = format!("model.layers.{idx}");
        for name in names::layer_norm_tensors(&lp) {
            map.insert(name.clone(), tensor(&name, WeightDtype::BF16).1);
        }
        let attention = if mla.contains(&idx) || idx == 42 {
            names::mla_attention_tensors(&lp)
        } else {
            names::kda_attention_tensors(&lp)
        };
        for name in attention {
            map.insert(name.clone(), tensor(&name, WeightDtype::BF16).1);
        }
        if idx < 2 {
            for name in names::dense_mlp_tensors(&lp) {
                map.insert(name.clone(), tensor(&name, WeightDtype::BF16).1);
            }
        } else {
            for name in names::router_tensors(&lp)
                .into_iter()
                .chain(names::shared_expert_tensors(&lp))
            {
                let dtype = if name.ends_with(".gate.weight") || name.ends_with(".expert_bias") {
                    WeightDtype::FP32
                } else {
                    WeightDtype::BF16
                };
                map.insert(name.clone(), tensor(&name, dtype).1);
            }
            for expert in 0..512 {
                for name in names::expert_tensor_names(&lp, expert) {
                    let dtype = if name.ends_with("weight_scale") {
                        WeightDtype::FP8E4M3
                    } else if name.ends_with("weight_packed") {
                        WeightDtype::UInt8
                    } else {
                        WeightDtype::FP32
                    };
                    map.insert(name.clone(), tensor(&name, dtype).1);
                }
            }
        }
    }
    for name in mtp::mtp_tensors("model.layers.42") {
        map.insert(name.clone(), tensor(&name, WeightDtype::BF16).1);
    }
    WeightStore::from_map(map)
}

#[test]
fn the_ling_layout_classifies_and_validates() {
    let store = ling_store();
    let plan = build(&store, &config()).unwrap();
    assert_eq!(plan.layers.len(), 42);
    assert_eq!(plan.mtp_layers.len(), 1);
    let mla: &[usize] = &[5, 11, 17, 23, 29, 35, 41];
    for (idx, layer) in plan.layers.iter().enumerate() {
        let attention = if mla.contains(&idx) {
            AttentionKind::GatedMla
        } else {
            AttentionKind::Kda
        };
        assert_eq!(layer.attention, attention, "layer {idx}");
        let mlp = if idx < 2 {
            MlpKind::Dense
        } else {
            MlpKind::Moe
        };
        assert_eq!(layer.mlp, mlp, "layer {idx}");
        if mlp == MlpKind::Moe {
            assert_eq!(layer.experts_seen, 512, "layer {idx}");
        }
    }
    let mtp = &plan.mtp_layers[0];
    assert_eq!(mtp.attention, AttentionKind::GatedMla);
    assert_eq!(mtp.mlp, MlpKind::Moe);
    assert_eq!(mtp.experts_seen, 512);
    assert_eq!(plan.first_k_dense_replace, 2);
    assert_eq!(plan.vision_tensors, 1);
    assert_eq!(plan.projector_tensors, 1);
    assert!(
        plan.summary().contains("35 KDA / 7 gated-MLA"),
        "{}",
        plan.summary()
    );
}

#[test]
fn a_missing_expert_member_fails_the_layer_plan() {
    let missing = "model.layers.3.mlp.experts.7.gate_proj.input_global_scale";
    let store = ling_store();
    let mut map: HashMap<String, WeightTensor> = HashMap::new();
    for name in store.names() {
        if name == missing {
            continue;
        }
        let t = store.get(name).unwrap();
        map.insert(
            name.to_string(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: Vec::new(),
                dtype: t.dtype,
            },
        );
    }
    assert!(!map.contains_key(missing));
    let pruned = WeightStore::from_map(map);
    let err = build(&pruned, &config()).unwrap_err();
    assert!(err.to_string().contains("input_global_scale"), "{err}");
}

#[test]
fn a_missing_attention_tensor_fails_the_layer_plan() {
    // The KDA signature (names.rs) is either A_log OR dt_bias: removing a
    // single stem still classifies the layer as KDA. Remove both to reach
    // the "neither KDA nor MLA" arm.
    let missing = [
        "model.layers.0.attention.A_log",
        "model.layers.0.attention.dt_bias",
    ];
    let mut map: HashMap<String, WeightTensor> = HashMap::new();
    for name in ling_store().names() {
        if missing.contains(&name) {
            continue;
        }
        map.insert(
            name.to_string(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: Vec::new(),
                dtype: WeightDtype::BF16,
            },
        );
    }
    let store = WeightStore::from_map(map);
    let err = build(&store, &config()).unwrap_err();
    assert!(err.to_string().contains("neither KDA nor MLA"), "{err}");
}

#[test]
fn a_stray_tensor_is_refused() {
    let mut map: HashMap<String, WeightTensor> = HashMap::new();
    let store = ling_store();
    for name in store.names() {
        map.insert(
            name.to_string(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: Vec::new(),
                dtype: WeightDtype::BF16,
            },
        );
    }
    map.insert(
        "model.shortcut.weight".to_string(),
        WeightTensor {
            ptr: DevicePtr::NULL,
            shape: Vec::new(),
            dtype: WeightDtype::BF16,
        },
    );
    let store = WeightStore::from_map(map);
    let err = build(&store, &config()).unwrap_err();
    assert!(err.to_string().contains("unrecognized"), "{err}");
}

#[test]
fn a_config_with_too_few_layers_is_refused() {
    // Drop the MTP layer entirely but keep declaring 42 + 1 layers.
    let small = ling_store();
    let mut map = HashMap::new();
    for name in small.names() {
        if name.starts_with("model.layers.42.") {
            continue;
        }
        let t = small.get(name).unwrap();
        map.insert(
            name.to_string(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape: Vec::new(),
                dtype: t.dtype,
            },
        );
    }
    let store = WeightStore::from_map(map);
    let config = config();
    let err = build(&store, &config).unwrap_err();
    assert!(
        err.to_string().contains("layer 42") || err.to_string().contains("missing"),
        "{err}"
    );
}
