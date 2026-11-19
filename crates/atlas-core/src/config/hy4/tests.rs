// SPDX-License-Identifier: AGPL-3.0-only

use serde_json::{Value, json};

use super::super::{LayerType, parse_config};

const CONFIG: &str = include_str!("../fixtures/hy4/config.json");
const INDEX: &str = include_str!("../fixtures/hy4/index-contract.json");
const CONTRACT: &str = include_str!("../fixtures/hy4/tensor-contract.json");

fn edited(edit: impl FnOnce(&mut Value)) -> anyhow::Result<super::super::ModelConfig> {
    let mut v: Value = serde_json::from_str(CONFIG).unwrap();
    edit(&mut v["text_config"]);
    parse_config(&v.to_string())
}

#[test]
fn checkpoint_maps_nested_text_without_loading_vision_or_drafter() {
    let c = parse_config(CONFIG).unwrap();
    let h = c.hy4.as_ref().unwrap();
    assert_eq!(c.model_type, "hy_v4_flash");
    assert!(c.nested_config);
    assert_eq!(c.weight_prefix, "model");
    assert!(c.vision.is_none());
    assert_eq!(
        (c.hidden_size, c.num_hidden_layers, c.vocab_size),
        (3072, 42, 120832)
    );
    assert_eq!(
        (c.q_lora_rank, c.kv_lora_rank, c.head_dim),
        (1024, 512, 256)
    );
    assert_eq!(
        (c.qk_nope_head_dim, c.qk_rope_head_dim, c.v_head_dim),
        (192, 64, 256)
    );
    assert_eq!(c.mlp_only_layers, [0]);
    assert_eq!((c.num_experts, c.num_experts_per_tok), (512, 16));
    assert_eq!(c.shared_expert_intermediate_size, 1536);
    assert_eq!(h.routed_expert_hidden_size, 1536);
    assert!(c.norm_topk_prob && c.use_routing_bias && c.router_logits_f32);
    assert_eq!(c.scoring_func, "sigmoid");
    assert_eq!(c.routed_scaling_factor, 3.998);
    assert_eq!(h.engram_layer_ids, [2, 12, 22, 32]);
    assert_eq!((c.hc_mult, c.hc_sinkhorn_iters), (4, 0));
    assert_eq!(h.hc_magnitude, 2.0);
    assert_eq!(c.hc_eps, 1e-6);
    assert_eq!(c.rms_norm_eps, 1e-5);
    assert_eq!(c.sliding_window, 512);
    assert_eq!(h.rope_parameters.deepseek_sparse_attention.rope_theta, 1e7);
    assert_eq!(h.rope_parameters.sliding_attention.rope_theta, 10000.0);
    assert_eq!(c.mrope_section, [12, 10, 10]);
    let dsa: Vec<_> = c
        .layer_types
        .iter()
        .enumerate()
        .filter_map(|(i, t)| (*t == LayerType::FullAttention).then_some(i))
        .collect();
    assert_eq!(dsa, [0, 1, 5, 9, 13, 17, 21, 25, 29, 33, 37, 41]);
    assert_eq!(c.num_attention_heads_per_layer[0], 32);
    assert_eq!(c.num_attention_heads_per_layer[2], 64);
}

#[test]
fn flat_text_config_is_also_supported() {
    let raw: Value = serde_json::from_str(CONFIG).unwrap();
    let c = parse_config(&raw["text_config"].to_string()).unwrap();
    assert!(!c.nested_config);
    assert!(c.hy4.is_some());
}

#[test]
fn dfly_mask_is_checkpoint_metadata_without_a_fallback() {
    let mut raw: Value = serde_json::from_str(CONFIG).unwrap();
    for mask in [0, 17, 120009] {
        raw["dfly_mask_token_id"] = json!(mask);
        let c = parse_config(&raw.to_string()).unwrap();
        assert_eq!(c.hy4.unwrap().dfly_mask_token_id().unwrap(), mask);
    }
    raw.as_object_mut().unwrap().remove("dfly_mask_token_id");
    let c = parse_config(&raw.to_string()).unwrap();
    assert!(c.hy4.unwrap().dfly_mask_token_id().is_err());
    for mask in [
        json!(-1),
        json!(120832),
        json!(4294967296u64),
        json!(1.5),
        json!("120009"),
    ] {
        raw["dfly_mask_token_id"] = mask;
        assert!(parse_config(&raw.to_string()).is_err());
    }
}

#[test]
fn null_dfly_mask_is_absent_until_dfly_is_enabled() {
    let mut raw: Value = serde_json::from_str(CONFIG).unwrap();
    raw["dfly_mask_token_id"] = Value::Null;
    let config = parse_config(&raw.to_string()).unwrap();
    let hy = config.hy4.unwrap();
    assert_eq!(hy.dfly_mask_token_id, None);
    assert!(hy.dfly_mask_token_id().is_err());
}

#[test]
fn rejects_missing_required_values_instead_of_inheriting_other_models() {
    for key in [
        "hidden_size",
        "num_hidden_layers",
        "layer_types",
        "swa_pattern",
        "mlp_layer_types",
        "hc_mult",
        "hc_eps",
        "hc_magnitude",
        "enable_ihc",
        "engram_layer_ids",
        "engram_seed",
        "rope_parameters",
        "n_routed_experts",
        "norm_topk_prob",
        "routed_scaling_factor",
        "moe_router_use_sigmoid",
        "num_experts_per_tok",
        "gated_mla",
        "sliding_window",
        "qk_head_dim",
    ] {
        assert!(
            edited(|v| {
                v.as_object_mut().unwrap().remove(key);
            })
            .is_err(),
            "{key}"
        );
    }
}

#[test]
fn rejects_empty_malformed_and_inconsistent_layer_maps() {
    for field in ["swa_pattern", "layer_types", "mlp_layer_types"] {
        assert!(edited(|v| v[field] = json!([])).is_err(), "{field}");
    }
    assert!(edited(|v| v["swa_pattern"][0] = json!(1)).is_err());
    assert!(edited(|v| v["swa_pattern"][0] = json!(2)).is_err());
    assert!(edited(|v| v["layer_types"][0] = json!("full_attention")).is_err());
    assert!(edited(|v| v["mlp_layer_types"][0] = json!("sparse")).is_err());
    assert!(edited(|v| v["first_k_dense_replace"] = json!(43)).is_err());
}

#[test]
fn rejects_invalid_hc_rope_moe_and_engram_contracts() {
    for (key, value) in [
        ("enable_ihc", json!(false)),
        ("hc_mult", json!(0)),
        ("hc_eps", json!(-1)),
        ("hc_magnitude", json!(0)),
        ("num_experts_per_tok", json!(513)),
        ("routed_expert_hidden_size", json!(0)),
        ("swiglu_limit", json!(-1)),
        ("engram_layer_ids", json!([0])),
        ("engram_layer_ids", json!([2, 2])),
        ("engram_layer_ids", json!([43])),
        ("engram_n_head_per_ngram", json!(0)),
        ("compressed_indexer_candidate_topk", json!(2047)),
        ("qk_head_dim", json!(64)),
        ("moe_router_use_sigmoid", json!(false)),
    ] {
        assert!(edited(|v| v[key] = value).is_err(), "{key}");
    }
    assert!(
        edited(|v| v["rope_parameters"]["sliding_attention"]["mrope_section"] = json!([1, 1, 1]))
            .is_err()
    );
    assert!(
        edited(|v| v["rope_parameters"]["deepseek_sparse_attention"]["rope_theta"] = json!(0))
            .is_err()
    );
    assert!(parse_config("{}").is_err());
    assert!(parse_config(r#"{"model_type":"hy_v4_flash","text_config":null}"#).is_err());
}

#[test]
fn explicit_empty_engram_is_supported_without_implicit_tables() {
    let c = edited(|v| {
        v["engram_layer_ids"] = json!([]);
        v["engram_layer_head_vocab_sizes"] = json!({});
    })
    .unwrap();
    assert!(c.hy4.unwrap().engram_layer_ids.is_empty());
}

#[test]
fn tensor_contract_covers_indexed_representatives_and_all_expert_counts() {
    let index: Value = serde_json::from_str(INDEX).unwrap();
    let contract: Value = serde_json::from_str(CONTRACT).unwrap();
    let tensors = contract["tensors"].as_object().unwrap();
    assert_eq!(
        tensors.len(),
        index["representative_names"].as_array().unwrap().len()
    );
    for name in index["representative_names"].as_array().unwrap() {
        let name = name.as_str().unwrap();
        let tensor = &tensors[name];
        assert!(
            ["U8", "BF16", "F32", "F8_E4M3", "I32"].contains(&tensor["dtype"].as_str().unwrap())
        );
        if let Some(shape) = tensor["shape"].as_array() {
            assert!(shape.iter().all(|n| n.as_u64().unwrap() > 0));
            if shape.is_empty() {
                assert!(name.ends_with(".input_scale") || name.ends_with(".weight_scale_2"));
                assert_eq!(tensor["dtype"], "F32");
            }
        } else {
            assert!(tensor["shape_note"].is_string(), "{name}");
        }
    }
    for i in 0..42 {
        assert_eq!(
            index["layer_expert_projection_count"][i.to_string()],
            if i == 0 { 0 } else { 1536 }
        );
    }
    assert_eq!(
        tensors["model.layers.0.self_attn.q_b_proj.weight"]["shape"],
        json!([8192, 1024])
    );
    assert_eq!(
        tensors["model.layers.2.self_attn.q_b_proj.weight"]["shape"],
        json!([16384, 1024])
    );
    assert_eq!(
        tensors["model.layers.1.mlp.experts.0.gate_proj.weight"]["shape"],
        json!([1536, 768])
    );
    assert_eq!(
        tensors["model.layers.1.mlp.experts.0.gate_proj.weight_scale"]["shape"],
        json!([1536, 96])
    );
    assert_eq!(
        tensors["model.layers.1.mlp.routed_expert_down_proj.weight"]["shape"],
        json!([1536, 3072])
    );
    assert_eq!(
        tensors["model.layers.1.engram_layer.multi_head_embedding.embedding.weight"]["shape"],
        json!([164994380, 96])
    );
}

#[test]
fn contract_matches_read_only_header_observations_and_marks_unobserved_tensors() {
    let index: Value = serde_json::from_str(INDEX).unwrap();
    let contract: Value = serde_json::from_str(CONTRACT).unwrap();
    let observations: Value =
        serde_json::from_str(include_str!("../fixtures/hy4/header-observations.json")).unwrap();
    let mut seen = std::collections::BTreeSet::new();
    for (shard, tensors) in observations["headers"].as_object().unwrap() {
        for (name, header) in tensors.as_object().unwrap() {
            assert!(seen.insert(name.as_str()));
            assert_eq!(index["representative_shards"][name], *shard);
            assert_eq!(contract["tensors"][name], *header, "{name}");
        }
    }
    assert_eq!(seen.len(), 208);
    let unverified = contract["provenance"]["unverified_names"]
        .as_array()
        .unwrap();
    assert_eq!(unverified.len(), 5);
    for name in contract["tensors"].as_object().unwrap().keys() {
        assert_ne!(
            seen.contains(name.as_str()),
            unverified.contains(&json!(name)),
            "{name}"
        );
    }
    let tensors = &contract["tensors"];
    assert_eq!(
        tensors["model.layers.0.hc_attn_layer.hc_pre.hc_fn"]["dtype"],
        "F32"
    );
    assert_eq!(
        tensors["model.layers.2.self_attn.learnable_sink_param"]["dtype"],
        "F32"
    );
    assert_eq!(
        tensors["model.layers.1.mlp.gate.e_score_correction_bias"]["dtype"],
        "F32"
    );
    assert_eq!(
        tensors["model.layers.1.mlp.experts.0.gate_proj.input_scale"]["shape"],
        json!([])
    );
    assert_eq!(
        tensors["model.layers.1.engram_layer.lookup_table"]["shape"],
        json!([120818])
    );
}
