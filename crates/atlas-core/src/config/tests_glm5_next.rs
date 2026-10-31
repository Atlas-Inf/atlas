// SPDX-License-Identifier: AGPL-3.0-only

//! `glm5_next` (GLM-5.3-Flash) config parsing, against the published
//! `nvidia/GLM-5.3-Flash-NVFP4` `config.json` (revision `09b04e5e`).

use super::*;

const NVIDIA_CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/glm5_next_nvidia_nvfp4_config.json"
));

fn text_config_with(edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>)) -> String {
    let mut raw: serde_json::Value = serde_json::from_str(NVIDIA_CONFIG).unwrap();
    edit(raw["text_config"].as_object_mut().unwrap());
    raw.to_string()
}

#[test]
fn nvidia_checkpoint_maps_hybrid_kda_mla_moe_shape() {
    let cfg = parse_config(NVIDIA_CONFIG).unwrap();
    assert_eq!(cfg.model_type, "glm5_next");
    assert_eq!(cfg.weight_prefix, "model.language_model");
    assert!(cfg.nested_config);
    assert!(!cfg.attn_gated);

    assert_eq!(cfg.hidden_size, 4096);
    assert_eq!(cfg.num_hidden_layers, 45);
    assert_eq!(cfg.num_attention_layers(), 11);
    assert_eq!(cfg.num_ssm_layers(), 34);
    assert_eq!(cfg.vocab_size, 154_880);

    // MLA without RoPE: the Q/K width is the NoPE part alone.
    assert_eq!(cfg.kv_lora_rank, 512);
    assert_eq!(cfg.qk_nope_head_dim, 256);
    assert_eq!(cfg.qk_rope_head_dim, 0);
    assert_eq!(cfg.head_dim, 256);
    assert_eq!(cfg.partial_rotary_factor, 0.0);

    // KDA linear attention.
    assert_eq!(cfg.linear_num_key_heads, 64);
    assert_eq!(cfg.linear_num_value_heads, 64);
    assert_eq!(cfg.linear_key_head_dim, 128);
    assert_eq!(cfg.linear_conv_kernel_dim, 4);
    assert_eq!(cfg.kda_gate_lower_bound, -5.0);

    // MoE: sigmoid noaux_tc routing, one shared expert, three dense layers.
    assert_eq!(cfg.num_experts, 288);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.moe_intermediate_size, 2048);
    assert_eq!(cfg.shared_expert_intermediate_size, 2048);
    assert_eq!(cfg.scoring_func, "sigmoid");
    assert!(cfg.use_routing_bias);
    assert!(cfg.router_logits_f32);
    assert_eq!(cfg.router_logit_bytes(), 4);
    assert_eq!(cfg.mlp_only_layers, vec![0, 1, 2]);

    // mHC and the k-pool semantic indexer.
    assert_eq!(cfg.hc_mult, 4);
    assert_eq!(cfg.hc_sinkhorn_iters, 20);
    assert_eq!(cfg.index_n_heads, 32);
    assert_eq!(cfg.index_head_dim, 128);
    assert_eq!(cfg.index_topk, 2048);
    assert_eq!(cfg.index_kpool, 4);
    assert!(cfg.index_kpool_always_select_tail);

    assert_eq!(cfg.num_mtp_modules, 1);
    let vision = cfg.vision.as_ref().expect("GLM vision config");
    assert!(vision.is_glm5_next);
    assert_eq!(vision.in_channels, 3);
    assert_eq!(vision.out_hidden_size, 4096);
    assert!(vision.image_start_token_id != 0 && vision.image_end_token_id != 0);
    let quant = cfg
        .quantization_config
        .expect("ModelOpt quantization config");
    assert_eq!(quant.quant_algo, "NVFP4");
}

#[test]
fn layer_types_follow_the_checkpoint_not_a_fixed_period() {
    let cfg = parse_config(NVIDIA_CONFIG).unwrap();
    let raw: serde_json::Value = serde_json::from_str(NVIDIA_CONFIG).unwrap();
    let declared = raw["text_config"]["layer_types"].as_array().unwrap();
    for (layer, kind) in declared.iter().enumerate() {
        let expected = match kind.as_str().unwrap() {
            "linear_attention" => LayerType::LinearAttention,
            _ => LayerType::FullAttention,
        };
        assert_eq!(cfg.layer_type(layer), expected, "layer {layer}");
    }
}

#[test]
fn router_logits_are_float32_and_nothing_else() {
    let json = text_config_with(|t| {
        t.remove("moe_router_dtype");
    });
    assert!(parse_config(&json).unwrap().router_logits_f32);
    let json = text_config_with(|t| {
        t["moe_router_dtype"] = "bfloat16".into();
    });
    let err = parse_config(&json).unwrap_err().to_string();
    assert!(err.contains("moe_router_dtype"), "{err}");
}

#[test]
fn rejects_unknown_layer_type() {
    let json = text_config_with(|t| {
        t["layer_types"].as_array_mut().unwrap()[0] = "sliding_attention".into();
    });
    let err = parse_config(&json).unwrap_err().to_string();
    assert!(err.contains("unsupported glm5_next layer type"), "{err}");
}

#[test]
fn rejects_topk_not_divisible_by_kpool() {
    let json = text_config_with(|t| {
        t.insert("index_topk".into(), 2047.into());
    });
    let err = parse_config(&json).unwrap_err().to_string();
    assert!(err.contains("divisible by index_kpool"), "{err}");
}

#[test]
fn rejects_uncompressed_kpool_and_partial_indexers() {
    let json = text_config_with(|t| {
        t.insert("index_kpool_compress".into(), false.into());
    });
    assert!(parse_config(&json).is_err());

    let json = text_config_with(|t| {
        t["indexer_types"].as_array_mut().unwrap()[0] = "none".into();
    });
    let err = parse_config(&json).unwrap_err().to_string();
    assert!(err.contains("full indexer on every layer"), "{err}");
}

#[test]
fn missing_required_field_names_it() {
    let json = text_config_with(|t| {
        t.remove("kv_lora_rank");
    });
    let err = format!("{:#}", parse_config(&json).unwrap_err());
    assert!(err.contains("kv_lora_rank"), "{err}");
}

/// nvidia/GLM-5.3-Flash-FP4-Local (revision `76348ddb`) `config.json`, its
/// `quantized_layers` trimmed to every csf_32x2_v1 entry plus one of each
/// other scheme (its `hf_quant_config.json` is the same object).
const FP4_LOCAL_CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/glm5_next_fp4local_config.json"
));

#[test]
fn fp4_local_declares_csf32_routed_expert_scales() {
    let cfg = parse_config(FP4_LOCAL_CONFIG).unwrap();
    assert_eq!(cfg.model_type, "glm5_next");
    let q = cfg
        .quantization_config
        .expect("ModelOpt quantization config");
    assert_eq!(
        (q.quant_method.as_str(), q.quant_algo.as_str()),
        ("modelopt", "MIXED_PRECISION")
    );
    let spec = |path: &str| q.quantized_layers.get(path).cloned().unwrap_or_default();
    let csf32 = |algo: &str| QuantLayerSpec {
        quant_algo: algo.into(),
        group_size: 16,
        weight_scale_encoding: Some("csf_32x2_v1".into()),
        weight_block_size: vec![],
    };
    // Routed experts of layers 3..=44 ship csf_32x2_v1 scales; the MTP
    // layer's (45) are plain W4A16_NVFP4.
    for layer in 3..=44 {
        let path = format!("model.language_model.layers.{layer}.mlp.experts");
        assert_eq!(spec(&path), csf32("NVFP4"), "{path}");
    }
    let mtp = spec("model.language_model.layers.45.mlp.experts");
    assert_eq!(
        (
            mtp.quant_algo.as_str(),
            mtp.weight_scale_encoding.as_deref().unwrap_or("")
        ),
        ("W4A16_NVFP4", "")
    );
    let encoded = q
        .quantized_layers
        .values()
        .filter(|s| {
            s.weight_scale_encoding
                .as_deref()
                .is_some_and(|e| !e.is_empty())
        })
        .count();
    assert_eq!(
        encoded, 42,
        "only the 42 routed-expert entries are compressed"
    );
    for path in [
        "lm_head",
        "model.language_model.layers.3.mlp.shared_experts.down_proj",
        "model.language_model.layers.45.self_attn.q_a_proj",
    ] {
        let entry = spec(path).weight_scale_encoding;
        assert_eq!(entry.as_deref().unwrap_or(""), "", "{path}");
    }
}
