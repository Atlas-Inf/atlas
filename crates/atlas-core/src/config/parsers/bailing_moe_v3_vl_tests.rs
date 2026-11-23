// SPDX-License-Identifier: AGPL-3.0-only

use crate::config::{LayerType, parse_config};

const EXACT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/test_data/bailing_moe_v3_vl_config.json"
));

fn reference_config() -> String {
    let mut text: serde_json::Value = serde_json::from_str(
        r#"{
        "image_patch_token": 157157, "video_patch_token": 156909,
        "image_start_token": 157158, "video_start_token": 157160,
        "num_hidden_layers": 42, "hidden_size": 2560, "intermediate_size": 6144,
        "eos_token_id": 156895, "pad_token_id": 156892, "first_k_dense_replace": 2,
        "max_position_embeddings": 262144, "moe_intermediate_size": 768,
        "num_experts_per_tok": 8, "num_attention_heads": 32, "q_lora_rank": null,
        "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64,
        "v_head_dim": 128, "num_experts": 512, "num_key_value_heads": 32,
        "rope_theta": 6000000, "rms_norm_eps": 0.000001, "head_dim": 128,
        "vocab_size": 157184, "partial_rotary_factor": 0.5, "router_dtype": "fp32",
        "moe_router_enable_expert_bias": true, "routed_scaling_factor": 2.5,
        "n_group": 8, "topk_group": 4, "use_qk_norm": true, "score_function": "sigmoid",
        "moe_shared_expert_intermediate_size": 768, "layer_group_size": 6,
        "num_kv_heads_for_linear_attn": 0, "group_norm_size": 1, "linear_silu": true,
        "rotary_dim": 64, "use_mla_nope": false, "short_conv_kernel_size": 4,
        "use_nGPT": false, "scale_router_input": false, "value_norm": false,
        "up_proj_norm": false, "gated_attention_proj_granularity_type": "head_wise",
        "mtp_use_kda": false, "no_kda_lora": true, "use_kda_lora": false,
        "kda_safe_gate": true, "kda_lower_bound": -5.0, "norm_topk_prob": true,
        "num_nextn_predict_layers": 1
    }"#,
    )
    .unwrap();
    text["expert_swiglu_limit_list"] = serde_json::Value::Array(
        (0..42)
            .map(|i| serde_json::Value::from(if i >= 35 { 4 } else { 0 }))
            .collect(),
    );
    text["share_expert_swiglu_limit_list"] = serde_json::Value::Array(
        (0..42)
            .map(|i| {
                serde_json::Value::from(if i >= 40 {
                    7
                } else if i >= 34 {
                    5
                } else {
                    0
                })
            })
            .collect(),
    );

    serde_json::json!({
        "model_type": "bailing_moe_v3_vl",
        "architectures": ["BailingMoeV3VLForConditionalGeneration"],
        "torch_dtype": "bfloat16",
        "tie_word_embeddings": false,
        "image_token_id": 157157,
        "video_token_id": 156909,
        "vision_start_token_id": 157158,
        "vision_end_token_id": 157159,
        "mrope_section": [8, 12, 12],
        "text_config": text,
        "vision_config": {
            "model_type": "qwen3_moe_vit",
            "depth": 27,
            "hidden_size": 1152,
            "hidden_act": "gelu_pytorch_tanh",
            "intermediate_size": 4304,
            "num_heads": 16,
            "in_channels": 3,
            "patch_size": 16,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "out_hidden_size": 4096,
            "num_position_embeddings": 2304,
            "initializer_range": 0.02,
            "disable_merger_proj": true
        },
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["model.visual", "linear_proj", "lm_head", "model.word_embeddings", "b_proj", "mlp.gate"]
        }
    })
    .to_string()
}

#[test]
fn parses_reference_hybrid_layout_and_quantization() {
    let config = parse_config(&reference_config()).expect("reference Ling config");
    assert_eq!(config.model_type, "bailing_moe_v3_vl");
    assert_eq!((config.hidden_size, config.vocab_size), (2560, 157184));
    assert_eq!(config.num_hidden_layers, 42);
    assert_eq!(config.mtp_num_hidden_layers, 1);
    assert_eq!(config.num_experts, 512);
    assert_eq!(config.num_experts_per_tok, 8);
    assert_eq!(config.mlp_only_layers, [0, 1]);
    assert_eq!(config.num_attention_layers(), 7);
    assert_eq!(config.num_ssm_layers(), 35);
    assert_eq!(config.layer_type(0), LayerType::LinearAttention);
    assert_eq!(config.layer_type(5), LayerType::FullAttention);
    assert_eq!(config.layer_type(41), LayerType::FullAttention);
    assert_eq!(config.rotary_dim(), 64);
    assert_eq!(config.mrope_section, [8, 12, 12]);
    assert!(config.mrope_interleaved);
    assert!(config.vision.is_some());
    assert_eq!(config.vision.as_ref().unwrap().image_pad_token_id, 157157);
    assert_eq!(config.vision.as_ref().unwrap().video_pad_token_id, 156909);
    assert_eq!(
        config.quantization_config.as_ref().unwrap().quant_method,
        "fp8"
    );
}

#[test]
fn parses_the_exact_upstream_3_2_air_config() {
    let config = parse_config(EXACT).expect("exact Ling 3.2 air config");
    assert_eq!(config.layer_types.len(), 42);
    assert_eq!(config.layer_types[5], LayerType::FullAttention);
    assert_eq!(config.layer_types[41], LayerType::FullAttention);
    assert_eq!(config.mtp_num_hidden_layers, 1);
    assert_eq!(config.vision.as_ref().unwrap().video_start_token_id, 157160);
}

#[test]
fn refuses_unsupported_bailing_layout_variants() {
    let mut raw: serde_json::Value = serde_json::from_str(&reference_config()).unwrap();
    raw.get_mut("text_config").unwrap()["layer_group_size"] = 5.into();
    assert!(parse_config(&raw.to_string()).is_err());

    let text = raw.get_mut("text_config").unwrap();
    text["layer_group_size"] = 6.into();
    text["no_kda_lora"] = false.into();
    assert!(parse_config(&raw.to_string()).is_err());
}
