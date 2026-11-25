// SPDX-License-Identifier: AGPL-3.0-only

//! `parse_bailing` against a faithful slice of the shipped
//! `primitive-ai/Ling-3.2-air-NVFP4` config.json (42-layer stack; the
//! canonical facts live in the campaign context copy of the checkpoint
//! config). `serde_json::json!` exceeds the crate's recursion limit, so the
//! sample is a string.

use serde_json::{Value, from_str};

fn sample_config() -> Value {
    let routed = format!(
        "[{}]",
        (0..42)
            .map(|i| if i >= 35 { "4" } else { "0" })
            .collect::<Vec<_>>()
            .join(",")
    );
    let shared = format!(
        "[{}]",
        (0..42)
            .map(|i| match i {
                0..=33 => "0",
                34..=39 => "5",
                _ => "7",
            })
            .collect::<Vec<_>>()
            .join(",")
    );
    let text = format!(
        r#"{{
            "model_type": "bailing_hybrid",
            "hidden_size": 2560,
            "num_hidden_layers": 42,
            "intermediate_size": 6144,
            "vocab_size": 157184,
            "max_position_embeddings": 262144,
            "rms_norm_eps": 1e-6,
            "num_attention_heads": 32,
            "num_key_value_heads": 32,
            "head_dim": 128,
            "kv_lora_rank": 512,
            "qk_nope_head_dim": 128,
            "qk_rope_head_dim": 64,
            "v_head_dim": 128,
            "rotary_dim": 64,
            "partial_rotary_factor": 0.5,
            "rope_theta": 6000000,
            "gated_attention_proj_granularity_type": "head_wise",
            "no_kda_lora": true,
            "short_conv_kernel_size": 4,
            "kda_lower_bound": -5.0,
            "num_experts": 512,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 768,
            "moe_shared_expert_intermediate_size": 768,
            "score_function": "sigmoid",
            "norm_topk_prob": true,
            "routed_scaling_factor": 2.5,
            "router_dtype": "fp32",
            "moe_router_enable_expert_bias": true,
            "n_group": 8,
            "topk_group": 4,
            "first_k_dense_replace": 2,
            "layer_group_size": 6,
            "expert_swiglu_limit_list": {routed},
            "share_expert_swiglu_limit_list": {shared},
            "num_nextn_predict_layers": 1,
            "mtp_use_kda": false,
            "eos_token_id": 156895,
            "pad_token_id": 156892,
            "tie_word_embeddings": false,
            "mrope_section": [8, 12, 12]
        }}"#
    );
    from_str(&format!(
        r#"{{"model_type": "bailing_moe_v3_vl", "text_config": {text} }}"#
    ))
    .expect("valid json")
}

fn parse() -> crate::config::ModelConfig {
    let raw = sample_config();
    super::parse_bailing(&raw).expect("parse")
}

#[test]
fn parses_the_ling_hybrid_layout() {
    let config = parse();
    assert_eq!(config.model_type, "bailing_moe_v3_vl");
    assert_eq!(config.hidden_size, 2560);
    assert_eq!(config.num_hidden_layers, 42);
    assert_eq!(config.vocab_size, 157184);
    assert_eq!(config.max_position_embeddings, 262144);
    assert_eq!(config.weight_prefix, "model");
    assert_eq!(config.mlp_only_layers, vec![0, 1]);
    // 7 gated-MLA layers at the group boundaries; 35 KDA layers.
    let mla: Vec<usize> = (0..42)
        .filter(|i| config.layer_type(*i) == crate::config::LayerType::FullAttention)
        .collect();
    assert_eq!(mla, vec![5, 11, 17, 23, 29, 35, 41]);
    assert_eq!(config.num_attention_layers(), 7);
    assert_eq!(config.num_ssm_layers(), 35);
}

#[test]
fn kda_state_geometry_matches_the_recurrence_kernels() {
    let config = parse();
    // Recurrent state [32, 128, 128] FP32 per KDA layer = 2 MiB.
    assert_eq!(config.ssm_h_state_bytes(), 32 * 128 * 128 * 4);
    // Conv window [3 planes x 4096 channels, 4] FP32 per KDA layer.
    assert_eq!(config.ssm_conv_state_bytes(), 3 * 4096 * 4 * 4);
    assert_eq!(config.linear_conv_kernel_dim, 4);
    assert_eq!(config.kda_gate_lower_bound, -5.0);
}

#[test]
fn moe_router_and_swiglu_limits_round_trip() {
    let config = parse();
    assert_eq!(config.num_experts, 512);
    assert_eq!(config.num_experts_per_tok, 8);
    assert!(config.norm_topk_prob);
    assert_eq!(config.routed_scaling_factor, 2.5);
    assert!(config.router_logits_f32);
    let bailing = config.bailing.as_ref().expect("bailing params");
    assert_eq!(bailing.n_group, 8);
    assert_eq!(bailing.topk_group, 4);
    assert!(bailing.expert_bias);
    // Ground truth from the checkpoint lists: routed clamp 4 on layers
    // 35..=41; shared clamp 5 on 34..=39 and 7 on 40..=41.
    for layer in 0..42 {
        assert_eq!(
            bailing.routed_swiglu[layer],
            if layer >= 35 { 4 } else { 0 },
            "routed {layer}"
        );
        let shared_expected = match layer {
            0..=33 => 0,
            34..=39 => 5,
            _ => 7,
        };
        assert_eq!(
            bailing.shared_swiglu[layer], shared_expected,
            "shared {layer}"
        );
    }
}

#[test]
fn mtp_and_tokenizer_facts() {
    let config = parse();
    assert_eq!(config.mtp_num_hidden_layers, 1);
    assert_eq!(config.eos_token_id, 156895);
    let _ = 156892u32;
    assert!(!config.tie_word_embeddings);
}

#[test]
fn rope_scaling_out_of_scope_refused() {
    let mut raw = sample_config();
    raw["text_config"]["rope_scaling"] = from_str(
        r#"{"rope_type": "deepseek_yarn", "factor": 4,
            "original_max_position_embeddings": 262144}"#,
    )
    .expect("json");
    let err = super::parse_bailing(&raw).expect_err("yarn refused");
    assert!(err.to_string().contains("rope_scaling"), "{err}");
}

#[test]
fn non_sigmoid_router_refused() {
    let mut raw = sample_config();
    raw["text_config"]["score_function"] = Value::String("softmax".into());
    let err = super::parse_bailing(&raw).expect_err("softmax refused");
    assert!(err.to_string().contains("sigmoid"), "{err}");
}
