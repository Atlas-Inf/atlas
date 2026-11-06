// SPDX-License-Identifier: AGPL-3.0-only

use super::super::super::{LayerType, parse_config};

const CONFIG: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../test_data/mimo_v2_6_flash_config.json"
));

#[test]
fn parses_published_mimo_v2_6_flash() {
    let c = parse_config(CONFIG).expect("parse");
    assert_eq!(c.model_type, "mimo_v2");
    assert_eq!(c.num_hidden_layers, 48);
    assert_eq!(c.hidden_size, 4096);
    assert_eq!(c.num_attention_heads, 64);
    assert_eq!(c.num_key_value_heads, 8);
    assert_eq!(c.full_num_key_value_heads, 4);
    assert_eq!((c.head_dim, c.v_head_dim, c.rotary_dim), (192, 128, 64));
    assert_eq!(c.rope_theta, 1.0e7);
    assert_eq!(c.swa_rope_theta, 1.0e4);
    assert!((c.attention_value_scale - 0.707).abs() < 1e-9);
    assert_eq!(c.sliding_window, 128);
    assert_eq!(c.rms_norm_eps, 1e-6);
    assert_eq!(c.num_experts, 256);
    assert_eq!(c.num_experts_per_tok, 8);
    assert_eq!(c.moe_intermediate_size, 2048);
    assert_eq!(c.intermediate_size, 16384);
    assert_eq!(c.shared_expert_intermediate_size, 0);
    assert_eq!(c.scoring_func, "sigmoid");
    assert!(c.use_routing_bias && c.norm_topk_prob);
    assert_eq!(c.routed_scaling_factor, 1.0);
    assert_eq!(c.mlp_only_layers, vec![0]);
    assert_eq!(c.vocab_size, 152576);
    assert_eq!(c.eos_token_id, 151645);
    assert_eq!(c.num_mtp_modules, 3);
    assert_eq!(c.mtp_num_hidden_layers, 3);
    assert!(c.vision.is_none());
    let full: Vec<usize> = c
        .layer_types
        .iter()
        .enumerate()
        .filter(|(_, t)| **t == LayerType::FullAttention)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(full, vec![0, 5, 11, 17, 23, 29, 35, 41, 47]);
    assert_eq!(c.num_attention_layers(), 48);
    assert!(c.quantization_config.is_some());
}

#[test]
fn rejects_unsupported_head_dims() {
    let mut raw: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
    raw["head_dim"] = 128.into();
    raw["swa_head_dim"] = 128.into();
    assert!(parse_config(&raw.to_string()).is_err());
}
