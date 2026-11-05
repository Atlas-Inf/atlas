// SPDX-License-Identifier: AGPL-3.0-only

use crate::config::LayerType;

const CONFIG: &str = r#"{
    "architectures": ["RicursiveForCausalLM"],
    "model_type": "ricursive",
    "hidden_size": 5120,
    "intermediate_size": 40960,
    "num_hidden_layers": 5,
    "num_attention_heads": 80,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 100352,
    "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention",
                    "sliding_attention", "full_attention"],
    "sliding_window": 1024,
    "rope_theta": 10000.0,
    "rms_norm_eps": 0.000001,
    "max_position_embeddings": 2097152,
    "bos_token_id": 100257,
    "eos_token_id": [100265, 100257],
    "pad_token_id": 100277,
    "tie_word_embeddings": false,
    "embed_scale": 71.55417527999327,
    "ffn_clamp_threshold": 20.0,
    "gate_full_precision": true,
    "is_hybrid_swa": true,
    "use_gated_attention": true,
    "use_valres": true
}"#;

#[test]
fn parses_hybrid_swa_gated_valres_layout() {
    let config = crate::config::parse_config(CONFIG).expect("parse ricursive");
    assert_eq!(config.model_type, "ricursive");
    assert_eq!(config.num_hidden_layers, 5);
    assert!(config.layer_types.iter().all(|t| *t == LayerType::FullAttention));
    assert_eq!(config.sliding_attention_layers, [0, 1, 2, 3]);
    assert_eq!(config.sliding_window, 1024);
    assert_eq!(config.embed_scale, 71.554_18_f32);
    assert_eq!(config.eos_token_id, 100265);
    assert!(config.attn_gated);
    assert_eq!(config.qk_norm_type, "per_head");
    assert_eq!(config.rotary_dim(), 128);
    assert_eq!(config.num_attention_layers(), 5);
    assert_eq!(config.num_ssm_layers(), 0);
    assert!(config.skip_lm_head_quantization());
}

#[test]
fn rejects_other_clamp_limits() {
    let invalid = CONFIG.replace("\"ffn_clamp_threshold\": 20.0", "\"ffn_clamp_threshold\": 10.0");
    let error = crate::config::parse_config(&invalid).expect_err("clamp is compiled in");
    assert!(error.to_string().contains("ffn_clamp_threshold"));
}

#[test]
fn rejects_disabled_value_residuals() {
    let invalid = CONFIG.replace("\"use_valres\": true", "\"use_valres\": false");
    let error = crate::config::parse_config(&invalid).expect_err("valres is required");
    assert!(error.to_string().contains("use_valres"));
}
