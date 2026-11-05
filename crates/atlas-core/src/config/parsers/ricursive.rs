// SPDX-License-Identifier: AGPL-3.0-only

//! Ricursive Ripple configuration parser (`model_type = "ricursive"`).
//!
//! Dense PeriNorm transformer: per-head QK RMSNorm, elementwise sigmoid
//! attention gate (`g_proj`), 4:1 sliding:global layers (sliding = RoPE +
//! window, global = NoPE full attention), value residuals from the scaled
//! layer-0 embeddings, and a clamped SwiGLU. Reference:
//! `modeling_ricursive.py` and `inference/local/model.py` in the checkpoint.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};

pub(crate) fn parse_ricursive(raw: &Value) -> Result<ModelConfig> {
    let mut normalized = raw.clone();
    let object = normalized
        .as_object_mut()
        .context("ricursive config.json must be an object")?;

    // `eos_token_id` is [<|im_end|>, <|endoftext|>]; the typed field holds
    // one id, and `finalize_config` collects the full list from `raw`.
    let primary_eos = match raw.get("eos_token_id") {
        Some(Value::Number(n)) => n.as_u64(),
        Some(Value::Array(ids)) => ids.first().and_then(Value::as_u64),
        _ => None,
    }
    .context("ricursive eos_token_id must be an integer or non-empty integer array")?;
    object.insert("eos_token_id".into(), Value::from(primary_eos));

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse ricursive config.json")?;
    ensure!(config.hidden_size > 0, "ricursive hidden_size must be non-zero");
    ensure!(config.head_dim > 0, "ricursive head_dim must be non-zero");
    ensure!(
        config.num_key_value_heads > 0
            && config
                .num_attention_heads
                .is_multiple_of(config.num_key_value_heads),
        "ricursive Q heads must be a non-zero multiple of KV heads"
    );
    ensure!(
        config.layer_types.len() == config.num_hidden_layers
            && config.layer_types.iter().all(|t| matches!(
                t,
                LayerType::SlidingAttention | LayerType::FullAttention
            )),
        "ricursive layer_types must list sliding/full attention for every layer"
    );
    ensure!(
        config.sliding_window > 0,
        "ricursive sliding_window must be non-zero"
    );

    // Dense model: clear the MoE/SSM/MTP fields serde may have defaulted.
    config.num_experts = 0;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 0;
    config.shared_expert_intermediate_size = 0;
    config.linear_num_key_heads = 0;
    config.linear_key_head_dim = 0;
    config.linear_num_value_heads = 0;
    config.linear_value_head_dim = 0;
    config.linear_conv_kernel_dim = 0;
    config.mtp_num_hidden_layers = 0;
    config.num_mtp_modules = 0;
    config.mtp_transformer_layers = 0;

    // Sliding layers rotate the full head_dim (neox); global layers are NoPE,
    // which the loader sets per layer.
    config.partial_rotary_factor = 1.0;
    config.qk_norm_type = "per_head".to_string();
    // The gate is folded into q_proj as the Qwen3.5 [q | gate] per-head
    // interleave at load (see the ricursive weight loader).
    config.attn_gated = raw
        .get("use_gated_attention")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    config.weight_prefix.clear();
    config.embed_scale = raw
        .get("embed_scale")
        .and_then(Value::as_f64)
        .map_or(0.0, |s| s as f32);
    config.ffn_clamp_threshold = raw
        .get("ffn_clamp_threshold")
        .and_then(Value::as_f64)
        .map_or(0.0, |c| c as f32);
    config.value_residual = raw.get("use_valres").and_then(Value::as_bool) == Some(true);
    // Keep the sliding/global pattern for the loader, then list every layer
    // as FullAttention (see `ModelConfig::sliding_layers`).
    config.sliding_layers = config
        .layer_types
        .iter()
        .map(|t| *t == LayerType::SlidingAttention)
        .collect();
    config.layer_types = vec![LayerType::FullAttention; config.num_hidden_layers];

    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
mod tests {
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
        "rms_norm_eps": 1e-06,
        "max_position_embeddings": 2097152,
        "bos_token_id": 100257,
        "eos_token_id": [100265, 100257],
        "pad_token_id": 100277,
        "tie_word_embeddings": false,
        "embed_scale": 71.55417527999327,
        "ffn_clamp_threshold": 20.0,
        "gate_full_precision": true,
        "use_gated_attention": true,
        "use_valres": true
    }"#;

    #[test]
    fn parses_ripple_fields() {
        let raw: serde_json::Value = serde_json::from_str(CONFIG).unwrap();
        let c = super::parse_ricursive(&raw).unwrap();
        assert_eq!(c.model_type, "ricursive");
        assert_eq!(c.num_attention_heads, 80);
        assert_eq!(c.num_key_value_heads, 8);
        assert_eq!(c.head_dim, 128);
        assert_eq!(c.sliding_window, 1024);
        assert!(c.layer_types.iter().all(|t| *t == LayerType::FullAttention));
        assert_eq!(c.sliding_layers, vec![true, true, true, true, false]);
        assert!(c.attn_gated);
        assert!(c.value_residual);
        assert_eq!(c.ffn_clamp_threshold, 20.0);
        assert!((c.embed_scale - 71.554_18).abs() < 1e-3);
        assert_eq!(c.num_experts, 0);
        assert!(!c.tie_word_embeddings);
    }
}
