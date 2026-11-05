// SPDX-License-Identifier: AGPL-3.0-only

//! Ricursive Ripple configuration parser (`model_type: ricursive`).
//!
//! Dense 80-layer transformer: four sliding-window RoPE layers then one
//! full-attention NoPE layer, repeated. Gemma-style pre/post sublayer norms
//! with plain `x * w` weights, per-head Q/K RMSNorm, an elementwise sigmoid
//! output gate (`g_proj`, fused into `q_proj` at load), value residuals from
//! the scaled layer-0 embedding, and a clamped SwiGLU.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};

/// The SwiGLU clamp the shipped `moe_silu_mul` shadow hardcodes
/// (`kernels/gb10/ricursive-ripple-65b/nvfp4/moe_silu_mul.cu`). The kernel has
/// no argument for it, so a checkpoint declaring any other limit must fail
/// here rather than run with the wrong activation.
pub const RICURSIVE_FFN_CLAMP: f64 = 20.0;

fn required<'a>(raw: &'a Value, key: &str) -> Result<&'a Value> {
    raw.get(key)
        .with_context(|| format!("ricursive config missing required field `{key}`"))
}

fn required_bool(raw: &Value, key: &str) -> Result<bool> {
    required(raw, key)?
        .as_bool()
        .with_context(|| format!("ricursive config field `{key}` must be a boolean"))
}

fn required_f64(raw: &Value, key: &str) -> Result<f64> {
    required(raw, key)?
        .as_f64()
        .with_context(|| format!("ricursive config field `{key}` must be numeric"))
}

pub(crate) fn parse_ricursive(raw: &Value) -> Result<ModelConfig> {
    let mut normalized = raw.clone();
    let object = normalized
        .as_object_mut()
        .context("ricursive config.json must be an object")?;
    let primary_eos = match required(raw, "eos_token_id")? {
        Value::Number(n) => n.as_u64(),
        Value::Array(ids) => ids.first().and_then(Value::as_u64),
        _ => None,
    }
    .context("ricursive eos_token_id must be an integer or non-empty integer array")?;
    object.insert("eos_token_id".into(), Value::from(primary_eos));

    // Every architectural switch the loader implements is required to be on;
    // the off variants are different models this port does not implement.
    for key in ["use_gated_attention", "use_valres", "is_hybrid_swa"] {
        ensure!(
            required_bool(raw, key)?,
            "ricursive `{key}` must be true (only the gated/value-residual/hybrid-SWA layout is implemented)"
        );
    }
    ensure!(
        !required_bool(raw, "tie_word_embeddings")?,
        "ricursive tie_word_embeddings must be false"
    );
    let clamp = required_f64(raw, "ffn_clamp_threshold")?;
    ensure!(
        clamp == RICURSIVE_FFN_CLAMP,
        "ricursive ffn_clamp_threshold {clamp} != {RICURSIVE_FFN_CLAMP} compiled into the SwiGLU kernel"
    );

    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse ricursive config.json")?;
    ensure!(
        config.layer_types.len() == config.num_hidden_layers,
        "ricursive layer_types must list every layer"
    );
    ensure!(
        config.layer_types.iter().all(|t| matches!(
            t,
            LayerType::SlidingAttention | LayerType::FullAttention
        )),
        "ricursive layer_types must be sliding_attention or full_attention"
    );
    ensure!(
        config.sliding_window > 0,
        "ricursive sliding_window must be non-zero"
    );
    ensure!(
        config.head_dim > 0 && config.num_key_value_heads > 0,
        "ricursive head_dim and num_key_value_heads must be non-zero"
    );
    ensure!(
        config
            .num_attention_heads
            .is_multiple_of(config.num_key_value_heads),
        "ricursive num_attention_heads must be divisible by num_key_value_heads"
    );

    config.embed_scale = required_f64(raw, "embed_scale")? as f32;
    config.final_logit_softcapping = 0.0;
    config.partial_rotary_factor = 1.0;
    config.qk_norm_type = "per_head".to_string();
    // g_proj is interleaved into q_proj per head at load: [Q_h | G_h].
    config.attn_gated = true;
    config.weight_prefix.clear();
    config.mtp_num_hidden_layers = 0;
    config.num_mtp_modules = 0;
    config.mtp_transformer_layers = 0;

    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
#[path = "ricursive_tests.rs"]
mod tests;
