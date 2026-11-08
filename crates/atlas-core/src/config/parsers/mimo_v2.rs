// SPDX-License-Identifier: AGPL-3.0-only

//! Xiaomi MiMo-V2 (MiMo-V2.6-Flash) config parser.
//!
//! Text backbone only; the vision / audio towers are ignored.
//!   * 48 layers, `hybrid_layer_pattern`: 0 = full attention, 1 = sliding
//!     window (128). Full layers use 4 KV heads and RoPE theta 1e7, SWA layers
//!     use 8 KV heads and theta 1e4.
//!   * Q/K head dim 192, V head dim 128, partial RoPE on the first 64 dims.
//!   * `moe_layer_freq`: 0 = dense FFN (layer 0), 1 = 256-expert top-8
//!     sigmoid MoE with `e_score_correction_bias`, no shared expert.
//!   * `attention_value_scale` multiplies V before attention.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};

/// Keys removed before the flat serde pass: nested towers and objects whose
/// shape differs from the `ModelConfig` fields of the same name.
const DROPPED_KEYS: &[&str] = &[
    "audio_config",
    "vision_config",
    "processor_config",
    "rope_parameters",
    "auto_map",
    "architectures",
    "hybrid_layer_pattern",
    "moe_layer_freq",
];

pub(crate) fn parse_mimo_v2(raw: &Value) -> Result<ModelConfig> {
    let obj = raw
        .as_object()
        .context("mimo_v2 config must be an object")?;
    let mut flat = obj.clone();
    flat.retain(|k, v| !v.is_null() && !DROPPED_KEYS.contains(&k.as_str()));
    let mut config: ModelConfig = serde_json::from_value(Value::Object(flat))
        .context("Failed to parse mimo_v2 config.json")?;
    config.model_type = "mimo_v2".to_string();
    config.weight_prefix = "model".to_string();
    config.vision = None;

    let get_u = |k: &str| raw.get(k).and_then(Value::as_u64).map(|v| v as usize);
    let get_f = |k: &str| raw.get(k).and_then(Value::as_f64);

    if let Some(eps) = get_f("layernorm_epsilon") {
        config.rms_norm_eps = eps;
    }

    // ── Attention ──
    let full_kv = config.num_key_value_heads;
    let swa_kv = get_u("swa_num_key_value_heads").unwrap_or(full_kv);
    let swa_q = get_u("swa_num_attention_heads").unwrap_or(config.num_attention_heads);
    ensure!(
        swa_q == config.num_attention_heads,
        "mimo_v2: swa_num_attention_heads ({swa_q}) != num_attention_heads ({}) is unsupported",
        config.num_attention_heads
    );
    for (k, expect) in [
        ("swa_head_dim", config.head_dim),
        ("swa_v_head_dim", config.v_head_dim),
    ] {
        if let Some(v) = get_u(k) {
            ensure!(
                v == expect,
                "mimo_v2: {k}={v} differs from the full-attention value {expect}; unsupported"
            );
        }
    }
    ensure!(
        config.head_dim == 192 && config.v_head_dim == 128,
        "mimo_v2: kernels are built for head_dim=192 / v_head_dim=128, got {}/{}",
        config.head_dim,
        config.v_head_dim
    );
    // Buffers size off `num_key_value_heads`; keep the max and record the
    // full-attention count separately for per-layer overrides.
    config.full_num_key_value_heads = full_kv;
    config.num_key_value_heads = full_kv.max(swa_kv);
    config.swa_rope_theta = get_f("swa_rope_theta").unwrap_or(config.rope_theta);
    config.attention_value_scale = get_f("attention_value_scale").unwrap_or(1.0);
    // HF: int(head_dim * partial_rotary_factor) → 64 for 192 × 0.334.
    config.rotary_dim = (config.head_dim as f64 * config.partial_rotary_factor) as usize;
    if let Some(w) = get_u("sliding_window").or_else(|| get_u("sliding_window_size")) {
        config.sliding_window = w as u32;
    }

    let pattern = raw
        .get("hybrid_layer_pattern")
        .and_then(Value::as_array)
        .context("mimo_v2 config missing hybrid_layer_pattern")?;
    config.layer_types = pattern
        .iter()
        .map(|v| match v.as_u64() {
            Some(0) => Ok(LayerType::FullAttention),
            Some(1) => Ok(LayerType::SlidingAttention),
            other => anyhow::bail!("mimo_v2: unexpected hybrid_layer_pattern entry {other:?}"),
        })
        .collect::<Result<_>>()?;

    // ── MoE ──
    config.num_experts = get_u("n_routed_experts").unwrap_or(config.num_experts);
    config.shared_expert_intermediate_size = 0;
    config.scoring_func = raw
        .get("scoring_func")
        .and_then(Value::as_str)
        .unwrap_or("sigmoid")
        .to_string();
    config.use_routing_bias = raw.get("topk_method").and_then(Value::as_str) == Some("noaux_tc");
    config.routed_scaling_factor = get_f("routed_scaling_factor").unwrap_or(1.0);
    for (k, max) in [("n_group", 1), ("topk_group", 1)] {
        let v = get_u(k).unwrap_or(1);
        ensure!(v <= max, "mimo_v2: grouped routing {k}={v} is unsupported");
    }
    if let Some(freq) = raw.get("moe_layer_freq").and_then(Value::as_array) {
        config.mlp_only_layers = freq
            .iter()
            .enumerate()
            .filter(|(_, v)| v.as_u64() == Some(0))
            .map(|(i, _)| i)
            .collect();
    }

    // ── MTP: three single-layer draft modules in model_mtp.safetensors ──
    let mtp = get_u("num_nextn_predict_layers").unwrap_or(0);
    config.num_mtp_modules = mtp;
    config.mtp_num_hidden_layers = mtp;
    config.mtp_transformer_layers = 1;

    // ── Experimental top-k override (quality-unvalidated; opt-in) ──
    // ATLAS_MOE_TOPK=<k> (1..=trained top_k) reduces the routed-expert count
    // per token. Expert weight bytes — and decode time — scale ~linearly
    // with k, but the model was trained at the checkpoint's top_k, so any
    // reduction is a quality experiment, never a default.
    if let Ok(v) = std::env::var("ATLAS_MOE_TOPK") {
        let k = v
            .parse::<usize>()
            .context("ATLAS_MOE_TOPK must be an integer")?;
        ensure!(
            k >= 1 && k <= config.num_experts_per_tok,
            "ATLAS_MOE_TOPK={k} out of range 1..={}",
            config.num_experts_per_tok
        );
        config.num_experts_per_tok = k;
    }

    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
#[path = "mimo_v2_tests.rs"]
mod tests;
