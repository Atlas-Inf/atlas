// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for the nested `glm5_next` conditional-generation config.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config, parse_glm5_vision_config};

fn usize_field(raw: &Value, name: &str) -> Result<usize> {
    let value = raw
        .get(name)
        .and_then(Value::as_u64)
        .with_context(|| format!("glm5_next text_config missing integer `{name}`"))?;
    ensure!(value > 0, "glm5_next text_config `{name}` must be non-zero");
    Ok(value as usize)
}

pub fn parse_glm5_next(raw: &Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("glm5_next config missing text_config")?;
    let linear = text
        .get("linear_attn_config")
        .context("glm5_next text_config missing linear_attn_config")?;

    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "glm5_next".to_string();
    config.hidden_size = usize_field(text, "hidden_size")?;
    config.num_hidden_layers = usize_field(text, "num_hidden_layers")?;
    config.intermediate_size = usize_field(text, "intermediate_size")?;
    config.vocab_size = usize_field(text, "vocab_size")?;
    config.max_position_embeddings = usize_field(text, "max_position_embeddings")?;
    config.rms_norm_eps = text
        .get("rms_norm_eps")
        .and_then(Value::as_f64)
        .context("glm5_next text_config missing rms_norm_eps")?;

    config.num_attention_heads = usize_field(text, "num_attention_heads")?;
    // Expanded MLA K/V heads. The model factory still stores a single
    // compressed latent row in the paged cache.
    config.num_key_value_heads = usize_field(text, "num_key_value_heads")?;
    config.kv_lora_rank = usize_field(text, "kv_lora_rank")?;
    config.q_lora_rank = usize_field(text, "q_lora_rank")?;
    config.qk_nope_head_dim = usize_field(text, "qk_nope_head_dim")?;
    config.qk_rope_head_dim =
        text.get("qk_rope_head_dim")
            .and_then(Value::as_u64)
            .context("glm5_next text_config missing qk_rope_head_dim")? as usize;
    config.v_head_dim = usize_field(text, "v_head_dim")?;
    // Expanded MLA Q/K head width. The model factory independently sizes the
    // paged cache as kv_lora_rank + rope for every MLA model.
    config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    config.partial_rotary_factor = 0.0;

    let linear_heads = usize_field(linear, "num_heads")?;
    let linear_head_dim = usize_field(linear, "head_dim")?;
    config.linear_num_key_heads = linear_heads;
    config.linear_num_value_heads = linear_heads;
    config.linear_key_head_dim = linear_head_dim;
    config.linear_value_head_dim = linear_head_dim;
    config.linear_conv_kernel_dim = usize_field(linear, "short_conv_kernel_size")?;
    config.kda_gate_lower_bound = linear
        .get("gate_lower_bound")
        .and_then(Value::as_f64)
        .context("glm5_next linear_attn_config missing gate_lower_bound")?
        as f32;

    config.num_experts = usize_field(text, "n_routed_experts")?;
    config.n_routed_experts = config.num_experts;
    config.num_experts_per_tok = usize_field(text, "num_experts_per_tok")?;
    config.moe_intermediate_size = usize_field(text, "moe_intermediate_size")?;
    config.shared_expert_intermediate_size = text
        .get("n_shared_experts")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize
        * config.moe_intermediate_size;
    config.norm_topk_prob = text
        .get("norm_topk_prob")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    config.scoring_func = text
        .get("scoring_func")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    config.use_routing_bias = text.get("topk_method").and_then(Value::as_str) == Some("noaux_tc");
    config.routed_scaling_factor = text
        .get("routed_scaling_factor")
        .and_then(Value::as_f64)
        .unwrap_or(1.0);
    let first_dense = text
        .get("first_k_dense_replace")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    config.mlp_only_layers = (0..first_dense).collect();

    let layer_types = text
        .get("layer_types")
        .and_then(Value::as_array)
        .context("glm5_next text_config missing layer_types")?;
    config.layer_types = layer_types
        .iter()
        .map(|kind| match kind.as_str() {
            Some("linear_attention") => Ok(LayerType::LinearAttention),
            Some("deepseek_sparse_attention") => Ok(LayerType::FullAttention),
            Some(other) => anyhow::bail!("unsupported glm5_next layer type `{other}`"),
            None => anyhow::bail!("glm5_next layer type must be a string"),
        })
        .collect::<Result<Vec<_>>>()?;

    config.hc_mult = text.get("hc_mult").and_then(Value::as_u64).unwrap_or(0) as usize;
    config.hc_sinkhorn_iters = text
        .get("hc_sinkhorn_iters")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    config.hc_eps = text.get("hc_eps").and_then(Value::as_f64).unwrap_or(0.0) as f32;
    config.index_n_heads = usize_field(text, "index_n_heads")?;
    config.index_head_dim = usize_field(text, "index_head_dim")?;
    config.index_topk = usize_field(text, "index_topk")?;
    config.index_kpool = usize_field(text, "index_kpool")?;
    ensure!(
        config.index_topk.is_multiple_of(config.index_kpool),
        "glm5_next index_topk must be divisible by index_kpool"
    );
    ensure!(
        text.get("index_kpool_compress").and_then(Value::as_bool) == Some(true),
        "glm5_next Atlas support requires index_kpool_compress=true"
    );
    config.index_kpool_always_select_tail = text
        .get("index_kpool_always_select_tail")
        .and_then(Value::as_bool)
        .context("glm5_next text_config missing boolean `index_kpool_always_select_tail`")?;
    let indexer_types = text
        .get("indexer_types")
        .and_then(Value::as_array)
        .context("glm5_next text_config missing indexer_types")?;
    ensure!(
        indexer_types.len() == config.num_hidden_layers,
        "glm5_next indexer_types length must equal num_hidden_layers"
    );
    ensure!(
        indexer_types
            .iter()
            .all(|kind| kind.as_str() == Some("full")),
        "glm5_next Atlas support currently requires a full indexer on every layer"
    );

    config.attn_gated = false;
    config.nested_config = true;
    config.weight_prefix = "model.language_model".to_string();
    // GLM-5 stores each next-token predictor as a full decoder layer appended
    // after the target layers (rather than under Atlas' generic `mtp.*`
    // namespace).  The model-specific loader consumes those layers when
    // `--speculative` is requested; ordinary decode continues to ignore them.
    config.num_mtp_modules = text
        .get("num_nextn_predict_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    config.mtp_num_hidden_layers = config.num_mtp_modules;
    config.vision = parse_glm5_vision_config(raw, text)?;
    finalize_config(&mut config, raw)?;
    Ok(config)
}
