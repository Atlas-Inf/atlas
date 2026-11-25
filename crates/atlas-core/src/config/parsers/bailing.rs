// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for the nested `bailing_moe_v3_vl` conditional-generation config
//! (Ling-3.2-air NVFP4). The text tower (`text_config`, upstream
//! `model_type = bailing_hybrid`) is a KDA / gated-MLA / MoE hybrid; the
//! vision tower and projector are skipped in this phase (text-only) and do
//! not appear here.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};

/// Bailing family facts with no generic [`ModelConfig`] counterpart,
/// consumed by the spark-model Ling family driver
/// (`model/bailing_c1.rs`): the group-limited router constants and the
/// per-LAYER SwiGLU clamp lists (`expert_swiglu_limit_list` /
/// `share_expert_swiglu_limit_list`, indexed by decoder layer).
#[derive(Clone, Debug)]
pub struct BailingParams {
    /// `n_group` — router groups over the expert axis (Ling: 8).
    pub n_group: usize,
    /// `topk_group` — groups surviving group-limited top-k (Ling: 4).
    pub topk_group: usize,
    /// `moe_router_enable_expert_bias` — the FP32 `mlp.gate.expert_bias`
    /// participates in selection.
    pub expert_bias: bool,
    /// Per-layer routed-expert SwiGLU clamp (0 = none).
    pub routed_swiglu: Vec<i64>,
    /// Per-layer shared-expert SwiGLU clamp (0 = none).
    pub shared_swiglu: Vec<i64>,
}

fn usize_field(text: &Value, name: &str) -> Result<usize> {
    let value = text
        .get(name)
        .and_then(Value::as_u64)
        .with_context(|| format!("bailing text_config missing integer `{name}`"))?;
    ensure!(value > 0, "bailing text_config `{name}` must be non-zero");
    Ok(value as usize)
}

/// `first_k_dense_replace`-style per-layer int list: absent → all zeros.
fn int_list(text: &Value, name: &str, layers: usize) -> Result<Vec<i64>> {
    match text.get(name) {
        None => Ok(vec![0; layers]),
        Some(Value::Array(entries)) => {
            ensure!(
                entries.len() == layers,
                "bailing text_config `{name}` has {} entries, expected {layers}",
                entries.len()
            );
            entries
                .iter()
                .map(|v| {
                    v.as_i64()
                        .with_context(|| format!("bailing `{name}` entries must be integers"))
                })
                .collect()
        }
        Some(_) => anyhow::bail!("bailing text_config `{name}` must be an array"),
    }
}

/// Layer-type layout (`BailingMoeV3DecoderLayer.__init__`): gated MLA at
/// every `(i + 1) % layer_group_size == 0`, or from the last whole group
/// boundary to the end of the stack; KDA everywhere else. The checkpoint
/// index agrees ({5, 11, 17, 23, 29, 35, 41} for the 42-layer stack) — the
/// loader's plan validates the tensors against this layout at load.
fn layer_types(num_layers: usize, group: usize) -> Vec<LayerType> {
    let boundary = num_layers / group * group;
    (0..num_layers)
        .map(|i| {
            if (i + 1).is_multiple_of(group) || i >= boundary {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect()
}

pub fn parse_bailing(raw: &Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("bailing config missing text_config")?;
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "bailing_moe_v3_vl".to_string();
    config.hidden_size = usize_field(text, "hidden_size")?;
    config.num_hidden_layers = usize_field(text, "num_hidden_layers")?;
    config.intermediate_size = usize_field(text, "intermediate_size")?;
    config.vocab_size = usize_field(text, "vocab_size")?;
    config.max_position_embeddings = usize_field(text, "max_position_embeddings")?;
    config.rms_norm_eps = text
        .get("rms_norm_eps")
        .and_then(Value::as_f64)
        .context("bailing text_config missing rms_norm_eps")?;

    // ── attention ──
    // The native context is 262,144 positions: this NVFP4 checkpoint ships
    // NO rope_scaling (YaRN beyond native is not served by this parser).
    ensure!(
        text.get("rope_scaling").is_none(),
        "bailing: rope_scaling is out of scope for the Ling-3.2-air Atlas \\\n             \
         bring-up (native 262,144-position context)"
    );
    config.num_attention_heads = usize_field(text, "num_attention_heads")?;
    // Expanded-MLA projection; the paged cache still stores one compressed
    // latent row (kv_lora + rope) per token — the factory sizes it from
    // kv_lora_rank/qk_rope_head_dim below.
    config.num_key_value_heads = usize_field(text, "num_key_value_heads")?;
    config.head_dim = usize_field(text, "head_dim")?;
    config.kv_lora_rank = usize_field(text, "kv_lora_rank")?;
    config.q_lora_rank = text.get("q_lora_rank").and_then(Value::as_u64).unwrap_or(0) as usize;
    config.qk_nope_head_dim = usize_field(text, "qk_nope_head_dim")?;
    config.qk_rope_head_dim = usize_field(text, "qk_rope_head_dim")?;
    config.v_head_dim = usize_field(text, "v_head_dim")?;
    config.rotary_dim = usize_field(text, "rotary_dim")?;
    config.partial_rotary_factor = text
        .get("partial_rotary_factor")
        .and_then(Value::as_f64)
        .context("bailing text_config missing partial_rotary_factor")?;
    config.rope_theta = text
        .get("rope_theta")
        .and_then(Value::as_f64)
        .context("bailing text_config missing rope_theta")?;
    ensure!(
        text.get("gated_attention_proj_granularity_type")
            .and_then(Value::as_str)
            == Some("head_wise"),
        "bailing: only head_wise gated_attention_proj_granularity_type is \\\n             \
         implemented (the gated-MLA kernels fuse a per-head sigmoid gate)"
    );

    // ── KDA linear attention ──
    // Ling-3.2-air uses the lora-less KDA layout: 32 heads × head_dim 128
    // (state [32, 128, 128] FP32), short conv kernel 4 over the q/k/v
    // planes (3 × 4096 channels), bounded decay gate.
    ensure!(
        text.get("no_kda_lora").and_then(Value::as_bool) == Some(true),
        "bailing: the lora-less KDA layout (no_kda_lora=true) is the only \\\n             \
         one the Ling kernels map"
    );
    let linear_heads = config.num_attention_heads;
    let linear_dim = config.head_dim;
    config.linear_num_key_heads = linear_heads;
    config.linear_num_value_heads = linear_heads;
    config.linear_key_head_dim = linear_dim;
    config.linear_value_head_dim = linear_dim;
    config.linear_conv_kernel_dim = usize_field(text, "short_conv_kernel_size")?;
    config.kda_gate_lower_bound =
        text.get("kda_lower_bound")
            .and_then(Value::as_f64)
            .context("bailing text_config missing kda_lower_bound")? as f32;

    // ── MoE ──
    config.num_experts = usize_field(text, "num_experts")?;
    config.num_experts_per_tok = usize_field(text, "num_experts_per_tok")?;
    config.moe_intermediate_size = usize_field(text, "moe_intermediate_size")?;
    config.shared_expert_intermediate_size =
        usize_field(text, "moe_shared_expert_intermediate_size")?;
    config.norm_topk_prob = text
        .get("norm_topk_prob")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    config.scoring_func = text
        .get("score_function")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    ensure!(
        config.scoring_func == "sigmoid",
        "bailing: only sigmoid routing is implemented (BailingMoeV3Gate)"
    );
    config.routed_scaling_factor = text
        .get("routed_scaling_factor")
        .and_then(Value::as_f64)
        .context("bailing text_config missing routed_scaling_factor")?;
    config.router_logits_f32 = text.get("router_dtype").and_then(Value::as_str) == Some("fp32");
    ensure!(
        config.router_logits_f32,
        "bailing: router_dtype must be fp32 (the reference computes FP32 logits)"
    );
    let first_dense =
        text.get("first_k_dense_replace")
            .and_then(Value::as_u64)
            .context("bailing text_config missing first_k_dense_replace")? as usize;
    config.mlp_only_layers = (0..first_dense).collect();

    let group = usize_field(text, "layer_group_size")?;
    config.layer_types = layer_types(config.num_hidden_layers, group);
    config.full_attention_interval = 0;

    let params = BailingParams {
        n_group: usize_field(text, "n_group")?,
        topk_group: usize_field(text, "topk_group")?,
        expert_bias: text
            .get("moe_router_enable_expert_bias")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        routed_swiglu: int_list(text, "expert_swiglu_limit_list", config.num_hidden_layers)?,
        shared_swiglu: int_list(
            text,
            "share_expert_swiglu_limit_list",
            config.num_hidden_layers,
        )?,
    };
    ensure!(
        params.n_group > 0 && config.num_experts.is_multiple_of(params.n_group),
        "bailing: num_experts must be divisible by n_group"
    );
    ensure!(
        params.topk_group > 0 && params.topk_group <= params.n_group,
        "bailing: topk_group must be in 1..=n_group"
    );
    config.bailing = Some(params);

    // ── tokens / MTP ──
    config.eos_token_id = text
        .get("eos_token_id")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    config.bos_token_id = config.eos_token_id;
    config.tie_word_embeddings = text
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    // One NEXTN layer ships as `model.layers.<num_hidden_layers>.*`; the
    // draft-side MTP path is OUT of this phase, so the loader's MTP hooks
    // return None and the scheduler runs base decode.
    let nextn = text
        .get("num_nextn_predict_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    config.mtp_num_hidden_layers = nextn;
    config.num_mtp_modules = nextn;

    config.attn_gated = false;
    config.nested_config = true;
    config.weight_prefix = "model".to_string();
    // Text-only phase: no vision tower config is parsed; the loader skips
    // `model.visual.*` / `linear_proj.*` tensors at upload (bailing::vision).
    config.vision = None;

    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
#[path = "bailing_tests.rs"]
mod tests;
