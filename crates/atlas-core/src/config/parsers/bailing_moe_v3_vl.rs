// SPDX-License-Identifier: AGPL-3.0-only

//! Configuration boundary for Ling 3.2 air (`bailing_moe_v3_vl`).

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, VisionConfig, finalize_config};

fn field<'a>(raw: &'a Value, key: &str) -> Result<&'a Value> {
    raw.get(key)
        .with_context(|| format!("bailing_moe_v3_vl config missing `{key}`"))
}

fn usize_field(raw: &Value, key: &str) -> Result<usize> {
    field(raw, key)?
        .as_u64()
        .map(|n| n as usize)
        .with_context(|| format!("bailing_moe_v3_vl `{key}` must be an unsigned integer"))
}

fn f64_field(raw: &Value, key: &str) -> Result<f64> {
    field(raw, key)?
        .as_f64()
        .with_context(|| format!("bailing_moe_v3_vl `{key}` must be numeric"))
}

fn bool_field(raw: &Value, key: &str) -> Result<bool> {
    field(raw, key)?
        .as_bool()
        .with_context(|| format!("bailing_moe_v3_vl `{key}` must be boolean"))
}

fn string_field<'a>(raw: &'a Value, key: &str) -> Result<&'a str> {
    field(raw, key)?
        .as_str()
        .with_context(|| format!("bailing_moe_v3_vl `{key}` must be a string"))
}

fn nullable_usize(raw: &Value, key: &str) -> Result<Option<usize>> {
    let value = field(raw, key)?;
    if value.is_null() {
        return Ok(None);
    }
    Ok(Some(usize_field(raw, key)?))
}

fn usize_list(raw: &Value, key: &str) -> Result<Vec<usize>> {
    field(raw, key)?
        .as_array()
        .with_context(|| format!("bailing_moe_v3_vl `{key}` must be an array"))?
        .iter()
        .map(|value| {
            value.as_u64().map(|n| n as usize).with_context(|| {
                format!("bailing_moe_v3_vl `{key}` entries must be unsigned integers")
            })
        })
        .collect()
}

fn top_u32(raw: &Value, key: &str) -> Result<u32> {
    let value = usize_field(raw, key)?;
    u32::try_from(value).with_context(|| format!("bailing_moe_v3_vl `{key}` exceeds u32"))
}

fn expect_reference_lists(text: &Value, layers: usize) -> Result<()> {
    let expert = usize_list(text, "expert_swiglu_limit_list")?;
    let shared = usize_list(text, "share_expert_swiglu_limit_list")?;
    ensure!(
        expert.len() == layers && shared.len() == layers,
        "bailing_moe_v3_vl SwiGLU limit lists must have {layers} entries"
    );
    ensure!(
        expert
            .iter()
            .enumerate()
            .all(|(i, &value)| value == if i >= layers - 7 { 4 } else { 0 }),
        "bailing_moe_v3_vl expert_swiglu_limit_list does not match Ling 3.2"
    );
    ensure!(
        shared.iter().enumerate().all(|(i, &value)| {
            value
                == if i >= layers - 2 {
                    7
                } else if i >= layers - 8 {
                    5
                } else {
                    0
                }
        }),
        "bailing_moe_v3_vl share_expert_swiglu_limit_list does not match Ling 3.2"
    );
    Ok(())
}

fn parse_mrope(raw: &Value) -> Result<[usize; 3]> {
    let sections = field(raw, "mrope_section")?
        .as_array()
        .context("bailing_moe_v3_vl mrope_section must be an array")?;
    ensure!(
        sections.len() == 3,
        "bailing_moe_v3_vl mrope_section must contain [T, H, W]"
    );
    let mut out = [0; 3];
    for (i, value) in sections.iter().enumerate() {
        out[i] = value
            .as_u64()
            .map(|n| n as usize)
            .with_context(|| format!("bailing_moe_v3_vl mrope_section[{i}] must be unsigned"))?;
        ensure!(
            out[i] > 0,
            "bailing_moe_v3_vl mrope sections must be non-zero"
        );
    }
    Ok(out)
}

fn parse_vision(raw: &Value, text: &Value) -> Result<VisionConfig> {
    let mut vision = super::vision::parse_vision_config(raw)
        .context("bailing_moe_v3_vl config missing vision_config")?;
    ensure!(
        vision.depth == 27
            && vision.hidden_size == 1152
            && vision.num_heads == 16
            && vision.patch_size == 16
            && vision.temporal_patch_size == 2
            && vision.spatial_merge_size == 2
            && vision.intermediate_size == 4304
            && vision.out_hidden_size == 4096,
        "bailing_moe_v3_vl vision_config is not the Ling 3.2 Qwen3-VL tower"
    );
    vision.image_start_token_id = top_u32(raw, "vision_start_token_id")?;
    vision.image_end_token_id = top_u32(raw, "vision_end_token_id")?;
    vision.video_start_token_id = top_u32(text, "video_start_token")?;
    vision.video_end_token_id = vision.image_end_token_id;
    vision.is_qwen3vl_video = true;
    Ok(vision)
}

pub(crate) fn parse_bailing_moe_v3_vl(raw: &Value) -> Result<ModelConfig> {
    let text = field(raw, "text_config")?;
    let layers = usize_field(text, "num_hidden_layers")?;
    let group = usize_field(text, "layer_group_size")?;
    let first_dense = usize_field(text, "first_k_dense_replace")?;
    ensure!(layers >= 8 && group > 0 && layers.is_multiple_of(group));
    ensure!(
        first_dense < layers,
        "bailing_moe_v3_vl first_k_dense_replace must leave routed layers"
    );
    ensure!(
        nullable_usize(text, "q_lora_rank")?.is_none(),
        "bailing_moe_v3_vl q_lora_rank must be null"
    );
    let mtp = if text.get("num_nextn_predict_layers").is_some() {
        usize_field(text, "num_nextn_predict_layers")?
    } else {
        0
    };
    ensure!(
        mtp <= 1,
        "Ling supports up to one next-token prediction layer"
    );
    ensure!(
        usize_field(text, "num_kv_heads_for_linear_attn")? == 0,
        "bailing_moe_v3_vl only supports the full-head KDA layout"
    );
    ensure!(
        bool_field(text, "no_kda_lora")?
            && !bool_field(text, "use_kda_lora")?
            && !bool_field(text, "mtp_use_kda")?,
        "bailing_moe_v3_vl only supports no-KDA-LoRA layers"
    );
    ensure!(
        bool_field(text, "kda_safe_gate")?
            && bool_field(text, "linear_silu")?
            && bool_field(text, "use_qk_norm")?
            && bool_field(text, "moe_router_enable_expert_bias")?
            && bool_field(text, "norm_topk_prob")?,
        "bailing_moe_v3_vl checkpoint uses unsupported routing or KDA flags"
    );
    ensure!(
        string_field(text, "router_dtype")? == "fp32"
            && string_field(text, "score_function")? == "sigmoid"
            && string_field(text, "gated_attention_proj_granularity_type")? == "head_wise",
        "bailing_moe_v3_vl checkpoint uses unsupported router or attention-gate modes"
    );
    ensure!(usize_field(text, "n_group")? == 8 && usize_field(text, "topk_group")? == 4);
    ensure!(usize_field(text, "group_norm_size")? == 1);
    let decay_lower_bound = f64_field(text, "kda_lower_bound")?;
    ensure!(decay_lower_bound == -5.0);
    expect_reference_lists(text, layers)?;

    let vision = parse_vision(raw, text)?;
    let mrope_section = parse_mrope(raw)?;
    let mut normalized = text.clone();
    normalized["q_lora_rank"] = 0.into();
    let mut config: ModelConfig = serde_json::from_value(normalized)
        .context("failed to parse bailing_moe_v3_vl text_config")?;

    let heads = config.num_attention_heads;
    let head_dim = config.head_dim;
    config.model_type = "bailing_moe_v3_vl".to_string();
    config.tie_word_embeddings = raw
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .context("bailing_moe_v3_vl config missing tie_word_embeddings")?;
    config.layer_types = (0..layers)
        .map(|i| {
            if (i + 1).is_multiple_of(group) {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    config.full_attention_interval = group;
    config.kda_gate_lower_bound = decay_lower_bound as f32;
    config.mlp_only_layers = (0..first_dense).collect();
    config.linear_num_key_heads = heads;
    config.linear_key_head_dim = head_dim;
    config.linear_num_value_heads = heads;
    config.linear_value_head_dim = head_dim;
    config.linear_conv_kernel_dim = usize_field(text, "short_conv_kernel_size")?;
    config.moe_shared_expert_intermediate_size =
        usize_field(text, "moe_shared_expert_intermediate_size")?;
    config.shared_expert_intermediate_size = config.moe_shared_expert_intermediate_size;
    config.routed_scaling_factor = f64_field(text, "routed_scaling_factor")?;
    config.scoring_func = "sigmoid".to_string();
    config.use_routing_bias = true;
    config.router_logits_f32 = true;
    config.attn_gated = true;
    config.nested_config = true;
    config.mtp_num_hidden_layers = mtp;
    config.mrope_section = mrope_section;
    // Ling carries three position streams through Atlas metadata. Its native
    // decoder applies the checkpoint's body-diagonal channel mapping (rather
    // than Qwen3.6's round-robin kernel), so the shared staging/layout must
    // still reserve H/W streams for every prefill and decode row.
    config.mrope_interleaved = true;
    config.vision = Some(vision);

    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
#[path = "bailing_moe_v3_vl_tests.rs"]
mod tests;
