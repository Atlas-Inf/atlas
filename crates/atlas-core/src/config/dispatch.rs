// SPDX-License-Identifier: AGPL-3.0-only

//! Top-level model-type dispatch for [`super::parse_config`]. Split out of
//! `config.rs` for file-size budget — handles the JSON `model_type` field
//! and routes to the appropriate parser sub-module.

#![allow(unused_imports)]

use anyhow::{Context, Result};

use super::{
    LayerType, ModelConfig, default_partial_rotary, default_rope_theta, finalize_config,
    parse_deepseek_v4, parse_gemma4_params, parse_laguna, parse_longcat_ngram, parse_minimax_m2,
    parse_mistral_params, parse_quantization_config, parse_qwen4_exp, parse_step3p7,
    parse_vision_config, validate_config,
};

use super::dispatch_nemotron;

fn required_u64(raw: &serde_json::Value, key: &str, model_type: &str) -> Result<u64> {
    let value = raw
        .get(key)
        .with_context(|| format!("{model_type} config missing required field `{key}`"))?;
    value
        .as_u64()
        .with_context(|| format!("{model_type} config field `{key}` must be an unsigned integer"))
}

fn required_nonzero_usize(raw: &serde_json::Value, key: &str, model_type: &str) -> Result<usize> {
    let value = required_u64(raw, key, model_type)? as usize;
    if value == 0 {
        anyhow::bail!("{model_type} config field `{key}` must be greater than zero");
    }
    Ok(value)
}

fn required_u32(raw: &serde_json::Value, key: &str, model_type: &str) -> Result<u32> {
    let value = required_u64(raw, key, model_type)?;
    u32::try_from(value)
        .with_context(|| format!("{model_type} config field `{key}` does not fit in u32"))
}

pub fn parse_config(json: &str) -> Result<ModelConfig> {
    // First, probe the top-level model_type.
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Invalid JSON in config.json")?;

    // A remote-code checkpoint may declare ONLY `architectures` + `auto_map`
    // and no `model_type` at all — LongCat-Flash-Lite ships exactly that
    // (`architectures: ["LongcatFlashNgramForCausalLM"]`). Without this
    // fallback such a config silently falls through to the generic parse and
    // loses its family, so map the known architecture names onto their
    // model_type. Only consulted when `model_type` is absent/empty, so no
    // existing checkpoint changes behaviour.
    let top_model_type = raw
        .get("model_type")
        .and_then(serde_json::Value::as_str)
        .filter(|s| !s.is_empty())
        .or_else(|| {
            raw.get("architectures")
                .and_then(serde_json::Value::as_array)
                .and_then(|a| a.first())
                .and_then(serde_json::Value::as_str)
                .and_then(|arch| match arch {
                    "LongcatFlashNgramForCausalLM" => Some("longcat_flash_ngram"),
                    "LongcatFlashForCausalLM" => Some("longcat_flash"),
                    "Qwen4ExpForConditionalGeneration"
                    | "Qwen3_8FlashNextForConditionalGeneration" => Some("qwen4_exp"),
                    _ => None,
                })
        })
        .unwrap_or("");

    match top_model_type {
        "qwen3_vl_moe" | "qwen3_5_moe" | "qwen3_5" => {
            let text_config = raw
                .get("text_config")
                .context("qwen3_5_moe config missing text_config")?;
            let mut config: ModelConfig = serde_json::from_value(text_config.clone())
                .context("Failed to parse text_config")?;
            // Override model_type to the top-level one (text_config has "*_text" suffix)
            config.model_type = top_model_type.to_string();
            // Weight prefix is auto-detected from store keys in main.rs after loading
            // (different quantizers use different prefixes)
            // eos_token_id from text_config
            if config.eos_token_id == 0 {
                config.eos_token_id = text_config
                    .get("eos_token_id")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as u32;
            }
            // Vocab size can also be at top level
            if config.vocab_size == 0 {
                config.vocab_size = raw
                    .get("vocab_size")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(0) as usize;
            }
            // rope_theta and partial_rotary_factor from nested rope_parameters
            if let Some(rope_params) = text_config.get("rope_parameters") {
                if config.rope_theta == default_rope_theta()
                    && let Some(theta) = rope_params
                        .get("rope_theta")
                        .and_then(serde_json::Value::as_f64)
                {
                    config.rope_theta = theta;
                }
                // FP8 checkpoints store partial_rotary_factor inside rope_parameters
                if config.partial_rotary_factor == default_partial_rotary()
                    && let Some(prf) = rope_params
                        .get("partial_rotary_factor")
                        .and_then(serde_json::Value::as_f64)
                {
                    config.partial_rotary_factor = prf;
                }
            }
            // Qwen3.5 MoE unconditionally normalizes top-K expert weights
            // (hardcoded in HF's Qwen3_5MoeTopKRouter, no config toggle).
            config.norm_topk_prob = true;
            // Architecture flags
            config.nested_config = true;
            config.attn_gated = top_model_type != "qwen3_vl_moe";
            // Parse vision_config for VL models. Qwen3.6 also ships a ViT
            // tower (detected via the mrope_interleaved flag set below,
            // but we don't have that until after this block, so also
            // trigger when the raw config has a `vision_config` key).
            if top_model_type == "qwen3_vl_moe" || raw.get("vision_config").is_some() {
                config.vision = parse_vision_config(&raw);
            }
            // MRoPE detection: Qwen3.6 MoE sets mrope_interleaved + mrope_section
            // inside text_config.rope_parameters. When present on a MoE
            // variant, rewrite model_type to "qwen3_6_moe" so kernel-target
            // resolution picks the right directory (Qwen3.5-MoE and
            // Qwen3.6-MoE share hidden_size=2048 and would otherwise collide).
            // The backing weight loader stays in the qwen3_5 family — MoE
            // architecture is identical except for MRoPE layout and the
            // full-attention layer gate.
            //
            // Kbenkhaled's Qwen3.5-27B-NVFP4 is dense (top_model_type="qwen3_5",
            // no experts) but also enables MRoPE. For dense, do NOT rewrite:
            // the Qwen35 MoE weight loader would fail looking for mlp.gate.
            // The qwen3.5-27b kernel target handles MRoPE at runtime via the
            // mrope_interleaved / mrope_section flags.
            if let Some(rope_params) = text_config.get("rope_parameters") {
                if let Some(ms) = rope_params.get("mrope_section").and_then(|v| v.as_array())
                    && ms.len() == 3
                {
                    config.mrope_section = [
                        ms[0].as_u64().unwrap_or(0) as usize,
                        ms[1].as_u64().unwrap_or(0) as usize,
                        ms[2].as_u64().unwrap_or(0) as usize,
                    ];
                }
                config.mrope_interleaved = rope_params
                    .get("mrope_interleaved")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);
                let is_moe = top_model_type == "qwen3_5_moe" || top_model_type == "qwen3_vl_moe";
                if is_moe
                    && config.mrope_interleaved
                    && config.mrope_section.iter().sum::<usize>() > 0
                {
                    config.model_type = "qwen3_6_moe".to_string();
                }
            }
            // Holo-3.1 (Hcompany) is a fine-tune of Qwen3.6-35B-A3B and shares
            // its ENTIRE config — same vision tower, same image_token_id
            // (248056), same MRoPE layout. The one structural difference is
            // that Hcompany strips the MTP head from its releases
            // (text_config has no mtp_num_hidden_layers), while every
            // official Qwen3.6-35B checkpoint ships mtp_num_hidden_layers=1.
            // Gate on that: without it the flagship Qwen/Qwen3.6-35B-A3B-FP8
            // was misdetected as holo3_1_moe and failed kernel-target
            // resolution (targets declare qwen3_6_moe).
            if top_model_type == "qwen3_5_moe"
                && config.vision.is_some()
                && config.mtp_num_hidden_layers == 0
                && raw
                    .get("image_token_id")
                    .and_then(serde_json::Value::as_u64)
                    == Some(248_056)
            {
                config.model_type = "holo3_1_moe".to_string();
            }
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
        "nemotron_h" | "nemotron_h_puzzle" => {
            dispatch_nemotron::parse_nemotron_h(&raw, top_model_type)
        }
        "nemotron_h_omni" => dispatch_nemotron::parse_nemotron_h_omni(&raw),
        "gemma4" => parse_gemma4_params(&raw),
        "laguna" => parse_laguna(&raw),
        "longcat_flash_ngram" | "longcat_flash" => parse_longcat_ngram(&raw),
        // Nested text_config like qwen3_5_moe, but hyper-connections, the QSA
        // indexer and PLE n-gram injection put it outside that arm.
        //
        // TWO NAMES, ONE ARCHITECTURE. Qwen3.8-Flash-Next shipped under
        // `qwen3_8_flash_next` and was later renamed `qwen4_exp`; quantizers
        // pinned to different transformers revisions emit different names
        // (RadixArk -> qwen4_exp, Inferact -> qwen3_8_flash_next). Their
        // `text_config`s are otherwise IDENTICAL field-for-field, so the
        // alias is the whole difference at the config layer.
        "qwen4_exp" | "qwen3_8_flash_next" => parse_qwen4_exp(&raw),
        "m2m_100" | "nllb" => {
            let mut config = ModelConfig::qwen3_next_80b_nvfp4();
            config.model_type = "m2m_100".to_string();
            config.hidden_size = required_nonzero_usize(&raw, "d_model", top_model_type)?;
            config.num_hidden_layers =
                required_nonzero_usize(&raw, "decoder_layers", top_model_type)?;
            config.intermediate_size =
                required_nonzero_usize(&raw, "decoder_ffn_dim", top_model_type)?;
            config.vocab_size = required_nonzero_usize(&raw, "vocab_size", top_model_type)?;
            config.num_attention_heads =
                required_nonzero_usize(&raw, "decoder_attention_heads", top_model_type)?;
            config.num_key_value_heads = config.num_attention_heads;
            if !config
                .hidden_size
                .is_multiple_of(config.num_attention_heads)
            {
                anyhow::bail!(
                    "{} config has d_model ({}) not divisible by decoder_attention_heads ({})",
                    top_model_type,
                    config.hidden_size,
                    config.num_attention_heads,
                );
            }
            config.head_dim = config.hidden_size / config.num_attention_heads;
            config.max_position_embeddings =
                required_nonzero_usize(&raw, "max_position_embeddings", top_model_type)?;
            config.bos_token_id = required_u32(&raw, "bos_token_id", top_model_type)?;
            config.eos_token_id = required_u32(&raw, "eos_token_id", top_model_type)?;
            config.tie_word_embeddings = true;
            config.attn_gated = false;
            config.weight_prefix = "model.decoder".to_string();
            config.num_experts = 0;
            config.num_experts_per_tok = 1;
            config.moe_intermediate_size = 0;
            config.shared_expert_intermediate_size = 0;
            config.layer_types.clear();
            config.full_attention_interval = 1;
            config.linear_num_key_heads = 0;
            config.linear_key_head_dim = 0;
            config.linear_num_value_heads = 0;
            config.linear_value_head_dim = 0;
            config.mtp_num_hidden_layers = 0;
            config.vision = None;
            config.quantization_config = parse_quantization_config(&raw);
            validate_config(&config)?;
            Ok(config)
        }
        "minimax_m2" => parse_minimax_m2(&raw),
        "step3p7" => parse_step3p7(&raw),
        "deepseek_v4" => parse_deepseek_v4(json),
        _ => {
            // Flat config (qwen3_next, etc.)
            let mut config: ModelConfig =
                serde_json::from_str(json).context("Failed to parse config.json")?;
            config.attn_gated = true;
            finalize_config(&mut config, &raw)?;
            Ok(config)
        }
    }
}
