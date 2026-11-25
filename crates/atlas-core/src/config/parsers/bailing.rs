// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for the nested `bailing_moe_v3_vl` conditional-generation config
//! (Ling-3.2 family; checkpoint `primitive-ai/Ling-3.2-air-NVFP4`).
//!
//! Ground truth: the shipped `config.json` (embedded verbatim in
//! [`bailing_tests`]) plus `modeling_bailing_moe_v3_vl.py`. The outer HF
//! wrapper is `bailing_moe_v3_vl`; every text-decoder fact lives in the
//! nested `text_config` (upstream inner `model_type = bailing_hybrid`,
//! omitted by the NVFP4 build — accepted either way, matching
//! `spark-model`'s `weight_loader/bailing/config.rs`).
//!
//! Hybrid layout (`BailingMoeV3DecoderLayer.__init__`, modeling ~line 1514):
//! gated-MLA full attention at every `(i + 1) % layer_group_size == 0` layer
//! and from the last whole group boundary down — Ling (group 6, 42 layers):
//! layers {5, 11, 17, 23, 29, 35, 41}; Kimi-delta (KDA) linear attention on
//! the remaining 35 layers, including the dense-MLP prefix 0..=1
//! (`first_k_dense_replace = 2`). ONE NEXTN MTP layer ships appended after
//! the decoder stack (`model.layers.42.*`, gated-MLA shape:
//! `mtp_use_kda = false`).
//!
//! Constants without a [`ModelConfig`] counterpart are VALIDATED here —
//! keeping this parser consistent with the loader-side
//! `weight_loader/bailing/{config,gate}.rs` parse of the same checkpoint —
//! and recorded there for the family kernels: `n_group = 8` /
//! `topk_group = 4` group-limited sigmoid routing, the per-layer SwiGLU
//! clamp lists, `kda_safe_gate`, `linear_silu`, and the head-wise MLA
//! output gate (`gated_attention_proj_granularity_type`). Constants that DO
//! shape the runtime schema are mapped below, with the mapping decisions
//! documented at each site.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};

/// Bailing-specific routing constants with no generic [`ModelConfig`]
/// counterpart — carried on `ModelConfig::bailing` for the spark-model Ling
/// family driver (`model/bailing_c1.rs`) and the MoE kernels:
/// `n_group`/`topk_group` group-limited top-k plus the per-layer SwiGLU
/// clamps (routed 4 on layers 35..=41; shared 5 on 34..=39, 7 on 40..=41).
#[derive(Debug, Clone, PartialEq)]
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

/// Optional per-layer index list (`expert_swiglu_limit_list` /
/// `share_expert_swiglu_limit_list`): when present it must carry exactly
/// `num_layers` non-negative integer entries. The values (routed experts
/// clamped at 4 on layers 35..=41; shared expert at 5 on 34..=39 and 7 on
/// 40..=41 in the Ling checkpoint) have no `ModelConfig` field — the MoE
/// kernels consume them, mirrored in `weight_loader/bailing/gate.rs`.
fn read_index_list(text: &Value, name: &str, num_layers: usize) -> Result<Vec<i64>> {
    let Some(list) = text.get(name) else {
        return Ok(Vec::new());
    };
    let Value::Array(entries) = list else {
        anyhow::bail!("bailing text_config `{name}` must be an array");
    };
    ensure!(
        entries.len() == num_layers,
        "bailing text_config `{name}` has {} entries but num_hidden_layers is {num_layers}",
        entries.len()
    );
    entries
        .iter()
        .map(|entry| {
            entry
                .as_i64()
                .with_context(|| format!("bailing text_config `{name}` entries must be integers"))
        })
        .collect()
}

pub fn parse_bailing(raw: &Value) -> Result<ModelConfig> {
    let text = raw
        .get("text_config")
        .context("bailing_moe_v3_vl config missing text_config")?;
    if let Some(inner) = text.get("model_type").and_then(Value::as_str) {
        ensure!(
            inner == "bailing_hybrid",
            "bailing: expected text_config model_type bailing_hybrid, got {inner:?}"
        );
    }

    // ── routing shape: validated up front, values owned by the kernels ──
    // `BailingMoeV3Gate.forward` (modeling ~line 469): FP32 logits →
    // sigmoid; selection on scores + expert_bias under group-limited top-k;
    // dispatch on the bias-free scores, renormalized + scaled 2.5.
    let score_function = text
        .get("score_function")
        .and_then(Value::as_str)
        .unwrap_or_default();
    ensure!(
        score_function == "sigmoid",
        "bailing: only sigmoid routing is implemented (modeling \
         BailingMoeV3Gate), config says {score_function:?}"
    );
    let router_dtype = text
        .get("router_dtype")
        .and_then(Value::as_str)
        .unwrap_or_default();
    ensure!(
        router_dtype == "fp32",
        "bailing: router_dtype must be fp32 (the reference computes FP32 \
         logits), config says {router_dtype:?}"
    );
    let num_experts = usize_field(text, "num_experts")?;
    let n_group = usize_field(text, "n_group")?;
    let topk_group = usize_field(text, "topk_group")?;
    ensure!(
        num_experts.is_multiple_of(n_group),
        "bailing: num_experts ({num_experts}) not divisible by n_group ({n_group})"
    );
    ensure!(
        topk_group <= n_group,
        "bailing: topk_group ({topk_group}) out of range for n_group ({n_group})"
    );

    // ── KDA invariants (the loader maps only these tensor shapes) ──
    ensure!(
        text.get("no_kda_lora")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "bailing: only the lora-less KDA layout (no_kda_lora=true, direct \
         f_proj/g_proj stems) is implemented"
    );
    ensure!(
        text.get("kda_safe_gate")
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "bailing: Atlas's bounded KDA gate implements the safe-gate form \
         (kda_safe_gate=true)"
    );
    ensure!(
        text.get("gated_attention_proj_granularity_type")
            .and_then(Value::as_str)
            == Some("head_wise"),
        "bailing: only head_wise gated MLA is implemented \
         (gated_attention_proj_granularity_type)"
    );

    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "bailing_moe_v3_vl".to_string();

    // ── core dims ──
    config.hidden_size = usize_field(text, "hidden_size")?;
    config.num_hidden_layers = usize_field(text, "num_hidden_layers")?;
    config.intermediate_size = usize_field(text, "intermediate_size")?;
    config.vocab_size = usize_field(text, "vocab_size")?;
    // Native context: the NVFP4 plan-of-record checkpoint ships NO
    // `rope_scaling` (262,144). A `deepseek_yarn` block, when present, is
    // recorded into the yarn fields below without touching this value — the
    // checkpoint writes the already-extended length.
    config.max_position_embeddings = usize_field(text, "max_position_embeddings")?;
    config.rms_norm_eps = text
        .get("rms_norm_eps")
        .and_then(Value::as_f64)
        .context("bailing text_config missing rms_norm_eps")?;
    config.eos_token_id = text
        .get("eos_token_id")
        .and_then(Value::as_u64)
        .unwrap_or(0) as u32;
    // No bos_token_id ships in the checkpoint: leave 0, not the template's
    // Qwen id.
    config.bos_token_id = 0;
    config.tie_word_embeddings = raw
        .get("tie_word_embeddings")
        .and_then(Value::as_bool)
        .unwrap_or(false);

    // ── hybrid layout ──
    let layer_group_size = usize_field(text, "layer_group_size")?;
    let num_layers = config.num_hidden_layers;
    let boundary = num_layers / layer_group_size * layer_group_size;
    config.layer_types = (0..num_layers)
        .map(|i| {
            if (i + 1).is_multiple_of(layer_group_size) || i >= boundary {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    // Fallback stride when layer_types is dropped; matches the derived rule.
    config.full_attention_interval = layer_group_size;
    let first_dense = text
        .get("first_k_dense_replace")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    ensure!(
        first_dense <= num_layers,
        "bailing: first_k_dense_replace ({first_dense}) exceeds the decoder stack ({num_layers})"
    );
    config.mlp_only_layers = (0..first_dense).collect();
    let routed_swiglu = read_index_list(text, "expert_swiglu_limit_list", num_layers)?;
    let shared_swiglu = read_index_list(text, "share_expert_swiglu_limit_list", num_layers)?;

    // ── gated MLA (BailingMoeV3MultiLatentAttention, modeling ~line 1101)──
    config.num_attention_heads = usize_field(text, "num_attention_heads")?;
    config.num_key_value_heads = usize_field(text, "num_key_value_heads")?;
    config.kv_lora_rank = usize_field(text, "kv_lora_rank")?;
    // `q_lora_rank: null` → direct q_proj [hidden, heads * (nope + rope)];
    // 0 is ModelConfig's "standard Q" value.
    config.q_lora_rank = text.get("q_lora_rank").and_then(Value::as_u64).unwrap_or(0) as usize;
    config.qk_nope_head_dim = usize_field(text, "qk_nope_head_dim")?;
    config.qk_rope_head_dim = usize_field(text, "qk_rope_head_dim")?;
    config.v_head_dim = usize_field(text, "v_head_dim")?;
    // Expanded MLA Q/K head width (glm5_next convention; the model factory
    // sizes the paged cache as kv_lora_rank + rope for every MLA model).
    // The checkpoint's literal `head_dim` (128) is the per-head V/output
    // width, carried by `v_head_dim`.
    config.head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim;
    config.partial_rotary_factor = text
        .get("partial_rotary_factor")
        .and_then(Value::as_f64)
        .context("bailing text_config missing partial_rotary_factor")?;
    // Partial rope: config ships the explicit integer `rotary_dim = 64`.
    // It wins over `partial_rotary_factor * head_dim` (0.5 * 192 = 96,
    // which would mis-size the RoPE tables for the 192-wide MLA heads).
    config.rotary_dim = usize_field(text, "rotary_dim")?;
    config.rope_theta = text
        .get("rope_theta")
        .and_then(Value::as_f64)
        .context("bailing text_config missing rope_theta")?;
    // M-RoPE [8, 12, 12]: text/temporal, height, width sections in
    // contiguous (Qwen3-VL) layout — the modeling applies plain
    // `apply_rotary_pos_emb`, non-interleaved. `sections.sum() * 2` must
    // equal the rotary dim.
    let mrope_section = parse_mrope_section(raw, text)?;
    ensure!(
        mrope_section.iter().sum::<usize>() * 2 == config.rotary_dim,
        "bailing: mrope_section {mrope_section:?} incompatible with rotary_dim {}",
        config.rotary_dim
    );
    config.mrope_section = mrope_section;
    config.mrope_interleaved = false;

    // ── KDA linear attention (BailingMoeV3KimiDeltaAttention, ~line 1237)─
    // The reference sizes KDA heads at `num_attention_heads` × `head_dim`
    // (both Q/K and V projections; hidden 2560 → 32 heads × 128).
    config.linear_num_key_heads = config.num_attention_heads;
    config.linear_key_head_dim = usize_field(text, "head_dim")?;
    config.linear_num_value_heads = config.num_attention_heads;
    config.linear_value_head_dim = config.linear_key_head_dim;
    config.linear_conv_kernel_dim = usize_field(text, "short_conv_kernel_size")?;
    config.kda_gate_lower_bound =
        text.get("kda_lower_bound")
            .and_then(Value::as_f64)
            .context("bailing text_config missing kda_lower_bound")? as f32;
    // The reference hardcodes the gated output norm at sigmoid
    // (`FusedRMSNormGated(..., activation='sigmoid')`), unlike the
    // SiLU-normed Qwen-family GDN models.
    config.gdn_norm_sigmoid = true;

    // ── MoE (BailingMoeV3SparseMoeBlock, ~line 518) ──
    config.num_experts = num_experts;
    config.n_routed_experts = num_experts;
    config.num_experts_per_tok = usize_field(text, "num_experts_per_tok")?;
    config.moe_intermediate_size = usize_field(text, "moe_intermediate_size")?;
    // ONE shared expert at the shared-intermediate width (the reference
    // multiplies by `num_shared_experts`, which defaults to 1).
    config.shared_expert_intermediate_size =
        usize_field(text, "moe_shared_expert_intermediate_size")?;
    config.moe_shared_expert_intermediate_size = config.shared_expert_intermediate_size;
    config.norm_topk_prob = text
        .get("norm_topk_prob")
        .and_then(Value::as_bool)
        .unwrap_or(true);
    config.scoring_func = "sigmoid".to_string();
    // `moe_router_enable_expert_bias`: the per-expert `gate.expert_bias`
    // participates in SELECTION only, never in the dispatch weights —
    // exactly this field's documented semantics.
    config.use_routing_bias = text
        .get("moe_router_enable_expert_bias")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    config.router_logits_f32 = true;
    config.routed_scaling_factor = text
        .get("routed_scaling_factor")
        .and_then(Value::as_f64)
        .context("bailing text_config missing routed_scaling_factor")?;

    // ── MTP: ONE NEXTN layer appended after the decoder stack ──
    let num_nextn = text
        .get("num_nextn_predict_layers")
        .and_then(Value::as_u64)
        .unwrap_or(0) as usize;
    if num_nextn > 0 {
        ensure!(
            !text
                .get("mtp_use_kda")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "bailing: only the gated-MLA MTP layer shape is implemented (mtp_use_kda=false)"
        );
    }
    config.num_mtp_modules = num_nextn;
    config.mtp_num_hidden_layers = num_nextn;

    // ── optional YaRN extension (official FP8 build; NOT the NVFP4 one) ──
    if let Some(scaling) = text.get("rope_scaling").filter(|v| !v.is_null()) {
        let rope_type = scaling
            .get("rope_type")
            .or_else(|| scaling.get("type"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        ensure!(
            rope_type == "deepseek_yarn",
            "bailing: unsupported rope_scaling type {rope_type:?}"
        );
        config.yarn_factor = scaling.get("factor").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        config.yarn_original_max_position_embeddings = scaling
            .get("original_max_position_embeddings")
            .and_then(Value::as_u64)
            .unwrap_or(0) as usize;
        config.yarn_mscale = scaling.get("mscale").and_then(Value::as_f64).unwrap_or(1.0) as f32;
        config.yarn_mscale_all_dim = scaling
            .get("mscale_all_dim")
            .and_then(Value::as_f64)
            .unwrap_or(0.0) as f32;
    }

    // ── architecture flags ──
    // The head-wise attention output gate is a SEPARATE `g_proj`
    // ([hidden, heads] sigmoid gate on the attention output, modeling
    // ~line 1199) — not the interleaved Q+gate layout this flag names.
    config.attn_gated = false;
    config.nested_config = true;
    // weight_prefix stays empty: the checkpoint stores the decoder directly
    // under `model.*` (model.word_embeddings / model.layers.N / model.norm /
    // lm_head), exactly what `ModelConfig::layer_prefix` composes.
    // Text-only phase: the qwen3_moe_vit tower (`model.visual.*`) and the
    // `linear_proj.*` projector are skipped by the family loader
    // (weight_loader/bailing/vision.rs carries the extension seam).
    config.vision = None;

    // ── family constants for the Ling driver: group-limited routing and
    // per-layer SwiGLU clamps (mirrored in weight_loader/bailing/gate.rs).
    config.bailing = Some(BailingParams {
        n_group,
        topk_group,
        expert_bias: config.use_routing_bias,
        routed_swiglu,
        shared_swiglu,
    });

    finalize_config(&mut config, raw)?;
    Ok(config)
}

/// `mrope_section` ships BOTH at the top level and inside text_config;
/// the top-level entry wins, mirroring the loader-side parse.
fn parse_mrope_section(raw: &Value, text: &Value) -> Result<[usize; 3]> {
    for (node, where_) in [(raw, "config"), (text, "text_config")] {
        if let Some(section) = node.get("mrope_section").and_then(Value::as_array) {
            ensure!(
                section.len() == 3,
                "bailing: mrope_section in {where_} must have 3 entries"
            );
            return section
                .iter()
                .map(|v| v.as_u64().unwrap_or(0) as usize)
                .collect::<Vec<_>>()
                .try_into()
                .map_err(|_| anyhow::anyhow!("bailing: mrope_section parse"));
        }
    }
    anyhow::bail!("bailing: config missing mrope_section");
}

#[cfg(test)]
#[path = "bailing_tests.rs"]
mod tests;
