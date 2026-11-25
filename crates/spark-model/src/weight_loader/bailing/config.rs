// SPDX-License-Identifier: AGPL-3.0-only

//! Bailing-Moe-V3-VL checkpoint config: the pure parse of `config.json`
//! into the Bailing-specific facts Atlas needs for weight loading.
//!
//! The outer HF wrapper is `model_type = bailing_moe_v3_vl` (VL: qwen3_moe_vit
//! vision tower + `linear_proj` projector, both skipped in this text-only
//! phase); every text-decoder fact lives in the nested `text_config`
//! (upstream `model_type = bailing_hybrid`; the primitive-ai NVFP4 build
//! omits the inner `model_type` key — accepted either way here).
//!
//! This module exists because the Bailing constants below
//! (hybrid layout, SwiGLU clamp lists, router grouping, KDA gates) have no
//! [`atlas_core::config::ModelConfig`] counterpart yet; the integration lead
//! wires the parser that maps the shared dims into `ModelConfig` and calls
//! through here for the family-specific remainder. The NVFP4 checkpoint
//! ships NO `rope_scaling`: native context is 262,144.

// Pure-parse library: the Bailing-specific config facts (layout, router
// grouping, SwiGLU clamp lists) are consumed by the checkpoint-config
// parser + kernel wiring landing with the ling-3.2-air integration step;
// the colocated tests exercise them against the real checkpoint config
// until then.
#![cfg_attr(not(test), allow(dead_code))]

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::gate::{RouterGating, SwigluLimits, required_usize};

/// Per-layer attention architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum AttentionKind {
    /// `BailingMoeV3KimiDeltaAttention` — linear attention.
    Kda,
    /// `BailingMoeV3MultiLatentAttention` — full attention, head-wise gated.
    GatedMla,
}

/// Per-layer FFN architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum MlpKind {
    /// Dense `BailingMoeV3MLP` (layers `0..first_k_dense_replace`).
    Dense,
    /// `BailingMoeV3SparseMoeBlock` — 512 routed experts + shared expert.
    Moe,
}

/// Outer YaRN `rope_scaling` block, when a checkpoint ships one.
/// `primitive-ai/Ling-3.2-air-NVFP4` does not (native 262k context).
#[derive(Clone, Debug)]
pub(super) struct RopeScaling {
    pub rope_type: String,
    pub factor: f64,
    pub original_max_position_embeddings: usize,
}

/// The Bailing-specific checkpoint facts (parsed, not defaulted).
#[derive(Clone, Debug)]
pub(crate) struct BailingConfig {
    // ── dims ──
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub vocab_size: usize,
    /// Dense-MLP intermediate (layers 0..first_k_dense_replace).
    pub intermediate_size: usize,
    pub moe_intermediate_size: usize,
    pub moe_shared_expert_intermediate_size: usize,
    pub rms_norm_eps: f64,
    pub max_position_embeddings: usize,

    // ── layout ──
    /// Decoder layers below this index carry a dense MLP.
    pub first_k_dense_replace: usize,
    /// Stride of the gated-MLA layers among the KDA layers.
    pub layer_group_size: usize,
    /// ONE NEXTN layer (`model.layers.42.*` for a 42-layer stack).
    pub num_nextn_predict_layers: usize,
    /// When true the MTP layer uses the KDA attention; Ling ships false (MLA).
    pub mtp_use_kda: bool,

    // ── attention ──
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub kv_lora_rank: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub gated_attention_proj_granularity_type: String,
    pub use_qk_norm: bool,
    pub rope_theta: f64,
    pub rotary_dim: usize,
    pub partial_rotary_factor: f64,

    // ── KDA ──
    pub short_conv_kernel_size: usize,
    pub linear_silu: bool,
    pub kda_safe_gate: bool,
    pub kda_lower_bound: f64,
    /// Ling loads the lora-less KDA shapes (`f_proj`/`g_proj` direct).
    pub no_kda_lora: bool,

    // ── routing + per-layer SwiGLU clamps ──
    pub gating: RouterGating,
    pub swiglu: SwigluLimits,

    // ── tokens / positions ──
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub mrope_section: [usize; 3],
    pub rope_scaling: Option<RopeScaling>,
}

impl BailingConfig {
    /// Parse the FULL outer `config.json` value (`bailing_moe_v3_vl` wrapper).
    pub(crate) fn from_value(raw: &Value) -> Result<Self> {
        let outer_type = raw.get("model_type").and_then(Value::as_str).unwrap_or("");
        ensure!(
            outer_type == "bailing_moe_v3_vl",
            "bailing: expected outer model_type bailing_moe_v3_vl, got {outer_type:?}"
        );
        let text_config = raw
            .get("text_config")
            .context("bailing: config missing text_config")?;
        if let Some(inner) = text_config.get("model_type").and_then(Value::as_str) {
            ensure!(
                inner == "bailing_hybrid",
                "bailing: expected text_config model_type bailing_hybrid, got {inner:?}"
            );
        }
        let config = Self::from_text_config(text_config)?;
        // mrope_section ships BOTH at the top level and inside text_config.
        Self::parse_mrope(raw, text_config).map(|mrope_section| Self {
            mrope_section,
            ..config
        })
    }

    /// Parse just the text-decoder `text_config` object.
    fn from_text_config(text_config: &Value) -> Result<Self> {
        let kda_lower_bound = text_config
            .get("kda_lower_bound")
            .and_then(Value::as_f64)
            .context("bailing: text_config missing kda_lower_bound")?;
        let no_kda_lora = text_config
            .get("no_kda_lora")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        ensure!(
            no_kda_lora,
            "bailing: only the lora-less KDA layout (no_kda_lora=true, direct \
             f_proj/g_proj stems) ships tensor names this loader maps"
        );
        let num_experts = required_usize(text_config, "num_experts")?;
        let config = Self {
            hidden_size: required_usize(text_config, "hidden_size")?,
            num_hidden_layers: required_usize(text_config, "num_hidden_layers")?,
            vocab_size: required_usize(text_config, "vocab_size")?,
            intermediate_size: required_usize(text_config, "intermediate_size")?,
            moe_intermediate_size: required_usize(text_config, "moe_intermediate_size")?,
            moe_shared_expert_intermediate_size: required_usize(
                text_config,
                "moe_shared_expert_intermediate_size",
            )?,
            rms_norm_eps: text_config
                .get("rms_norm_eps")
                .and_then(Value::as_f64)
                .context("bailing: text_config missing rms_norm_eps")?,
            max_position_embeddings: required_usize(text_config, "max_position_embeddings")?,

            first_k_dense_replace: required_usize(text_config, "first_k_dense_replace")?,
            layer_group_size: required_usize(text_config, "layer_group_size")?,
            num_nextn_predict_layers: positive_or_zero(text_config, "num_nextn_predict_layers")?,
            mtp_use_kda: text_config
                .get("mtp_use_kda")
                .and_then(Value::as_bool)
                .unwrap_or(false),

            num_attention_heads: required_usize(text_config, "num_attention_heads")?,
            num_key_value_heads: required_usize(text_config, "num_key_value_heads")?,
            head_dim: required_usize(text_config, "head_dim")?,
            kv_lora_rank: required_usize(text_config, "kv_lora_rank")?,
            qk_nope_head_dim: required_usize(text_config, "qk_nope_head_dim")?,
            qk_rope_head_dim: required_usize(text_config, "qk_rope_head_dim")?,
            v_head_dim: required_usize(text_config, "v_head_dim")?,
            gated_attention_proj_granularity_type: text_config
                .get("gated_attention_proj_granularity_type")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            use_qk_norm: text_config
                .get("use_qk_norm")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            rope_theta: text_config
                .get("rope_theta")
                .and_then(Value::as_f64)
                .context("bailing: text_config missing rope_theta")?,
            rotary_dim: required_usize(text_config, "rotary_dim")?,
            partial_rotary_factor: text_config
                .get("partial_rotary_factor")
                .and_then(Value::as_f64)
                .context("bailing: text_config missing partial_rotary_factor")?,

            short_conv_kernel_size: required_usize(text_config, "short_conv_kernel_size")?,
            linear_silu: text_config
                .get("linear_silu")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            kda_safe_gate: text_config
                .get("kda_safe_gate")
                .and_then(Value::as_bool)
                .unwrap_or(false),
            kda_lower_bound,
            no_kda_lora,

            gating: RouterGating::from_text_config(text_config)?,
            swiglu: SwigluLimits::from_text_config(text_config)?,

            eos_token_id: token_id(text_config, "eos_token_id"),
            pad_token_id: token_id(text_config, "pad_token_id"),
            mrope_section: [0, 0, 0],
            rope_scaling: RopeScaling::from_value(text_config)?,
        };
        ensure!(
            config.gating.num_experts == num_experts,
            "bailing: duplicate num_experts disagreement inside text_config"
        );
        config.validate()
    }

    /// Structural invariants the weight loader and kernels rely on.
    fn validate(self) -> Result<Self> {
        ensure!(
            self.layer_group_size > 0,
            "bailing: layer_group_size must be positive"
        );
        ensure!(
            self.first_k_dense_replace <= self.num_hidden_layers,
            "bailing: first_k_dense_replace exceeds the decoder stack"
        );
        ensure!(
            self.hidden_size
                .is_multiple_of(self.num_attention_heads.max(1)),
            "bailing: hidden_size not divisible by num_attention_heads"
        );
        if self.num_nextn_predict_layers > 0 {
            ensure!(
                !self.mtp_use_kda,
                "bailing: the MTP layer must use gated MLA in this loader \
                 (mtp_use_kda=false); the NEXTN tensors mapped are MLA-shaped"
            );
        }
        Ok(self)
    }

    fn parse_mrope(raw: &Value, text_config: &Value) -> Result<[usize; 3]> {
        for (node, where_) in [(raw, "config"), (text_config, "text_config")] {
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
        anyhow::bail!("bailing: config missing mrope_section")
    }

    /// The modeling rule (`BailingMoeV3DecoderLayer.__init__`): gated MLA at
    /// every `(i + 1) % layer_group_size == 0`, or from the last whole group
    /// boundary to the end of the stack; KDA everywhere else.
    pub(crate) fn attention_kind(&self, layer_idx: usize) -> AttentionKind {
        let boundary = self.num_hidden_layers / self.layer_group_size * self.layer_group_size;
        let full = (layer_idx + 1).is_multiple_of(self.layer_group_size) || layer_idx >= boundary;
        if full {
            AttentionKind::GatedMla
        } else {
            AttentionKind::Kda
        }
    }

    /// `BailingMoeV3DecoderLayer.__init__`: MoE from `first_k_dense_replace`
    /// on; dense `BailingMoeV3MLP` below it.
    pub(crate) fn mlp_kind(&self, layer_idx: usize) -> MlpKind {
        if self.gating.num_experts > 0 && layer_idx >= self.first_k_dense_replace {
            MlpKind::Moe
        } else {
            MlpKind::Dense
        }
    }

    /// Checkpoint layer indices of the NEXTN MTP modules
    /// (`BailingMoeV3Model`: appended after the decoder stack, so layer
    /// `num_hidden_layers + k` for k in `0..num_nextn_predict_layers`).
    pub(crate) fn mtp_layer_indices(&self) -> std::ops::Range<usize> {
        self.num_hidden_layers..self.num_hidden_layers + self.num_nextn_predict_layers
    }
}

fn positive_or_zero(text_config: &Value, key: &str) -> Result<usize> {
    let value = text_config
        .get(key)
        .and_then(Value::as_u64)
        .with_context(|| format!("bailing: text_config missing {key}"))?;
    Ok(value as usize)
}

fn token_id(text_config: &Value, key: &str) -> u32 {
    text_config.get(key).and_then(Value::as_u64).unwrap_or(0) as u32
}

impl RopeScaling {
    fn from_value(text_config: &Value) -> Result<Option<Self>> {
        let Some(raw) = text_config.get("rope_scaling") else {
            return Ok(None);
        };
        Ok(Some(Self {
            rope_type: raw
                .get("rope_type")
                .or_else(|| raw.get("type"))
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            factor: raw.get("factor").and_then(Value::as_f64).unwrap_or(1.0),
            original_max_position_embeddings: raw
                .get("original_max_position_embeddings")
                .and_then(Value::as_u64)
                .unwrap_or(0) as usize,
        }))
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
