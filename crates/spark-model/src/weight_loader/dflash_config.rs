// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter `config.json` schema and parser (split out of
//! `dflash_loader.rs` for the file-size budget; re-exported from there).

use anyhow::{Context, Result};
use serde::Deserialize;

/// Drafter HF `config.json` (subset Atlas consumes). Mirrors
/// `z-lab/Qwen3.6-35B-A3B-DFlash/config.json` field names verbatim so
/// `serde_json::from_str` works directly on the raw file.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    /// HF architecture identities; absent on older generic DFlash configs.
    #[serde(default)]
    pub architectures: Option<Vec<String>>,
    #[serde(default)]
    pub draft_vocab_size: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Lightning DSpark's required anchor/sampling declarations.
    #[serde(default)]
    pub dspark_bonus_anchor: Option<bool>,
    #[serde(default)]
    pub sample_from_anchor: Option<bool>,
    #[serde(default)]
    pub attention_sink_bias: Option<bool>,
    #[serde(default)]
    pub dspark_markov_rank: Option<usize>,
    #[serde(default)]
    pub target_layer_ids: Option<Vec<usize>>,
    #[serde(
        default,
        alias = "dspark_confidence_head",
        alias = "enable_confidence_head"
    )]
    pub confidence_head: Option<bool>,
    #[serde(default, alias = "dspark_adaptive", alias = "adaptive_verification")]
    pub adaptive: Option<bool>,
    #[serde(default)]
    pub quantization_config: Option<DflashQuantizationConfig>,
    /// Block size γ. Qwen3.6-DFlash ships `block_size: 16`.
    #[serde(default = "default_block_size")]
    pub block_size: usize,
    /// DFlash-specific nested config object.
    #[serde(default)]
    pub dflash_config: Option<DflashSubConfig>,
    /// Drafter base RoPE θ. Defaults to 10M (matches Qwen3.6-DFlash).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// HF-style `rope_scaling` block. `None` ⇒ plain RoPE (the v2 2026-04-27
    /// Qwen3.6-DFlash drafter ships `rope_scaling: null`). When present and
    /// `rope_type == "yarn"`, the drafter's YaRN parameters are used to
    /// build the inv_freq table at construction time.
    #[serde(default)]
    pub rope_scaling: Option<DflashRopeScaling>,
    /// DSpark Markov rank. `None` / 0 = DFlash-only (no sequential fixup).
    #[serde(default)]
    pub markov_rank: Option<usize>,
    /// Drafter RMSNorm epsilon (Qwen3 default 1e-6; GLM-5.3's DFlash2 ships 1e-5).
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,
    /// transformers-5 `rope_parameters`. Folded into `rope_theta` /
    /// `rope_scaling` by [`parse_dflash_config`] when those root keys are absent.
    #[serde(default)]
    pub rope_parameters: Option<DflashRopeParameters>,
    /// HF `is_causal`; wins over `dflash_config.causal` (see [`Self::query_causal`]).
    #[serde(default)]
    pub is_causal: Option<bool>,
    /// HF sliding-window keys, read by [`Self::sliding_window`].
    #[serde(default)]
    pub use_sliding_window: Option<bool>,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub layer_types: Option<Vec<String>>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

/// transformers-5 `rope_parameters`: `rope_theta` plus the scaling fields
/// older configs carried in `rope_scaling` (`incoai/GLM-5.3-Flash-DFlash2`
/// ships its theta only here).
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeParameters {
    #[serde(default)]
    pub rope_theta: Option<f32>,
    #[serde(flatten)]
    pub scaling: DflashRopeScaling,
}

/// Subset of HF `rope_scaling` block consumed by Atlas. Mirrors the field
/// names in `transformers`' Qwen3 config so `serde_json::from_str` works
/// directly on the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashRopeScaling {
    /// Currently only `"yarn"` is recognised; anything else falls back to
    /// plain RoPE with a warning logged at construction time.
    #[serde(default)]
    pub rope_type: Option<String>,
    #[serde(default)]
    pub factor: Option<f32>,
    #[serde(default)]
    pub beta_fast: Option<f32>,
    #[serde(default)]
    pub beta_slow: Option<f32>,
    #[serde(default)]
    pub original_max_position_embeddings: Option<f32>,
}

/// Minimal nested quantization metadata needed for runtime admission.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashQuantizationConfig {
    #[serde(default)]
    pub kv_cache_quant_algo: Option<String>,
}

fn default_block_size() -> usize {
    16
}

/// Nested `dflash_config` block in the drafter's `config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DflashSubConfig {
    /// Token id used to fill the γ "to-be-predicted" positions during draft
    /// inference. `248070` for Qwen3.6-DFlash and Qwen3.8-DFlash2.
    pub mask_token_id: u32,
    /// Target-model layer indices to capture intermediate hidden states from.
    /// `[1, 10, 19, 28, 37]` for Qwen3.6-35B-A3B-DFlash; `[5, 19, 33, 47, 61]` for Qwen3.8-27B.
    pub target_layer_ids: Vec<usize>,
    /// When `Some(true)`, every drafter layer uses causal γ-block attention.
    #[serde(default)]
    pub causal: Option<bool>,
    /// Force sliding-window attention on every drafter layer.
    #[serde(default)]
    pub use_swa: Option<bool>,
    /// Sliding-window size when `use_swa` or `layer_types` request SWA.
    #[serde(default)]
    pub swa_window_size: Option<usize>,
    /// When `Some(true)`, each layer must ship `self_attn.attention_sink_bias`.
    #[serde(default)]
    pub attention_sink_bias: Option<bool>,
    #[serde(default)]
    pub sample_from_anchor: Option<bool>,
    #[serde(
        default,
        alias = "dspark_confidence_head",
        alias = "enable_confidence_head"
    )]
    pub confidence_head: Option<bool>,
    #[serde(default, alias = "dspark_adaptive", alias = "adaptive_verification")]
    pub adaptive: Option<bool>,
    /// DFlash2 nested block size γ (typically 8).
    #[serde(default)]
    pub block_size: Option<usize>,
    /// Channels sharing a dynamic convolution kernel in DFlash2 (16).
    #[serde(default)]
    pub conv_group_size: Option<usize>,
    /// Convolution kernel size in DFlash2 (2 taps).
    #[serde(default)]
    pub conv_kernel_size: Option<usize>,
    /// Codebook embedding rank in DFlash2 CandidateSelector (256).
    #[serde(default)]
    pub selector_rank: Option<usize>,
    /// Candidate pool size evaluated by DFlash2 CandidateSelector (16).
    #[serde(default)]
    pub selector_top_k: Option<usize>,
}

impl DflashConfig {
    pub fn block_size(&self) -> usize {
        self.dflash_config
            .as_ref()
            .and_then(|c| c.block_size)
            .unwrap_or(self.block_size)
    }

    pub fn is_dflash2(&self) -> bool {
        self.architectures
            .as_deref()
            .map(|a| a.iter().any(|name| name == "DFlash2DraftModel"))
            .unwrap_or(false)
    }

    /// Causal γ-block attention: HF `is_causal` wins over
    /// `dflash_config.causal`; both absent ⇒ bidirectional.
    pub fn query_causal(&self) -> bool {
        self.is_causal
            .or_else(|| self.dflash_config.as_ref().and_then(|c| c.causal))
            .unwrap_or(false)
    }

    /// Checkpoint sliding window: `dflash_config.swa_window_size`, else the HF
    /// `sliding_window` when `use_sliding_window` (or `dflash_config.use_swa`)
    /// is set or every `layer_types` entry is `sliding_attention`.
    pub fn sliding_window(&self) -> Option<usize> {
        let sub = self.dflash_config.as_ref();
        if let Some(window) = sub.and_then(|c| c.swa_window_size) {
            return Some(window);
        }
        let all_sliding = self
            .layer_types
            .as_ref()
            .is_some_and(|t| !t.is_empty() && t.iter().all(|t| t == "sliding_attention"));
        let sliding = self.use_sliding_window == Some(true)
            || sub.and_then(|c| c.use_swa) == Some(true)
            || all_sliding;
        if sliding { self.sliding_window } else { None }
    }
}

/// Parse a DFlash drafter's `config.json` into a [`DflashConfig`]. Used by
/// `main.rs` after fetching the drafter's HF metadata to size the runtime
/// `BlockDiffusionDraftHead` (layer count, head_dim, vocab_size, the
/// `target_layer_ids` capture indices).
///
/// transformers-5 checkpoints may carry the rope settings ONLY in
/// `rope_parameters` (folded in when the root keys are absent), and DFlash2
/// nests `block_size` in `dflash_config` (folded with the precedence of
/// [`DflashConfig::block_size`]), so every consumer of `rope_theta` /
/// `block_size` sees the checkpoint's values instead of the Qwen3.6 defaults.
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    let raw: serde_json::Value =
        serde_json::from_str(json).context("Parsing DFlash drafter config.json")?;
    let mut config: DflashConfig =
        serde_json::from_value(raw.clone()).context("Parsing DFlash drafter config.json")?;
    if let Some(params) = config.rope_parameters.clone() {
        if raw.get("rope_theta").is_none()
            && let Some(theta) = params.rope_theta
        {
            config.rope_theta = theta;
        }
        let scaled = params
            .scaling
            .rope_type
            .as_deref()
            .is_some_and(|t| t != "default");
        if config.rope_scaling.is_none() && scaled {
            config.rope_scaling = Some(params.scaling);
        }
    }
    // Same precedence as [`DflashConfig::block_size`]: the nested value wins.
    config.block_size = config.block_size();
    Ok(config)
}
