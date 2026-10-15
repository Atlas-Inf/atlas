// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter `config.json` types. Split out of `dflash_loader.rs` for
//! the 500-LoC file budget — the serde surface mirrors the checkpoints'
//! field names verbatim so `serde_json::from_str` works on the raw file.

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
    /// NVIDIA LiLiCorr drafters ship their own `lm_head` (`Some(true)`);
    /// generic DFlash drafters omit the key and share the target's head.
    #[serde(default)]
    pub has_own_lm_head: Option<bool>,
}

fn default_rope_theta() -> f32 {
    10_000_000.0
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
    /// NVIDIA LiLiCorr markers. `"lilicorr"` + `lilicorr_enabled: true`
    /// declare the candidate re-ranker; its weights load but the forward
    /// isn't implemented — drafts use the drafter's own argmax.
    #[serde(default)]
    pub projector_type: Option<String>,
    #[serde(default)]
    pub lilicorr_enabled: Option<bool>,
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
}
