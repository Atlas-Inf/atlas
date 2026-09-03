// SPDX-License-Identifier: AGPL-3.0-only

//! DFlash drafter weight loader.
//!
//! Loads `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash`-style drafter checkpoints into
//! the typed [`DflashWeights`] structure consumed by
//! [`crate::layers::BlockDiffusionDraftHead`]. The drafter is a small
//! Qwen3-architecture transformer (8 layers, hidden=2048, GQA 32:4) with
//! these distinctive parts vs. a vanilla Qwen3:
//!
//!  * `model.fc` — `[len(target_layer_ids) * target_hidden, draft_hidden]`
//!    BF16 projection that maps the stack of captured target hidden states
//!    into the drafter's input space.
//!  * `model.hidden_norm` — RMSNorm applied to the projected target context
//!    before mixing with token embeddings.
//!  * `lm_head` — drafter ships its own (NOT tied to target's), so
//!    `tie_word_embeddings=false`.
//!  * Optional `d2t` — draft-vocab → target-vocab id remap (absent when
//!    drafter shares vocab with target, as in Qwen3.6-35B-A3B-DFlash where
//!    both = 248320).
//!  * Special `mask_token_id` (`248070` for Qwen3.6-DFlash) used for the γ
//!    "to-be-predicted" positions in block diffusion.
//!
//! Under TP the drafter is **not sharded** — it's small (~1–2 GB BF16),
//! every rank loads the full set. Mirrors the existing MTP-under-TP pattern
//! (`MTP loads ALL experts on every rank — no EP all_reduce needed`).

use anyhow::{Context, Result};
use serde::Deserialize;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::weight_map::{DenseWeight, dense_auto};

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

/// DFlash2 2-tap grouped dynamic convolution weights.
#[derive(Clone)]
pub struct DflashConvWeights {
    pub base_kernel: DenseWeight,
    pub kernel_projection: DenseWeight,
}

/// DFlash2 bilinear candidate selector weights.
#[derive(Clone)]
pub struct DflashCandidateSelectorWeights {
    pub hidden_projection: DenseWeight,
    pub predecessor_codebook: DenseWeight,
    pub successor_codebook: DenseWeight,
    pub rank: usize,
    pub top_k: usize,
}

/// Raw weight bundle for the DFlash drafter, post-load.
#[allow(dead_code)]
pub struct DflashWeights {
    pub config: DflashConfig,
    pub fc: DenseWeight,
    pub hidden_norm: DenseWeight,
    pub norm: DenseWeight,
    pub layers: Vec<DflashLayerWeights>,
    pub draft_id_to_target_id: Option<Vec<i64>>,
    pub markov_w1: Option<DenseWeight>,
    pub markov_w2: Option<DenseWeight>,
    pub markov_rank: usize,
    pub embed_tokens: Option<DenseWeight>,
    pub candidate_selector: Option<DflashCandidateSelectorWeights>,
}

/// Per-drafter-layer raw weights (BF16).
#[allow(dead_code)]
pub struct DflashLayerWeights {
    pub input_layernorm: DenseWeight,
    pub post_attention_layernorm: DenseWeight,
    pub q_proj: DenseWeight,
    pub k_proj: DenseWeight,
    pub v_proj: DenseWeight,
    pub o_proj: DenseWeight,
    pub q_norm: DenseWeight,
    pub k_norm: DenseWeight,
    pub gate_proj: DenseWeight,
    pub up_proj: DenseWeight,
    pub down_proj: DenseWeight,
    pub attention_sink_bias: Option<DenseWeight>,
    pub attention_conv: Option<DflashConvWeights>,
    pub mlp_conv: Option<DflashConvWeights>,
}

/// Probe a [`WeightStore`] for the presence of DFlash drafter weights.
/// Returns true if the store contains the unique `fc.weight` tensor that
/// DFlash drafters ship — a lightweight detection that doesn't load any
/// data. Both bare-key and `model.`-prefixed layouts are accepted; the
/// canonical `z-lab/Qwen3.6-{27B,35B-A3B}-DFlash` checkpoints ship the
/// bare layout (verified against commit 42d3b34, May 2026).
pub fn store_has_dflash_weights(store: &WeightStore) -> bool {
    store.contains("fc.weight") || store.contains("model.fc.weight")
}

/// Parse a DFlash drafter's `config.json` into a [`DflashConfig`]. Used by
/// `main.rs` after fetching the drafter's HF metadata to size the runtime
/// `BlockDiffusionDraftHead` (layer count, head_dim, vocab_size, the
/// `target_layer_ids` capture indices).
pub fn parse_dflash_config(json: &str) -> Result<DflashConfig> {
    serde_json::from_str(json).context("Parsing DFlash drafter config.json")
}

/// Load DFlash drafter weights from a separate [`WeightStore`] pointing at
/// the drafter checkpoint.
///
/// The drafter ships its weights at the **root** of the safetensors file
/// (no `model.` prefix), in the same naming convention as a vanilla Qwen3
/// transformer minus `embed_tokens` and `lm_head`. Atlas's runtime fills
/// those two from the *target* model's embedding / LM head at construction
/// time — exactly mirroring vLLM's "absent in checkpoint → skip_substrs →
/// share with parent" flow.
///
/// The probed key list (verified against `z-lab/Qwen3.6-35B-A3B-DFlash`):
///
/// ```text
///   fc.weight                                              [H, 5*H_target]
///   hidden_norm.weight                                     [H]
///   norm.weight                                            [H]
///   layers.{0..L-1}.input_layernorm.weight                 [H]
///   layers.{0..L-1}.post_attention_layernorm.weight        [H]
///   layers.{0..L-1}.self_attn.q_proj.weight                [Q*Hd, H]
///   layers.{0..L-1}.self_attn.k_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.v_proj.weight                [Kv*Hd, H]
///   layers.{0..L-1}.self_attn.o_proj.weight                [H, Q*Hd]
///   layers.{0..L-1}.self_attn.q_norm.weight                [Hd]
///   layers.{0..L-1}.self_attn.k_norm.weight                [Hd]
///   layers.{0..L-1}.mlp.gate_proj.weight                   [I, H]
///   layers.{0..L-1}.mlp.up_proj.weight                     [I, H]
///   layers.{0..L-1}.mlp.down_proj.weight                   [H, I]
/// ```
///
/// where `H=2048`, `H_target=2048`, `Q=32`, `Kv=4`, `Hd=128`, `I=6144`,
/// `L=8` for Qwen3.6-35B-A3B-DFlash.
///
/// Under TP the drafter is replicated, not sharded — `tp_size>1` produces
/// the same per-rank result as `tp_size=1`. Memory cost: ~948 MB BF16
/// per rank, trivially below the 119 GB GB10 budget.
pub fn load_dflash_weights(
    drafter_store: &WeightStore,
    drafter_config: &DflashConfig,
    gpu: &dyn GpuBackend,
    _tp_size: usize,
) -> Result<Option<DflashWeights>> {
    if !store_has_dflash_weights(drafter_store) {
        tracing::debug!("DFlash drafter store has no `fc.weight` — skipping");
        return Ok(None);
    }

    let prefix = if drafter_store.contains("model.fc.weight") {
        "model."
    } else {
        ""
    };

    // dense_auto: BF16 as-is, packed NVFP4 (Lightning DSpark MLP/fc/markov_w2)
    // dequanted once at load.
    let fc = dense_auto(drafter_store, &format!("{prefix}fc.weight"), gpu)
        .context("DFlash drafter: load fc.weight")?;
    let hidden_norm = dense_auto(drafter_store, &format!("{prefix}hidden_norm.weight"), gpu)
        .context("DFlash drafter: load hidden_norm.weight")?;
    let norm = dense_auto(drafter_store, &format!("{prefix}norm.weight"), gpu)
        .context("DFlash drafter: load norm.weight")?;

    let require_sinks = drafter_config
        .dflash_config
        .as_ref()
        .and_then(|c| c.attention_sink_bias)
        == Some(true);
    let layer_count = drafter_config.num_hidden_layers;
    let mut layers = Vec::with_capacity(layer_count);
    let mut sink_layers = 0usize;
    for i in 0..layer_count {
        let lp = format!("{prefix}layers.{i}");
        let sink_name = format!("{lp}.self_attn.attention_sink_bias");
        let attention_sink_bias = if drafter_store.contains(&sink_name) {
            sink_layers += 1;
            Some(dense_auto(drafter_store, &sink_name, gpu)?)
        } else if require_sinks {
            anyhow::bail!(
                "dflash_config.attention_sink_bias=true but {sink_name} is missing from the checkpoint"
            );
        } else {
            None
        };
        let attn_conv_base = format!("{lp}.attention_conv.base_kernel");
        let attn_conv_proj = format!("{lp}.attention_conv.kernel_projection.weight");
        let attention_conv =
            if drafter_store.contains(&attn_conv_base) && drafter_store.contains(&attn_conv_proj) {
                Some(DflashConvWeights {
                    base_kernel: dense_auto(drafter_store, &attn_conv_base, gpu)?,
                    kernel_projection: dense_auto(drafter_store, &attn_conv_proj, gpu)?,
                })
            } else {
                None
            };

        let mlp_conv_base = format!("{lp}.mlp_conv.base_kernel");
        let mlp_conv_proj = format!("{lp}.mlp_conv.kernel_projection.weight");
        let mlp_conv =
            if drafter_store.contains(&mlp_conv_base) && drafter_store.contains(&mlp_conv_proj) {
                Some(DflashConvWeights {
                    base_kernel: dense_auto(drafter_store, &mlp_conv_base, gpu)?,
                    kernel_projection: dense_auto(drafter_store, &mlp_conv_proj, gpu)?,
                })
            } else {
                None
            };

        let layer = DflashLayerWeights {
            input_layernorm: dense_auto(
                drafter_store,
                &format!("{lp}.input_layernorm.weight"),
                gpu,
            )?,
            post_attention_layernorm: dense_auto(
                drafter_store,
                &format!("{lp}.post_attention_layernorm.weight"),
                gpu,
            )?,
            q_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.q_proj.weight"), gpu)?,
            k_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.k_proj.weight"), gpu)?,
            v_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.v_proj.weight"), gpu)?,
            o_proj: dense_auto(drafter_store, &format!("{lp}.self_attn.o_proj.weight"), gpu)?,
            q_norm: dense_auto(drafter_store, &format!("{lp}.self_attn.q_norm.weight"), gpu)?,
            k_norm: dense_auto(drafter_store, &format!("{lp}.self_attn.k_norm.weight"), gpu)?,
            gate_proj: dense_auto(drafter_store, &format!("{lp}.mlp.gate_proj.weight"), gpu)?,
            up_proj: dense_auto(drafter_store, &format!("{lp}.mlp.up_proj.weight"), gpu)?,
            down_proj: dense_auto(drafter_store, &format!("{lp}.mlp.down_proj.weight"), gpu)?,
            attention_sink_bias,
            attention_conv,
            mlp_conv,
        };
        layers.push(layer);
    }

    let draft_id_to_target_id = if drafter_store.contains(&format!("{prefix}d2t"))
        || drafter_store.contains(&format!("{prefix}draft_id_to_target_id"))
    {
        tracing::warn!(
            "DFlash drafter has draft-id→target-id mapping; remapping path is not yet wired (Phase 2.5 follow-up)"
        );
        Some(Vec::new())
    } else {
        None
    };

    let markov_w1_name = format!("{prefix}markov_head.markov_w1.weight");
    let markov_w2_name = format!("{prefix}markov_head.markov_w2.weight");
    let (markov_w1, markov_w2, markov_rank) =
        if drafter_store.contains(&markov_w1_name) && drafter_store.contains(&markov_w2_name) {
            let rank = drafter_config.markov_rank.unwrap_or(512);
            tracing::info!("DSpark Markov head: rank={rank} (w1 BF16, w2 auto/NVFP4)");
            (
                Some(dense_auto(drafter_store, &markov_w1_name, gpu)?),
                Some(dense_auto(drafter_store, &markov_w2_name, gpu)?),
                rank,
            )
        } else {
            (None, None, 0)
        };

    let sub = drafter_config.dflash_config.as_ref();
    let cs_proj_name = format!("{prefix}candidate_selector.hidden_projection.weight");
    let cs_pred_name = format!("{prefix}candidate_selector.predecessor_codebook");
    let cs_succ_name = format!("{prefix}candidate_selector.successor_codebook");
    let candidate_selector = if drafter_store.contains(&cs_proj_name)
        && drafter_store.contains(&cs_pred_name)
        && drafter_store.contains(&cs_succ_name)
    {
        let rank = sub.and_then(|c| c.selector_rank).unwrap_or(256);
        let top_k = sub.and_then(|c| c.selector_top_k).unwrap_or(16);
        tracing::info!("DFlash2 candidate selector loaded: rank={rank}, top_k={top_k}");
        Some(DflashCandidateSelectorWeights {
            hidden_projection: dense_auto(drafter_store, &cs_proj_name, gpu)?,
            predecessor_codebook: dense_auto(drafter_store, &cs_pred_name, gpu)?,
            successor_codebook: dense_auto(drafter_store, &cs_succ_name, gpu)?,
            rank,
            top_k,
        })
    } else {
        None
    };

    let embed_name = format!("{prefix}embed_tokens.weight");
    let embed_tokens = if drafter_store.contains(&embed_name) {
        Some(dense_auto(drafter_store, &embed_name, gpu)?)
    } else {
        None
    };

    if let Ok(fc_meta) = drafter_store.get(&format!("{prefix}fc.weight"))
        && fc_meta.dtype == spark_runtime::weights::WeightDtype::UInt8
        && fc_meta.shape.len() == 2
    {
        tracing::info!(
            "DFlash fc NVFP4 unpack: on-disk {:?} U8 → logical [{}, {}] BF16",
            fc_meta.shape,
            fc_meta.shape[0],
            fc_meta.shape[1] * 2
        );
    }
    tracing::info!(
        "DFlash/DSpark drafter loaded: {} layers, hidden={}, vocab={}, γ={}, markov_rank={}, own_embed={}, causal={:?}, swa={:?}/{:?}, sinks={}/{}, candidate_selector={}, target_layers={:?}",
        layers.len(),
        drafter_config.hidden_size,
        drafter_config.vocab_size,
        drafter_config.block_size(),
        markov_rank,
        embed_tokens.is_some(),
        sub.and_then(|c| c.causal),
        sub.and_then(|c| c.use_swa),
        sub.and_then(|c| c.swa_window_size),
        sink_layers,
        layers.len(),
        candidate_selector.is_some(),
        sub.map(|c| c.target_layer_ids.as_slice()).unwrap_or(&[]),
    );

    Ok(Some(DflashWeights {
        config: drafter_config.clone(),
        fc,
        hidden_norm,
        norm,
        layers,
        draft_id_to_target_id,
        markov_w1,
        markov_w2,
        markov_rank,
        embed_tokens,
        candidate_selector,
    }))
}

#[cfg(test)]
#[path = "dflash_loader_tests.rs"]
mod tests;
