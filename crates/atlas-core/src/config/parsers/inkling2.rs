// SPDX-License-Identifier: AGPL-3.0-only

//! Thinking Machines Inkling 2 Mini configuration parser (`model_type: inkling2_loop`).
//!
//! 24 physical decoder layers. The core `[core_start, core_end)` runs
//! `num_passes` times per token with shared weights; every pass restarts from
//! the prelude output and reads a gated message from the previous pass. The
//! engine sees the unrolled visit sequence (prelude, core x passes, coda), so
//! each visit owns its KV cache and conv state while the loader points the
//! repeated visits at the same physical weights.
//!
//! Attention has no RoPE: a rank-`d_rel` relative-position bias over the last
//! `rel_extent` keys, per-token learned sinks, a sigmoid output gate, L2
//! clipping of Q/K, a gamma-free V norm and log-length query scaling. Each
//! physical layer is sliding-window (`sliding_window_size`) unless it is one of
//! the full-attention layers missing from `local_layer_ids`. Sublayer inputs
//! come from block attention residuals instead of a plain residual stream; the
//! MoE is 128 experts, top-8, sigmoid-routed with a bias and a softcapped SwiGLU.

use anyhow::{Context, Result, ensure};
use serde_json::Value;

use super::super::{LayerType, ModelConfig, finalize_config};

/// Inkling-specific architecture numbers the shared `ModelConfig` has no field
/// for. Visit indices are positions in the unrolled sequence the engine runs.
#[derive(Debug, Clone, PartialEq)]
pub struct Inkling2Config {
    pub physical_layers: usize,
    /// Physical layers with sliding-window attention; the rest are full.
    pub local_physical: Vec<usize>,
    pub core_start: usize,
    pub core_end: usize,
    pub num_passes: usize,
    pub loop_gate_rank: usize,
    pub unpadded_vocab_size: usize,
    pub d_rel: usize,
    pub rel_extent: usize,
    pub log_scaling_n_floor: usize,
    pub log_scaling_alpha: f32,
    pub qk_l2_clip_threshold: f32,
    pub rel_l2_clip_threshold: f32,
    pub sconv_kernel_size: usize,
    pub attn_res_block_size: usize,
    pub mlp_gate_softcap: f32,
    pub mlp_up_softcap: f32,
}

impl Inkling2Config {
    fn core_len(&self) -> usize {
        self.core_end - self.core_start
    }

    /// Number of visits in the unrolled sequence.
    pub fn num_visits(&self) -> usize {
        self.physical_layers + (self.num_passes - 1) * self.core_len()
    }

    /// Physical layer whose weights visit `v` runs.
    pub fn visit_physical(&self, v: usize) -> usize {
        let core_visits_end = self.core_start + self.num_passes * self.core_len();
        if v < self.core_start {
            v
        } else if v < core_visits_end {
            self.core_start + (v - self.core_start) % self.core_len()
        } else {
            v - core_visits_end + self.core_end
        }
    }

    /// Loop pass of visit `v` (0 outside the core and on the first pass).
    pub fn visit_pass(&self, v: usize) -> usize {
        let core_visits_end = self.core_start + self.num_passes * self.core_len();
        if (self.core_start..core_visits_end).contains(&v) {
            (v - self.core_start) / self.core_len()
        } else {
            0
        }
    }
}

fn get<'a>(raw: &'a Value, key: &str) -> Result<&'a Value> {
    raw.get(key)
        .with_context(|| format!("inkling2 config missing required field `{key}`"))
}

fn usize_of(raw: &Value, key: &str) -> Result<usize> {
    get(raw, key)?
        .as_u64()
        .map(|n| n as usize)
        .with_context(|| format!("inkling2 config field `{key}` must be an unsigned integer"))
}

fn f64_of(raw: &Value, key: &str) -> Result<f64> {
    get(raw, key)?
        .as_f64()
        .with_context(|| format!("inkling2 config field `{key}` must be numeric"))
}

fn bool_of(raw: &Value, key: &str) -> Result<bool> {
    get(raw, key)?
        .as_bool()
        .with_context(|| format!("inkling2 config field `{key}` must be a boolean"))
}

fn usize_list(raw: &Value, key: &str) -> Result<Vec<usize>> {
    get(raw, key)?
        .as_array()
        .with_context(|| format!("inkling2 config field `{key}` must be an array"))?
        .iter()
        .map(|v| {
            v.as_u64()
                .map(|n| n as usize)
                .with_context(|| format!("inkling2 `{key}` entries must be unsigned integers"))
        })
        .collect()
}

pub(crate) fn parse_inkling2_loop(raw: &Value) -> Result<ModelConfig> {
    let text = get(raw, "text_config")?;
    let lp = get(text, "loop_config")?;

    let physical_layers = usize_of(text, "num_hidden_layers")?;
    let ink = Inkling2Config {
        physical_layers,
        local_physical: usize_list(text, "local_layer_ids")?,
        core_start: usize_of(lp, "core_start")?,
        core_end: usize_of(lp, "core_end")?,
        num_passes: usize_of(lp, "num_passes")?,
        loop_gate_rank: usize_of(lp, "gate_rank")?,
        unpadded_vocab_size: usize_of(text, "unpadded_vocab_size")?,
        d_rel: usize_of(text, "d_rel")?,
        rel_extent: usize_of(text, "rel_extent")?,
        log_scaling_n_floor: usize_of(text, "log_scaling_n_floor")?,
        log_scaling_alpha: f64_of(text, "log_scaling_alpha")? as f32,
        qk_l2_clip_threshold: f64_of(text, "qk_l2_clip_threshold")? as f32,
        rel_l2_clip_threshold: f64_of(text, "rel_l2_clip_threshold")? as f32,
        sconv_kernel_size: usize_of(text, "sconv_kernel_size")?,
        attn_res_block_size: usize_of(text, "n_layers_per_attn_res_block")?,
        mlp_gate_softcap: f64_of(text, "mlp_gate_softcap")? as f32,
        mlp_up_softcap: f64_of(text, "mlp_up_softcap")? as f32,
    };

    // The variants below are different models; refuse them instead of running
    // them with the wrong layout.
    ensure!(
        !bool_of(lp, "is_varloop")?,
        "inkling2 variable-loop checkpoints are not implemented"
    );
    let block = ink.attn_res_block_size;
    ensure!(
        block > 0
            && ink.core_start > 0
            && ink.core_start < ink.core_end
            && ink.core_end <= physical_layers
            && ink.core_start.is_multiple_of(block)
            && ink.core_end.is_multiple_of(block)
            && physical_layers.is_multiple_of(block),
        "inkling2 loop core [{}, {}) must be block-aligned (block {block}) inside {physical_layers} layers",
        ink.core_start,
        ink.core_end
    );
    ensure!(
        ink.num_passes >= 2,
        "inkling2 fixed loop needs at least 2 passes, got {}",
        ink.num_passes
    );
    ensure!(
        usize_of(text, "n_shared_experts")? == 0,
        "inkling2 shared experts are not implemented"
    );
    ensure!(
        bool_of(text, "use_gate_bias")? && bool_of(text, "use_value_norm_no_gamma")?,
        "inkling2 use_gate_bias and use_value_norm_no_gamma must be true"
    );
    ensure!(
        !bool_of(text, "q_bias")? && !bool_of(text, "o_bias")?,
        "inkling2 q_bias/o_bias are not implemented"
    );
    let window = usize_of(text, "sliding_window_size")?;
    // The attention kernels apply the relative bias to every key of a sliding
    // window, which holds only while the window and the bias extent agree.
    ensure!(
        window == ink.rel_extent,
        "inkling2 sliding_window_size {window} must equal rel_extent {}",
        ink.rel_extent
    );
    ensure!(
        ink.local_physical.iter().all(|&l| l < physical_layers),
        "inkling2 local_layer_ids must index physical layers"
    );
    let (heads, kv_heads, head_dim) = (
        usize_of(text, "num_attention_heads")?,
        usize_of(text, "num_key_value_heads")?,
        usize_of(text, "head_dim")?,
    );
    ensure!(
        (heads, kv_heads, head_dim)
            == (
                usize_of(text, "swa_num_attention_heads")?,
                usize_of(text, "swa_num_key_value_heads")?,
                usize_of(text, "swa_head_dim")?,
            ),
        "inkling2 sliding and full attention must share head geometry"
    );

    let visits = ink.num_visits();
    let normalized = serde_json::json!({
        "model_type": "inkling2_loop",
        "hidden_size": usize_of(text, "hidden_size")?,
        "num_hidden_layers": visits,
        "intermediate_size": usize_of(text, "intermediate_size")?,
        "moe_intermediate_size": usize_of(text, "intermediate_size")?,
        "vocab_size": usize_of(text, "vocab_size")?,
        "num_attention_heads": heads,
        "num_key_value_heads": kv_heads,
        "head_dim": head_dim,
        "num_experts": usize_of(text, "n_routed_experts")?,
        "num_experts_per_tok": usize_of(text, "num_experts_per_tok")?,
        "norm_topk_prob": true,
        "sliding_window": window,
        "max_position_embeddings": usize_of(text, "model_max_length")?,
        "rms_norm_eps": f64_of(text, "rms_norm_eps")?,
        "eos_token_id": usize_of(raw, "eos_token_id")?,
        "tie_word_embeddings": false,
    });
    let mut config: ModelConfig =
        serde_json::from_value(normalized).context("Failed to parse inkling2 config")?;

    // Batched decode/verify only treat FullAttention as attention (as for
    // Gemma-4 and Ricursive): label every visit full and list the sliding ones.
    config.layer_types = vec![LayerType::FullAttention; visits];
    config.sliding_attention_layers = (0..visits)
        .filter(|&v| ink.local_physical.contains(&ink.visit_physical(v)))
        .collect();
    config.partial_rotary_factor = 0.0;
    config.rotary_dim = 0;
    config.attn_gated = true;
    config.qk_norm_type = "l2_clip".to_string();
    config.scoring_func = "sigmoid".to_string();
    config.use_routing_bias = true;
    config.routed_scaling_factor = f64_of(text, "route_scale")?;
    config.final_logit_softcapping = f64_of(text, "final_logit_softcapping")? as f32;
    // Layer keys are `model.llm.layers.<physical>`: callers index by
    // `Inkling2Config::visit_physical`, never by visit.
    config.weight_prefix = "model.llm".to_string();

    let mtp = get(raw, "mtp_config")?;
    config.dspark_block_size = usize_of(mtp, "block_size")?;
    config.dspark_markov_rank = usize_of(mtp, "markov_rank")?;
    config.dspark_target_layer_ids = usize_list(mtp, "target_layer_ids")?;
    ensure!(
        config.dspark_target_layer_ids.iter().all(|&v| v <= visits),
        "inkling2 mtp target_layer_ids must index visits (0..={visits})"
    );
    config.mtp_num_hidden_layers = 0;
    config.num_mtp_modules = 0;
    config.mtp_transformer_layers = 0;

    config.inkling2 = Some(ink);
    finalize_config(&mut config, raw)?;
    Ok(config)
}

#[cfg(test)]
#[path = "inkling2_tests.rs"]
mod tests;
