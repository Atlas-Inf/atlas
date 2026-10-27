// SPDX-License-Identifier: AGPL-3.0-only

//! Parser for DeepSeek-V4.1 (DeepSeek-V4.1-Flash, `DeepseekV41ForCausalLM`).
//!
//! V4.1 keeps V4's MLA-with-one-latent-head attention, grouped low-rank output,
//! mHC and sqrtsoftplus MoE, so the shared fields go through V4's parse. It is
//! its own family because the attention cache changes meaning, not just size:
//!
//! - every layer attends over a 128-token sliding window; layers with a
//!   compress ratio (2 or 1 here, 4/128 in V4) add compressed history on top;
//! - only `kv_source_layer_ids` write compressed KV and index keys, and every
//!   later compressing layer reads the most recent source's cache;
//! - only `index_source_layer_ids` run the indexer, and the layers after one
//!   reuse its top-k; from `candidate_source_layer_id` on, top-k is two-level;
//! - Engram n-gram tables add a gated lookup into the residual stream;
//! - three DSpark draft stages follow the backbone and attend over the
//!   window only.
//!
//! Running a V4.1 checkpoint through V4 code would compress at the wrong
//! layers and read the wrong caches while still producing tokens, so the two
//! are kept apart by `model_type`. The config nests the text model under
//! `text_config` beside a `vision_config`.
//!
//! Reference: `inference/model.py` of deepseek-ai/DeepSeek-V4.1-Flash-1017 (MIT).

use anyhow::{Context, Result, ensure};

use super::super::ModelConfig;
use super::deepseek_v4::parse_deepseek_family;

pub fn parse_deepseek_v41(raw: &serde_json::Value) -> Result<ModelConfig> {
    let mut flat = raw
        .get("text_config")
        .and_then(serde_json::Value::as_object)
        .context("deepseek_v41 config missing text_config")?
        .clone();
    // Top-level fields (quantization_config, token ids) fill in what the text
    // config does not carry; the vision tower is not part of the text model.
    for (key, value) in raw.as_object().into_iter().flatten() {
        if key != "text_config" && key != "vision_config" {
            flat.entry(key.clone()).or_insert_with(|| value.clone());
        }
    }
    let json = serde_json::Value::Object(flat).to_string();
    let config = parse_deepseek_family(&json, "deepseek_v41")?;
    validate_shared_attention(&config)?;
    // The pad id needs the tokenizer; the table geometry does not.
    config.deepseek_v41_engram(0)?;
    Ok(config)
}

fn validate_shared_attention(c: &ModelConfig) -> Result<()> {
    let layers = c.num_hidden_layers;
    let ratios = &c.compress_ratios;
    ensure!(
        ratios.len() == layers + c.num_mtp_modules,
        "compress_ratios has {} entries; {layers} layers and {} DSpark stages need {}",
        ratios.len(),
        c.num_mtp_modules,
        layers + c.num_mtp_modules
    );
    ensure!(
        ratios[layers..].iter().all(|&r| r == 0),
        "DSpark stages attend over the sliding window only; compress_ratios must be 0 past layer {}",
        layers - 1
    );
    for (name, sources) in [
        ("kv_source_layer_ids", &c.kv_source_layer_ids),
        ("index_source_layer_ids", &c.index_source_layer_ids),
    ] {
        ensure!(
            sources.windows(2).all(|w| w[0] < w[1])
                && sources.iter().all(|&l| l < layers && ratios[l] > 0),
            "{name} must be increasing compressing layers below {layers}"
        );
    }
    for layer in (0..layers).filter(|&l| ratios[l] > 0) {
        for (what, source) in [
            ("compressed KV", c.kv_source_layer(layer)),
            ("top-k", c.index_source_layer(layer)),
        ] {
            let source = source.with_context(|| {
                format!("layer {layer} compresses but no earlier layer produces its {what}")
            })?;
            ensure!(
                ratios[source] == ratios[layer],
                "layer {layer} compresses {}:1 but would read its {what} from layer {source}, \
                 which compresses {}:1",
                ratios[layer],
                ratios[source]
            );
        }
    }
    if let Some(source) = c.candidate_source_layer_id {
        ensure!(
            c.index_source_layer_ids.contains(&source),
            "candidate_source_layer_id {source} is not an index source"
        );
        ensure!(
            c.candidate_topk_blocks > 0 && c.candidate_block_size > 0,
            "candidate block filtering needs candidate_topk_blocks and candidate_block_size"
        );
    }
    ensure!(
        c.index_n_heads > 0 && c.index_head_dim > 0 && c.index_topk > 0,
        "the indexer needs index_n_heads, index_head_dim and index_topk"
    );
    if c.dspark_n_routed_experts > 0 {
        ensure!(
            (1..=c.dspark_n_routed_experts).contains(&c.dspark_num_experts_per_tok),
            "dspark_num_experts_per_tok must be in 1..={}",
            c.dspark_n_routed_experts
        );
    }
    Ok(())
}

impl ModelConfig {
    /// The layer whose compressed KV `layer` attends over: the most recent
    /// `kv_source_layer_ids` entry at or before it, or the layer itself when
    /// the config declares no sources (DeepSeek-V4). `None` for a layer that
    /// does not compress.
    pub fn kv_source_layer(&self, layer: usize) -> Option<usize> {
        if self.compress_ratios.get(layer).copied().unwrap_or(0) == 0 {
            return None;
        }
        if self.kv_source_layer_ids.is_empty() {
            return Some(layer);
        }
        self.kv_source_layer_ids
            .iter()
            .rev()
            .find(|&&s| s <= layer)
            .copied()
    }

    /// The layer whose indexer top-k `layer` reuses: the most recent
    /// `index_source_layer_ids` entry at or before it. `None` for a layer that
    /// does not compress, or when the config declares no index sources.
    pub fn index_source_layer(&self, layer: usize) -> Option<usize> {
        if self.compress_ratios.get(layer).copied().unwrap_or(0) == 0 {
            return None;
        }
        self.index_source_layer_ids
            .iter()
            .rev()
            .find(|&&s| s <= layer)
            .copied()
    }
}

#[cfg(test)]
#[path = "deepseek_v41_tests.rs"]
mod tests;
