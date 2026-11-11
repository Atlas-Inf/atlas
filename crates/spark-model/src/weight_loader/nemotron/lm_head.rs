// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H LM-head resolution: tensor naming under `weight_prefix` plus
//! the FP8-E4M3 dequant path the omni checkpoint needs. Split from
//! `nemotron.rs` (500-LoC cap).

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore};

use super::NemotronHWeightLoader;
use crate::weight_map::{DenseWeight, dense, dequant_fp8_to_bf16};

impl NemotronHWeightLoader {
    /// Load the LM head for a Nemotron-H checkpoint.
    ///
    /// `{head}.weight` wins when present; an FP8 E4M3 head (Nemotron 3.5
    /// Super VL ships `language_model.lm_head` that way) is dequanted to
    /// BF16 at load via its per-tensor `weight_scale`. When the head tensor
    /// is absent the embeddings stand in ONLY for tied checkpoints — on an
    /// untied checkpoint that fallback would silently serve the embedding
    /// table as the classifier, so it now bails.
    pub(super) fn load_lm_head_impl(
        store: &WeightStore,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
    ) -> Result<DenseWeight> {
        let head = lm_head_name(&config.weight_prefix)?;
        let weight_name = format!("{head}.weight");
        if store.contains(&weight_name) {
            if store.get(&weight_name)?.dtype == WeightDtype::FP8E4M3 {
                anyhow::ensure!(
                    store.contains(&format!("{head}.weight_scale")),
                    "{weight_name} is FP8 E4M3 but {head}.weight_scale is missing \
                     — refusing to serve the raw FP8 bytes as the LM head"
                );
                tracing::info!("LM head: FP8 E4M3 per-tensor → BF16 at load ({weight_name})");
                return dequant_fp8_to_bf16(store, &head, gpu);
            }
            return dense(store, &weight_name);
        }
        if config.tie_word_embeddings {
            return dense(
                store,
                &format!("{}.embeddings.weight", config.weight_prefix),
            );
        }
        anyhow::bail!(
            "untied checkpoint has no {weight_name} — refusing to serve the \
             embeddings as the LM head"
        )
    }
}

/// LM-head tensor name for a Nemotron-H `weight_prefix`: the head lives OUTSIDE
/// the `backbone` namespace, so strip the trailing `backbone` component and
/// append `lm_head`. `""`/`"backbone"` → `lm_head` (Nano / Super / Lightning /
/// Puzzle unprefixed head); `"language_model.backbone"` →
/// `language_model.lm_head` (the omni wrapper). Any other suffix is a weight
/// layout this loader doesn't know — fail closed rather than silently serving
/// the embeddings as the head on an untied checkpoint.
fn lm_head_name(weight_prefix: &str) -> Result<String> {
    if weight_prefix.is_empty() {
        return Ok("lm_head".to_string());
    }
    weight_prefix
        .strip_suffix("backbone")
        .map(|stem| format!("{stem}lm_head"))
        .context("nemotron weight_prefix must end in \"backbone\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lm_head_name_maps_prefixes() {
        assert_eq!(lm_head_name("").unwrap(), "lm_head");
        assert_eq!(lm_head_name("backbone").unwrap(), "lm_head");
        assert_eq!(
            lm_head_name("language_model.backbone").unwrap(),
            "language_model.lm_head"
        );
        assert!(lm_head_name("model.layers").is_err());
        assert!(lm_head_name("backbonex").is_err());
    }
}
