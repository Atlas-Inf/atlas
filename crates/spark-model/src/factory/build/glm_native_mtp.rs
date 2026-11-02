// SPDX-License-Identifier: AGPL-3.0-only
//! Native GLM MTP factory admission; target verification stays canonical.
use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};
pub(crate) fn native_mtp_requested(
    use_speculative: bool,
    dflash: bool,
    self_speculative: bool,
) -> bool {
    use_speculative && !dflash && !self_speculative
}

/// Native GLM MTP shares the released target verifier; it cannot silently
/// fall back to the ordinary BF16 target head when one of that lane's
/// rank-symmetric capabilities is missing.  Keep this check before NCCL-side
/// model construction completes so every rank fails with the same reason.
pub(crate) fn ensure_native_mtp_target_prerequisites(config: &ModelConfig) -> Result<()> {
    ensure_native_mtp_topology(config)?;
    anyhow::ensure!(
        crate::layers::canonical_verify::enabled(),
        "native GLM MTP requires ATLAS_GLM_CANONICAL_VERIFY=1 with ATLAS_W4A16_TC=1"
    );
    anyhow::ensure!(
        std::env::var("ATLAS_GLM_HC_BF16").as_deref() == Ok("1"),
        "native GLM MTP requires ATLAS_GLM_HC_BF16=1"
    );
    anyhow::ensure!(
        std::env::var("ATLAS_GLM_VERIFY_VOCAB_SPLIT").as_deref() == Ok("1"),
        "native GLM MTP requires ATLAS_GLM_VERIFY_VOCAB_SPLIT=1"
    );
    anyhow::ensure!(
        std::env::var("ATLAS_GLM_LM_HEAD_MXFP8").as_deref() == Ok("1"),
        "native GLM MTP requires ATLAS_GLM_LM_HEAD_MXFP8=1"
    );
    Ok(())
}

/// The model and the rank geometry native GLM MTP serves: `glm5_next` on
/// aligned TP2/EP2 or TP4/EP4, this rank inside it.
fn ensure_native_mtp_topology(config: &ModelConfig) -> Result<()> {
    anyhow::ensure!(
        config.model_type == "glm5_next",
        "native GLM MTP target prerequisites require glm5_next (got {})",
        config.model_type
    );
    anyhow::ensure!(
        crate::weight_loader::glm5_mode::supported_parallelism(
            config.ep_world_size,
            config.tp_world_size,
            config.ep_world_size,
        ) && config.tp_rank == config.ep_rank
            && config.tp_rank < config.tp_world_size,
        "native GLM MTP target prerequisites require aligned TP2/EP2 or TP4/EP4 (got TP{} / EP{})",
        config.tp_world_size,
        config.ep_world_size
    );
    Ok(())
}

/// Validate the rank-symmetric capability for the native GLM appended
/// predictor.  The body is loaded only on rank 0, but the marker and shared
/// full-vocabulary tensors must be present on every rank before the proposer
/// can be installed after the model is built.
pub(crate) fn native_mtp_capability(
    config: &ModelConfig,
    store: &WeightStore,
    use_speculative: bool,
    mtp_vocab_size: u32,
) -> Result<bool> {
    if config.model_type != "glm5_next" || !use_speculative {
        return Ok(false);
    }
    if config.num_mtp_modules == 0 {
        return Ok(false);
    }
    anyhow::ensure!(
        mtp_vocab_size == 0,
        "GLM-5 native MTP requires the full vocabulary head (set --mtp-vocab=0; got {})",
        mtp_vocab_size
    );
    anyhow::ensure!(
        config.num_mtp_modules == 1,
        "GLM-5 native MTP supports exactly one appended predictor layer (got {})",
        config.num_mtp_modules
    );
    for marker in crate::weight_loader::glm5::glm5_mtp_marker_names(config) {
        anyhow::ensure!(
            store.contains(&marker),
            "GLM-5 native MTP is advertised but rank {} is missing marker weight {marker}",
            config.tp_rank
        );
    }
    let embed_name = format!("{}.embed_tokens.weight", config.weight_prefix);
    let embed = store
        .get(&embed_name)
        .with_context(|| format!("GLM-5 native MTP: missing {embed_name}"))?;
    let lm_head = store
        .get("lm_head.weight")
        .context("GLM-5 native MTP: missing lm_head.weight")?;
    validate_full_bf16_head("embed_tokens.weight", embed, config)?;
    anyhow::ensure!(
        matches!(lm_head.dtype, WeightDtype::BF16 | WeightDtype::FP8E4M3)
            && lm_head.shape.len() == 2
            && lm_head.shape[0] >= config.vocab_size
            && lm_head.shape[1] == config.hidden_size
            && lm_head.ptr != DevicePtr::NULL,
        "native GLM MTP requires the publisher full-vocabulary BF16 or MXFP8 head"
    );
    Ok(true)
}

fn validate_full_bf16_head(name: &str, weight: &WeightTensor, config: &ModelConfig) -> Result<()> {
    anyhow::ensure!(
        weight.dtype == WeightDtype::BF16,
        "GLM-5 native MTP requires full BF16 {name}, got {:?}",
        weight.dtype
    );
    anyhow::ensure!(
        weight.shape.len() == 2
            && weight.shape[0] >= config.vocab_size
            && weight.shape[1] == config.hidden_size,
        "GLM-5 native MTP requires full {name} shape [>= {}, {}], got {:?}",
        config.vocab_size,
        config.hidden_size,
        weight.shape
    );
    anyhow::ensure!(
        weight.ptr != DevicePtr::NULL,
        "GLM-5 native MTP requires a resident {name} tensor"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn native_target_topology_covers_every_pair_and_four_rank_member() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".into();
        for world in [2, 4] {
            config.tp_world_size = world;
            config.ep_world_size = world;
            for rank in 0..world {
                config.tp_rank = rank;
                config.ep_rank = rank;
                ensure_native_mtp_topology(&config).unwrap();
                config.ep_rank = (rank + 1) % world;
                assert!(ensure_native_mtp_topology(&config).is_err());
            }
            config.tp_rank = world;
            config.ep_rank = world;
            assert!(ensure_native_mtp_topology(&config).is_err());
        }
        config.tp_rank = 0;
        config.ep_rank = 0;
        for (tp, ep) in [(1, 1), (2, 4), (4, 2), (8, 8)] {
            config.tp_world_size = tp;
            config.ep_world_size = ep;
            assert!(ensure_native_mtp_topology(&config).is_err());
        }
    }

    #[test]
    fn native_mtp_accepts_padded_checkpoint_head_but_rejects_missing_vocab_rows() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "glm5_next".into();
        config.num_mtp_modules = 1;
        config.hidden_size = 8;
        config.vocab_size = 10;
        let markers = crate::weight_loader::glm5::glm5_mtp_marker_names(&config);
        for (head_rows, valid) in [(12, true), (10, true), (9, false)] {
            let mut weights = std::collections::HashMap::new();
            for name in &markers {
                weights.insert(
                    name.clone(),
                    WeightTensor {
                        ptr: DevicePtr(4096),
                        dtype: WeightDtype::BF16,
                        shape: vec![8],
                    },
                );
            }
            weights.insert(
                format!("{}.embed_tokens.weight", config.weight_prefix),
                WeightTensor {
                    ptr: DevicePtr(8192),
                    dtype: WeightDtype::BF16,
                    shape: vec![12, 8],
                },
            );
            weights.insert(
                "lm_head.weight".into(),
                WeightTensor {
                    ptr: DevicePtr(12288),
                    dtype: WeightDtype::FP8E4M3,
                    shape: vec![head_rows, 8],
                },
            );
            let store = WeightStore::from_map(weights);
            assert_eq!(
                native_mtp_capability(&config, &store, true, 0).is_ok(),
                valid
            );
        }
    }
}
