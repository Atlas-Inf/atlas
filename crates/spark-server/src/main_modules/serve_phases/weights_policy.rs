// SPDX-License-Identifier: AGPL-3.0-only

use super::*;

pub(crate) fn quant_multiplier(config: &ModelConfig) -> Option<f64> {
    // Both Qwen3.8-Flash-Next NVFP4 packs measure device-after-load / on-disk
    // at ~1.15: nvidia pack 84.0 GB / 73.33 GB (2026-09-15 serve log,
    // docs/porting/QWEN38_NVIDIA_PORT_LOG_2026-09-15.md §B.2), RadixArk pack
    // ~90.4 GB peak / 78.19 GB on-disk (gb10 flash-next MODEL.toml memory
    // note). The generic 1.3 over-projects by ~10 GB — that is what refused
    // util >=0.85 + MTP on GB10 (§B.2 rows 014/015) and every boot on a 96 GB
    // Strix pool. 1.2 keeps ~4.5% margin over the measured ratio.
    if config.model_type == "qwen4_exp" {
        Some(1.2)
    } else if config.model_type == "glm5_next" {
        // The EP loader has already removed the remote half of the routed
        // experts. ModelOpt tensors are kept packed/zero-copy and the GLM
        // loader deliberately omits transposed MoE copies, so the generic
        // 1.30x NVFP4 estimate is not representative. Keep 5% for CUDA
        // metadata and the few BF16 dense/shared weights quantized at load.
        Some(1.05)
    } else if config.model_type == "minimax_m2" || config.model_type == "step3p7" {
        Some(1.02)
    } else if config
        .quantization_config
        .as_ref()
        .is_some_and(|qc| qc.quant_method == "fp8")
    {
        Some(1.05)
    } else {
        None
    }
}

/// Tensors this model reads from disk at use time rather than holding.
///
/// Only the `qwen4_exp` PLE n-gram table so far. It is 51.2 B parameters —
/// 41% of that checkpoint — and is gathered by row (~2.5 KB per token), so
/// making it resident is both unnecessary and, on a 119 GB GB10, impossible:
/// the pre-flight lands at 163 GB with it and ~96 GB without.
///
/// Keyed on `ple_layer_ids` rather than on `model_type`, because it is the
/// presence of the tower that decides this, not the family name.
pub(super) fn demand_paged_patterns(config: &ModelConfig) -> Vec<String> {
    if config.model_type == "hy_v4_flash" {
        return vec![".engram_layer.multi_head_embedding.embedding.weight".to_string()];
    }
    if config.ple_layer_ids.is_empty() {
        return Vec::new();
    }
    vec![".ple.ple_embedding.ngram_embedding.shard_".to_string()]
}

pub(super) fn load_policy(
    args: &cli::ServeArgs,
    config: &ModelConfig,
) -> Result<(Vec<String>, bool)> {
    let loader =
        spark_model::factory::loader_for_config(config).context("weight-load policy lookup")?;
    let mut dead = loader.dead_namespaces(config);
    if args.dfly_speculative {
        anyhow::ensure!(config.model_type == "hy_v4_flash", "DFly requires HY4");
        dead.retain(|name| name != "model.dfly_layers.");
    }
    let slab =
        loader.slab_pack_weights(config) && config.tp_world_size <= 1 && config.ep_world_size <= 1;
    Ok((dead, slab))
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[test]
    fn dfly_weights_are_retained_only_with_the_explicit_mode() {
        let config = atlas_core::config::parse_config(include_str!(
            "../../../../atlas-core/src/config/fixtures/hy4/config.json"
        ))
        .unwrap();
        let disabled = cli::ServeArgs::parse_from(["spark", "unused"]);
        let (dead, _) = load_policy(&disabled, &config).unwrap();
        assert!(dead.iter().any(|name| name == "model.dfly_layers."));
        let enabled =
            cli::ServeArgs::parse_from(["spark", "unused", "--dfly-speculative", "--num-drafts=5"]);
        let (dead, _) = load_policy(&enabled, &config).unwrap();
        assert!(!dead.iter().any(|name| name == "model.dfly_layers."));
        assert!(dead.iter().any(|name| name == "vit."));
        let mut unsupported = config;
        unsupported.model_type = "unknown".into();
        assert!(load_policy(&enabled, &unsupported).is_err());
    }
}
