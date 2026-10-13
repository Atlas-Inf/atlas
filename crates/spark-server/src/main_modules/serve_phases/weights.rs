// SPDX-License-Identifier: AGPL-3.0-only

//! Weight-store loading: main checkpoint, prefix auto-detect, DFlash drafter.

use std::path::Path;

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

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

pub(crate) fn load_weight_store(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    model_dir: &Path,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    ep_rank: usize,
    ep_size: usize,
    oom_reserve_bytes: usize,
) -> Result<spark_runtime::weights::WeightStore> {
    use spark_runtime::weights::WeightLoader;
    let mult = quant_multiplier(config);
    let glm_mtp_distributed = spark_model::speculative::glm_repair_policy::mtp_distributed();
    let unused_mtp_prefix = unused_glm_mtp_prefix(config, args.speculative);
    if glm_mtp_distributed {
        anyhow::ensure!(
            config.model_type == "glm5_next"
                && args.speculative
                && args.world_size == 2
                && args.tp_size == 2
                && args.ep_size == 2,
            "ATLAS_GLM_MTP_DISTRIBUTED=1 requires speculative GLM with overlapping \
             TP=EP=world=2 (model={}, speculative={}, TP={}, EP={}, world={})",
            config.model_type,
            args.speculative,
            args.tp_size,
            args.ep_size,
            args.world_size,
        );
        tracing::info!(
            "GLM-5 split-vocabulary MTP enabled: appended body is mirrored on both ranks"
        );
    }

    // GGUF checkpoints are dequantized to BF16 by a dedicated loader; take that
    // path whenever a .gguf file is present (fast/safetensors loaders can't read it).
    if spark_runtime::weights::find_gguf(model_dir).is_some() {
        tracing::info!("Detected GGUF weights; using GgufLoader (GPU dequant → BF16)");
        let mut loader = if ep_size > 1 {
            spark_runtime::weights::GgufLoader::with_ep(ep_rank, ep_size, config.num_experts)
        } else {
            spark_runtime::weights::GgufLoader::new()
        };
        loader.peak_memory_multiplier = mult;
        let store = loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights (GGUF loader)")?;
        tracing::info!("Loaded {} weight tensors (GGUF)", store.len());
        return Ok(store);
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
    fn demand_paged_patterns(config: &ModelConfig) -> Vec<String> {
        if config.ple_layer_ids.is_empty() {
            return Vec::new();
        }
        vec![".ple.ple_embedding.ngram_embedding.shard_".to_string()]
    }

    let use_fast_load =
        !args.no_fast_load && std::env::var("ATLAS_FAST_LOAD").ok().as_deref() != Some("0");
    let store = if use_fast_load {
        #[cfg(unix)]
        {
            tracing::info!("Using fast weight loader (O_DIRECT + pipelined read/copy)");
            let mut loader = if ep_size > 1 {
                spark_runtime::fast_weights::FastSafetensorsLoader::with_ep(
                    ep_rank,
                    ep_size,
                    config.num_experts,
                )
            } else {
                spark_runtime::fast_weights::FastSafetensorsLoader::new()
            };
            if config.model_type == "glm5_next" && args.speculative {
                let prefix = format!(".layers.{}.", config.num_hidden_layers);
                if glm_mtp_distributed {
                    loader.replicated_expert_prefix = Some(prefix);
                    tracing::info!(
                        "GLM-5 distributed MTP: appended-layer experts are replicated on both ranks"
                    );
                } else {
                    loader.rank0_only_expert_prefix = Some(prefix);
                    tracing::info!(
                        "GLM-5 MTP: appended-layer experts are rank-0-only and fully replicated"
                    );
                }
            }
            loader.peak_memory_multiplier = mult;
            loader.expert_tp = config.expert_tp;
            loader.demand_paged_patterns = demand_paged_patterns(config);
            if !loader.demand_paged_patterns.is_empty() {
                tracing::info!(
                    "Demand-paged (never resident, read by row at use time): {:?}",
                    loader.demand_paged_patterns
                );
            }
            loader.skip_activation_scales = skip_activation_scales(config);
            loader.skip_mtp = skip_mtp(config, args);
            loader.skip_layer_prefix = unused_mtp_prefix.clone();
            loader.prefetch_shards = args.fast_load_prefetch_shards
                || std::env::var("ATLAS_FAST_LOAD_PREFETCH_SHARDS")
                    .ok()
                    .is_some_and(|v| v == "1" || v.eq_ignore_ascii_case("true"));
            if loader.prefetch_shards {
                tracing::info!("Fast weight loader shard prefetch/readahead enabled");
            }
            loader
                .load(model_dir, gpu, oom_reserve_bytes)
                .context("Failed to load model weights (fast loader)")?
        }
        #[cfg(not(unix))]
        {
            anyhow::bail!("--fast-load requires a Unix host (needs O_DIRECT / posix_fadvise)");
        }
    } else {
        anyhow::ensure!(
            !config.expert_tp,
            "ATLAS_GLM_EXPERT_TP=1 needs the fast weight loader (ATLAS_FAST_LOAD)"
        );
        let mut loader = if ep_size > 1 {
            spark_runtime::weights::SafetensorsLoader::with_ep(ep_rank, ep_size, config.num_experts)
        } else {
            spark_runtime::weights::SafetensorsLoader::new()
        };
        if config.model_type == "glm5_next" && args.speculative {
            let prefix = format!(".layers.{}.", config.num_hidden_layers);
            if glm_mtp_distributed {
                loader.replicated_expert_prefix = Some(prefix);
            } else {
                loader.rank0_only_expert_prefix = Some(prefix);
            }
        }
        loader.peak_memory_multiplier = mult;
        loader.skip_activation_scales = skip_activation_scales(config);
        loader.skip_mtp = skip_mtp(config, args);
        loader.skip_layer_prefix = unused_mtp_prefix;
        loader
            .load(model_dir, gpu, oom_reserve_bytes)
            .context("Failed to load model weights")?
    };
    tracing::info!("Loaded {} weight tensors", store.len());
    Ok(store)
}

pub(crate) fn load_dflash_drafter(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<
    Option<(
        spark_runtime::weights::WeightStore,
        spark_model::weight_loader::DflashConfig,
    )>,
> {
    use spark_runtime::weights::WeightLoader;
    if !args.dflash {
        return Ok(None);
    }
    let drafter_id = args
        .draft_model
        .clone()
        .or_else(|| ptx_set.dflash.as_ref().map(|d| d.draft_model.to_string()))
        .context(
            "--dflash set but no drafter HF id provided: pass --draft-model <ID> \
             or use a target whose MODEL.toml has a [dflash] section",
        )?;
    tracing::info!("DFlash: resolving drafter '{drafter_id}'");
    let drafter_dir =
        crate::model_resolver::resolve_model_dir(&drafter_id, args.cache_dir.as_deref())
            .context("Failed to resolve DFlash drafter checkpoint")?;
    let drafter_config_json = std::fs::read_to_string(drafter_dir.join("config.json"))
        .with_context(|| {
            format!(
                "Failed to read drafter config.json at {}",
                drafter_dir.display()
            )
        })?;
    let drafter_config =
        spark_model::weight_loader::dflash_loader::parse_dflash_config(&drafter_config_json)?;
    // Only the head proposes. A worker keeps the drafter's config (its target
    // capture layers shape the verify rows and SSM pools) but no weights, so
    // no proposer is built there and its memory goes to the KV pool. Under
    // ATLAS_GLM_DRAFT_TP the worker reads half of each propose, from the
    // same weights.
    let worker_shares = spark_model::layers::dflash_head::rank_split::requested()?.is_some();
    if args.rank != 0 && !worker_shares {
        tracing::info!(
            "DFlash: rank {} keeps the drafter config only; the head proposes",
            args.rank
        );
        return Ok(Some((
            spark_runtime::weights::WeightStore::empty(),
            drafter_config,
        )));
    }
    let mut loader = spark_runtime::weights::SafetensorsLoader::new();
    loader.peak_memory_multiplier = None;
    let mut drafter_store = loader
        .load(&drafter_dir, gpu, 0)
        .context("Failed to load DFlash drafter weights")?;
    // Before the KV pool is sized, so the freed BF16 bytes become KV blocks.
    if spark_model::weight_loader::dflash_preshrink::requested() {
        spark_model::weight_loader::dflash_preshrink::preshrink(
            &mut drafter_store,
            drafter_config.num_hidden_layers,
            gpu,
        )?;
    }
    tracing::info!(
        "DFlash drafter store: {} tensors, {} bytes",
        drafter_store.len(),
        drafter_store.total_bytes()
    );
    Ok(Some((drafter_store, drafter_config)))
}

/// Startup-loaded LoRA adapter: its own WeightStore + parsed PEFT config.
/// One `LoraAdapterState` per repeated `--lora-adapter NAME=PATH`; each becomes
/// one resident pool slot. A single adapter is byte-identical to the v0 path.
pub(crate) struct LoraAdapterState {
    pub name: String,
    pub peft_config: atlas_core::config::PeftAdapterConfig,
    pub store: spark_runtime::weights::WeightStore,
}

/// Resolve + load every `--lora-adapter` into its own on-device `WeightStore`
/// (slot 0..N-1). Empty when no adapter is requested. Rejects >`--max-loras`
/// adapters and duplicate names up front.
pub(crate) fn load_lora_adapters(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
) -> Result<Vec<LoraAdapterState>> {
    if args.lora_adapter.is_empty() {
        return Ok(Vec::new());
    }
    if args.lora_adapter.len() > args.max_loras {
        anyhow::bail!(
            "--lora-adapter given {} times but --max-loras={} (pool has {} slots); \
             raise --max-loras or stage the extras on an $ATLAS_LORA_PEER",
            args.lora_adapter.len(),
            args.max_loras,
            args.max_loras,
        );
    }
    let mut states: Vec<LoraAdapterState> = Vec::with_capacity(args.lora_adapter.len());
    for (name, spec) in &args.lora_adapter {
        if states.iter().any(|s| &s.name == name) {
            anyhow::bail!("--lora-adapter name '{name}' given twice (names must be unique)");
        }
        tracing::info!("LoRA: resolving adapter '{name}' from '{spec}'");
        let adapter_dir =
            crate::model_resolver::resolve_adapter_dir(spec, args.cache_dir.as_deref())
                .context("Failed to resolve LoRA adapter")?;
        let cfg_path = adapter_dir.join("adapter_config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("Failed to read {}", cfg_path.display()))?;
        // Hard-error parser (atlas-core config/parsers/lora.rs) — scaling is read
        // per adapter (alpha/r, alpha/sqrt(r) under use_rslora), NEVER defaulted.
        let peft_config = atlas_core::config::parse_peft_adapter_config(&raw)
            .with_context(|| format!("Failed to parse {}", cfg_path.display()))?;
        if peft_config.r > args.max_lora_rank {
            anyhow::bail!(
                "LoRA adapter '{}' has r={} > --max-lora-rank {} — raise the flag \
                 (slot pool is rank-padded to it) or use a smaller adapter",
                name,
                peft_config.r,
                args.max_lora_rank,
            );
        }
        let store = spark_runtime::weights::adapter::load_adapter_safetensors(&adapter_dir, gpu, 0)
            .context("Failed to load LoRA adapter weights")?;
        tracing::info!(
            "LoRA adapter '{}': {} tensors, {} bytes loaded; r={}, alpha={}, \
             use_rslora={}, scaling={:.6}, target_modules={:?}",
            name,
            store.len(),
            store.total_bytes(),
            peft_config.r,
            peft_config.lora_alpha,
            peft_config.use_rslora,
            peft_config.scaling(),
            peft_config.target_modules,
        );
        states.push(LoraAdapterState {
            name: name.clone(),
            peft_config,
            store,
        });
    }
    Ok(states)
}

/// Whether this model's loader can skip the W4A4 `*.input_scale` activation
/// scales.
///
/// ModelOpt NVFP4 ships one 0-dim F32 scalar per quantized projection. On
/// Qwen3.8-Flash-Next that is ~74k four-byte allocations (48 layers x 512
/// experts x 3 projections), each taking a full allocation granule — GBs of
/// padding for values the w4a16 path never reads. The NVFP4 loader already
/// treats the key as optional (`if store.contains(..) else NULL`), so not
/// uploading them is identical to loading a checkpoint that never had them.
///
/// Deliberately an ALLOW-LIST, not a blanket skip: `step3p7` reads
/// `input_scale` on its own loader path, and silently withholding a tensor a
/// loader DOES read is exactly the class of bug that stays invisible until
/// the output is subtly wrong.
fn skip_activation_scales(config: &ModelConfig) -> bool {
    matches!(config.model_type.as_str(), "qwen4_exp")
}

/// Whether this model's loader builds no MTP head, so `mtp.*` need not be
/// uploaded at all.
///
/// Measured, not estimated: `mtp.*` on this checkpoint is 5.21 GB (31 tensors,
/// 5.03 GB of which is the stacked BF16 expert pair) — not the ~1.5 GB an
/// earlier note claimed. That is 3.5x the figure the original drop was argued
/// against, and on a box where 0.90 utilization is already near the boot floor
/// it is the difference between a usable KV cache and none.
fn skip_mtp(config: &ModelConfig, args: &cli::ServeArgs) -> bool {
    // qwen4_exp now BUILDS an MTP head, but only under `--speculative`
    // (`weight_loader/qwen4_exp/mtp.rs`). Without the flag the `mtp.*` upload is
    // 5.21 GB of BF16 held resident for nothing, which on a 119.6 GB unified box
    // comes straight out of the KV cache. With the flag it is the drafter, on
    // rank 0 only (`factory::build`): every other rank skips it as well.
    matches!(config.model_type.as_str(), "qwen4_exp") && (!args.speculative || config.ep_rank != 0)
}

#[cfg(test)]
mod tests {
    use super::{quant_multiplier, skip_mtp};
    use atlas_core::config::ModelConfig;
    use clap::Parser as _;

    /// qwen4_exp's `mtp.*` (5.21 GB) is uploaded only where it drafts: rank 0
    /// under `--speculative`.
    #[test]
    fn qwen4_exp_mtp_is_uploaded_on_the_drafting_rank_only() {
        let args = |more: &[&str]| {
            let argv = ["spark", "some/model"].iter().chain(more);
            crate::cli::ServeArgs::parse_from(argv)
        };
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        assert!(skip_mtp(&config, &args(&[])));
        assert!(!skip_mtp(&config, &args(&["--speculative"])));
        config.ep_rank = 1;
        assert!(skip_mtp(&config, &args(&["--speculative"])));
    }

    #[test]
    fn qwen4_exp_uses_measured_load_peak() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen4_exp".to_string();
        assert_eq!(quant_multiplier(&config), Some(1.2));
    }

    #[test]
    fn non_fp8_qwen35_keeps_generic_multiplier() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        config.model_type = "qwen3_5".to_string();
        assert_eq!(quant_multiplier(&config), None);
    }
}

// GLM's predictor is a physical appended layer, not the generic `mtp.*` tree.
// Without speculative decoding the factory never consumes any of its tensors.
fn unused_glm_mtp_prefix(config: &ModelConfig, speculative: bool) -> Option<String> {
    (config.model_type == "glm5_next" && !speculative).then(|| {
        format!(
            "{}.layers.{}.",
            config.weight_prefix, config.num_hidden_layers
        )
    })
}

#[cfg(test)]
mod unused_glm_mtp_tests {
    use super::*;

    #[test]
    fn unused_glm_mtp_policy_uses_actual_prefix_and_target_layer_count() {
        let mut config = ModelConfig::qwen3_next_80b_nvfp4();
        assert!(unused_glm_mtp_prefix(&config, false).is_none());
        config.model_type = "glm5_next".into();
        config.weight_prefix = "model.language_model".into();
        config.num_hidden_layers = 45;
        assert_eq!(
            unused_glm_mtp_prefix(&config, false).as_deref(),
            Some("model.language_model.layers.45.")
        );
        assert!(unused_glm_mtp_prefix(&config, true).is_none());
        config.weight_prefix = "fixture".into();
        config.num_hidden_layers = 3;
        assert_eq!(
            unused_glm_mtp_prefix(&config, false).as_deref(),
            Some("fixture.layers.3.")
        );
    }
}
