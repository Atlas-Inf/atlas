// SPDX-License-Identifier: AGPL-3.0-only

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use sha2::{Digest, Sha256};
use spark_runtime::graph_runtime::{
    CompatibilityRule, GraphFallbackReason, GraphFingerprint, GraphMode, GraphPhase, GraphPolicies,
    GraphRuntimeConfig, PhasePolicy, ShapeBucket, SpeculativeAlgorithm,
};
use std::path::Path;

use crate::cli;

pub(in crate::main_modules) fn resolve_graph_runtime_config(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    ptx: &atlas_kernels::TargetPtxSet,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    model_dir: &Path,
    drafter: Option<&spark_model::weight_loader::DflashConfig>,
) -> Result<GraphRuntimeConfig> {
    let mode: GraphMode = args
        .cuda_graph_mode
        .parse()
        .map_err(anyhow::Error::msg)
        .context("invalid --cuda-graph-mode")?;
    let token_limits = cli::graph_config::parse_bucket_limits(
        &args.cuda_graph_token_buckets,
        "--cuda-graph-token-buckets",
    )
    .map_err(anyhow::Error::msg)?;
    let request_limits = cli::graph_config::parse_bucket_limits(
        &args.cuda_graph_request_buckets,
        "--cuda-graph-request-buckets",
    )
    .map_err(anyhow::Error::msg)?;
    let shape_buckets = token_limits
        .into_iter()
        .flat_map(|token_limit| {
            request_limits
                .iter()
                .copied()
                .map(move |request_limit| ShapeBucket {
                    token_limit,
                    request_limit,
                })
        })
        .collect();
    let policies = policies(args, mode)?;
    let speculative_algorithm = speculative_algorithm(args, config, drafter)?;
    let environment =
        gpu.graph_environment()
            .unwrap_or_else(|| spark_runtime::graph_runtime::GraphEnvironment {
                device: format!("gpu{}:{}", args.gpu_ordinal, ptx.target.arch),
                cuda: "unavailable".to_string(),
                driver: "unavailable".to_string(),
            });
    let fingerprint = GraphFingerprint {
        runtime: cli::ATLAS_VERSION.to_string(),
        model: model_checksum(model_dir, config)?,
        kernel_build: kernel_build_id(ptx),
        device: format!("{}:{}", environment.device, ptx.target.arch),
        cuda: environment.cuda,
        driver: environment.driver,
        memory_layout: format!(
            "seq={};batch={};tokens={};block={};kv={:?};lm_head={};tp={};ep={}",
            args.max_seq_len,
            args.max_batch_size,
            args.max_prefill_tokens,
            args.block_size,
            args.kv_cache_dtype,
            args.lm_head_dtype,
            args.tp_size,
            args.ep_size,
        ),
    };
    if let Some(directory) = &args.cuda_graph_export_dir {
        std::fs::create_dir_all(directory).with_context(|| {
            format!("create CUDA graph export directory {}", directory.display())
        })?;
    }
    let prewarm_profile = args
        .cuda_graph_prewarm_profile
        .as_ref()
        .map(|path| {
            let bytes = std::fs::read(path)
                .with_context(|| format!("read CUDA graph prewarm profile {}", path.display()))?;
            spark_runtime::graph_runtime::GraphPrewarmProfile::from_json(&bytes)
                .with_context(|| format!("parse CUDA graph prewarm profile {}", path.display()))
        })
        .transpose()?;
    Ok(GraphRuntimeConfig {
        mode,
        speculative_algorithm,
        shape_buckets,
        policies,
        max_cache_entries: args.cuda_graph_cache_entries,
        max_cache_bytes: (args.cuda_graph_cache_mb as u64).saturating_mul(1024 * 1024),
        fingerprint,
        resource_generation: 1,
        compatibility_rules: compatibility_rules(),
        export_dir: args.cuda_graph_export_dir.clone(),
        prewarm_profile,
    })
}

/// Resolve the serve's speculative algorithm for the graph runtime.
///
/// `--dspark` is an explicit request and fails closed: it requires a
/// DFlash-family drafter whose checkpoint ships D-Spark artifacts. `--dflash`
/// keeps its historical behaviour of inferring D-Spark from the checkpoint
/// (drafter Markov head / `projector_type="dspark"`, or a target that declares
/// `dspark_block_size > 0`), so existing launches are unaffected.
fn speculative_algorithm(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    drafter: Option<&spark_model::weight_loader::DflashConfig>,
) -> Result<SpeculativeAlgorithm> {
    if args.dspark {
        if !drafter.is_some_and(|draft| draft.is_dspark()) {
            anyhow::bail!(
                "--dspark requires a DFlash-family drafter checkpoint that ships D-Spark \
                 artifacts (a Markov head or projector_type=\"dspark\"); pass --draft-model <ID>. \
                 A checkpoint-native D-Spark (target config dspark_block_size > 0) is enabled \
                 with --speculative/--dflash instead"
            );
        }
        return Ok(SpeculativeAlgorithm::Dspark);
    }
    Ok(if args.dflash {
        if config.dspark_block_size > 0 || drafter.is_some_and(|draft| draft.is_dspark()) {
            SpeculativeAlgorithm::Dspark
        } else {
            SpeculativeAlgorithm::Dflash
        }
    } else if args.speculative {
        SpeculativeAlgorithm::Mtp
    } else if args.self_speculative {
        SpeculativeAlgorithm::SelfSpeculative
    } else if args.ngram_speculative {
        SpeculativeAlgorithm::Ngram
    } else {
        SpeculativeAlgorithm::None
    })
}

fn policies(args: &cli::ServeArgs, mode: GraphMode) -> Result<GraphPolicies> {
    let enabled = mode != GraphMode::Disabled;
    let bytes = args
        .cuda_graph_cache_mb
        .checked_mul(1024 * 1024)
        .ok_or_else(|| anyhow::anyhow!("--cuda-graph-cache-mb overflows bytes"))?
        as u64;
    let quota = |percent: usize, capture: bool| PhasePolicy {
        max_entries: if enabled {
            (args.cuda_graph_cache_entries * percent / 100).max(1)
        } else {
            0
        },
        max_estimated_bytes: if enabled {
            (bytes * percent as u64 / 100).max(1)
        } else {
            0
        },
        capture_enabled: enabled && capture,
        replay_enabled: enabled && capture,
        prewarm_enabled: enabled && capture,
    };
    let segmented = matches!(mode, GraphMode::Breakable | GraphMode::Piecewise);
    GraphPolicies::new(
        quota(15, segmented),
        quota(30, true),
        quota(30, true),
        // Propose is the DFlash / D-Spark drafter's piecewise capture — the
        // speculative path itself, not an optional segment. It ran by default
        // before the graph runtime existed, so gating it to Breakable/Piecewise
        // silently turned every default-mode serve's drafter eager.
        quota(10, true),
        quota(15, true),
    )
    .map_err(anyhow::Error::msg)
}

fn compatibility_rules() -> Vec<CompatibilityRule> {
    let phases = vec![
        GraphPhase::Prefill,
        GraphPhase::Decode,
        GraphPhase::Verify,
        GraphPhase::Propose,
        GraphPhase::Fused,
    ];
    let modes = vec![GraphMode::Full, GraphMode::Breakable, GraphMode::Piecewise];
    vec![
        CompatibilityRule {
            id: "profile".to_string(),
            phases: phases.clone(),
            modes: modes.clone(),
            fallback_reason: GraphFallbackReason::Profiling,
            detail: "per-kernel profiling performs host synchronization".to_string(),
        },
        CompatibilityRule {
            id: "high_speed_swap".to_string(),
            phases,
            modes,
            fallback_reason: GraphFallbackReason::HighSpeedSwap,
            detail: "KV offload performs host I/O and dynamic memory registration".to_string(),
        },
    ]
}

fn kernel_build_id(ptx: &atlas_kernels::TargetPtxSet) -> String {
    let mut digest = Sha256::new();
    digest.update(ptx.target.arch.as_bytes());
    digest.update(ptx.target.model.as_bytes());
    digest.update(ptx.target.quant.as_bytes());
    for (name, module) in &ptx.modules {
        digest.update((name.len() as u64).to_le_bytes());
        digest.update(name.as_bytes());
        digest.update((module.len() as u64).to_le_bytes());
        digest.update(module);
    }
    format!("{:x}", digest.finalize())
}

fn model_checksum(model_dir: &Path, config: &ModelConfig) -> Result<String> {
    let mut digest = Sha256::new();
    digest.update(model_dir.as_os_str().as_encoded_bytes());
    digest.update(config.model_type.as_bytes());
    for value in [
        config.hidden_size,
        config.num_hidden_layers,
        config.vocab_size,
        config.num_experts,
        config.num_experts_per_tok,
    ] {
        digest.update((value as u64).to_le_bytes());
    }
    for name in [
        "config.json",
        "model.safetensors.index.json",
        "generation_config.json",
    ] {
        let path = model_dir.join(name);
        if path.is_file() {
            let bytes = std::fs::read(&path)
                .with_context(|| format!("read graph fingerprint input {}", path.display()))?;
            digest.update((name.len() as u64).to_le_bytes());
            digest.update(name.as_bytes());
            digest.update((bytes.len() as u64).to_le_bytes());
            digest.update(bytes);
        }
    }
    Ok(format!("{:x}", digest.finalize()))
}
