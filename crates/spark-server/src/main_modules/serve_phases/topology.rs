// SPDX-License-Identifier: AGPL-3.0-only

//! TP/EP topology resolution + NCCL communicator init.

// `Context` only used by the nccl-feature `init_nccl_comm` to wrap
// NCCL bootstrap errors; cuda-without-nccl and metal builds don't
// reach that path.
#[cfg(feature = "nccl")]
use anyhow::Context;
use anyhow::Result;

use atlas_core::config::ModelConfig;

use crate::cli;

pub(crate) struct Topology {
    pub(crate) world_size: usize,
    pub(crate) tp_size: usize,
    pub(crate) ep_size: usize,
    pub(crate) tp_rank: usize,
    pub(crate) ep_rank: usize,
}

pub(crate) fn resolve_topology(
    args: &cli::ServeArgs,
    config: &mut ModelConfig,
) -> Result<Topology> {
    let (tp_size, ep_size) = if args.tp_size == 1 && args.ep_size == 1 && args.world_size > 1 {
        (1usize, args.world_size)
    } else {
        (args.tp_size.max(1), args.ep_size.max(1))
    };
    let derived_world = if tp_size == ep_size {
        tp_size
    } else {
        tp_size * ep_size
    };
    let world_size = if args.world_size <= 1 && (tp_size > 1 || ep_size > 1) {
        tracing::info!(
            "Auto-derived world_size={} from --tp-size {} --ep-size {} (rule: \
             tp==ep → overlapping = tp; else orthogonal = tp×ep). Pass \
             --world-size to override.",
            derived_world,
            tp_size,
            ep_size,
        );
        derived_world
    } else {
        args.world_size
    };
    let (tp_rank, ep_rank) = if tp_size == ep_size && tp_size == world_size && tp_size > 1 {
        (args.rank, args.rank)
    } else if world_size == tp_size * ep_size {
        (args.rank % tp_size, args.rank / tp_size)
    } else {
        anyhow::bail!(
            "Invalid parallelism topology: world_size={} but tp_size={} × ep_size={} = {}. \
             Either use orthogonal mesh (world = tp × ep) or overlapping groups \
             (world = tp = ep, used for 2-GPU TP+EP composition).",
            world_size,
            tp_size,
            ep_size,
            tp_size * ep_size,
        );
    };
    config.tp_rank = tp_rank;
    config.tp_world_size = tp_size;
    config.ep_rank = ep_rank;
    config.ep_world_size = ep_size;
    // Every rank must resolve the same value (`rank_settings`): with it each
    // rank holds a slice of every routed expert, without it half of them.
    config.expert_tp = std::env::var("ATLAS_GLM_EXPERT_TP").as_deref() == Ok("1")
        && config.model_type == "glm5_next"
        && ep_size == 2
        && tp_size == ep_size
        && world_size == ep_size;
    if tp_size > 1 {
        // EXL3 packs each linear into 16x16 trellis tiles laid out
        // [in/16, out/16, ...]; a tile is atomic, so the tensor cannot be
        // row-sliced per rank. Refuse before any loader/TP-capability check.
        if config
            .quantization_config
            .as_ref()
            .is_some_and(|qc| qc.is_exl3())
        {
            anyhow::bail!(
                "EXL3 checkpoints are TP=1 only (packed 16x16 trellis tiles cannot be \
                 row-sliced); got --tp-size {tp_size}"
            );
        }
        let loader = spark_model::factory::loader_for_config(config)?;
        if !loader.supports_tp() {
            anyhow::bail!(
                "TP (--tp-size > 1) is not supported by the {} weight loader. \
                 Run with --tp-size 1 (EP-only). To extend TP to this architecture, \
                 wire `crate::tp_shard::slice_for_rank` per attention/MoE/SSM \
                 tensor in the loader and override `ModelWeightLoader::supports_tp()` \
                 to return true. See `weight_loader/minimax.rs` as the reference.",
                config.model_type,
            );
        }
        drop(loader);
        if !config.num_attention_heads.is_multiple_of(tp_size) {
            anyhow::bail!(
                "TP requires num_attention_heads ({}) divisible by tp_size ({})",
                config.num_attention_heads,
                tp_size,
            );
        }
        if !config.num_key_value_heads.is_multiple_of(tp_size) {
            anyhow::bail!(
                "TP requires num_key_value_heads ({}) divisible by tp_size ({})",
                config.num_key_value_heads,
                tp_size,
            );
        }
        config.num_attention_heads /= tp_size;
        config.num_key_value_heads /= tp_size;
        // GDN HeadParallel: linear-attention (SSM) key/value head counts are
        // sharded exactly like attention heads — each rank owns a contiguous
        // head range; the recurrence is head-parallel with one all-reduce after
        // out_proj. Only relevant for SSM-hybrid models (linear_*_heads > 0);
        // pure-attention configs leave these at 0 and skip the divide.
        if config.linear_num_key_heads > 0 || config.linear_num_value_heads > 0 {
            if !config.linear_num_key_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_key_heads ({}) divisible by tp_size ({})",
                    config.linear_num_key_heads,
                    tp_size,
                );
            }
            if !config.linear_num_value_heads.is_multiple_of(tp_size) {
                anyhow::bail!(
                    "TP requires linear_num_value_heads ({}) divisible by tp_size ({})",
                    config.linear_num_value_heads,
                    tp_size,
                );
            }
            config.linear_num_key_heads /= tp_size;
            config.linear_num_value_heads /= tp_size;
        }
        tracing::info!(
            "TP-local head counts: num_attention_heads={}, num_key_value_heads={}, \
             linear_num_key_heads={}, linear_num_value_heads={}",
            config.num_attention_heads,
            config.num_key_value_heads,
            config.linear_num_key_heads,
            config.linear_num_value_heads,
        );
    }
    if world_size > 1 {
        let (start, end) = config.local_expert_range();
        tracing::info!(
            "Parallelism: global rank {}/{} (tp_rank={}/{}, ep_rank={}/{}), local experts [{}, {}), \
             expert TP {}",
            args.rank,
            world_size,
            tp_rank,
            tp_size,
            ep_rank,
            ep_size,
            start,
            end,
            config.expert_tp,
        );
    }
    Ok(Topology {
        world_size,
        tp_size,
        ep_size,
        tp_rank,
        ep_rank,
    })
}

/// The settings this crate resolves that every rank must share, for
/// `spark_model::model::startup_parity`: a mismatch changes the rows of a
/// prefill pass, the exchanges it may take, what a rank computes, the state
/// it sizes and keeps, or whether it captures a step as a CUDA graph.
#[cfg(feature = "nccl")]
fn rank_settings(
    args: &cli::ServeArgs,
    expert_tp: bool,
    prefill_budget: usize,
    max_batch_tokens: usize,
) -> [spark_model::model::startup_parity::Setting; 12] {
    [
        ("ATLAS_GLM_EXPERT_TP", expert_tp as u64),
        // The arena and pair capacity, and what it is resolved from: equal
        // capacities can hide another chunk or another batch.
        ("max batch tokens", max_batch_tokens as u64),
        (
            "prefill chunk (--max-prefill-tokens)",
            prefill_budget as u64,
        ),
        ("--max-batch-size", args.max_batch_size as u64),
        ("--max-seq-len", args.max_seq_len as u64),
        ("--block-size", args.block_size as u64),
        ("--enable-prefix-caching", args.enable_prefix_caching as u64),
        // As the pool is built, not as given.
        (
            "--ssm-cache-slots",
            super::build::resolve_ssm_cache_slots(args) as u64,
        ),
        (
            "--ssm-checkpoint-interval",
            args.ssm_checkpoint_interval as u64,
        ),
        // The speculative state a rank builds for the head's verify commands.
        (
            "--speculative or --dflash",
            (args.speculative || args.dflash) as u64,
        ),
        // Both keep a step out of a CUDA graph.
        ("--high-speed-swap", args.high_speed_swap as u64),
        ("--profile", args.profile as u64),
    ]
}

/// `max_batch_tokens` and `hidden_size` size the 2-rank all-reduce receive
/// buffer. Together they bound the largest payload any caller can hand a
/// collective: prefill MoE, prefill attention and prefill SSM all reduce a
/// `[num_tokens, hidden_size]` BF16 tensor, and `num_tokens` is capped by
/// `max_batch_tokens` (the same bound the `moe_output` arena buffer is sized
/// on). Deriving the capacity here is what keeps a wider model or a larger
/// `--max-prefill-tokens` from overrunning a fixed allocation.
#[cfg(feature = "nccl")]
pub(crate) fn init_nccl_comm(
    args: &cli::ServeArgs,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
    prefill_budget: usize,
    max_batch_tokens: usize,
    config: &ModelConfig,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    use spark_comm::CommBackend;
    let hidden_size = config.hidden_size;
    if world_size <= 1 {
        return Ok(None);
    }
    let recv_capacity = spark_comm::nccl_backend::required_recv_bytes(
        max_batch_tokens,
        hidden_size,
        spark_comm::nccl_backend::ALL_REDUCE_DTYPE_BYTES,
    )
    .context("Failed to size the NCCL receive buffer")?;
    tracing::info!(
        "Initializing NCCL: rank {}/{}, master {}:{}, recv_buffer {} MiB \
         (max_batch_tokens={} × hidden_size={} × {} B)",
        args.rank,
        world_size,
        args.master_addr,
        args.master_port,
        recv_capacity / (1024 * 1024),
        max_batch_tokens,
        hidden_size,
        spark_comm::nccl_backend::ALL_REDUCE_DTYPE_BYTES,
    );
    let cuda_stream = gpu.default_stream();
    let backend = spark_comm::NcclBackend::new(
        args.rank,
        world_size,
        &args.master_addr,
        args.master_port,
        cuda_stream,
        recv_capacity,
    )
    .context("Failed to initialize NCCL")?;
    tracing::info!("NCCL initialized: rank {}", backend.rank());
    crate::ep_peer_lifeline::watch(&backend).context("Failed to arm the EP peer lifeline")?;
    // Before any other collective: a rank running another value of one of
    // these would deadlock or mispair the ranks at a prompt-dependent step.
    let settings = rank_settings(args, config.expert_tp, prefill_budget, max_batch_tokens);
    spark_model::model::startup_parity::agree(&backend, gpu, &settings)
        .context("Startup settings agreement")?;
    Ok(Some(
        std::sync::Arc::new(backend) as std::sync::Arc<dyn spark_comm::CommBackend>
    ))
}

/// CUDA-without-NCCL variant (SCALE/AMD gfx1151): the CUDA compute
/// backend is active but no NCCL library is linked, so multi-GPU
/// collectives are unavailable. `world_size > 1` is rejected explicitly
/// so a misconfigured `--rank > 0` invocation fails fast instead of
/// silently degrading to single-rank.
#[cfg(all(feature = "cuda", not(feature = "nccl")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
    _prefill_budget: usize,
    _max_batch_tokens: usize,
    _config: &ModelConfig,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available in this build (cuda feature \
             without nccl — SCALE/AMD gfx1151 has no NCCL library); \
             single-device only"
        );
    }
    Ok(None)
}

/// Metal-feature variant: NCCL multi-GPU isn't reachable on a single
/// Apple Silicon device, so collective ops fall back to the no-op
/// `SingleGpuBackend`. `world_size > 1` is rejected explicitly so a
/// misconfigured `--rank > 0` invocation fails fast instead of
/// silently degrading to single-rank.
#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_nccl_comm(
    _args: &cli::ServeArgs,
    _gpu: &dyn spark_runtime::gpu::GpuBackend,
    world_size: usize,
    _prefill_budget: usize,
    _max_batch_tokens: usize,
    _config: &ModelConfig,
) -> Result<Option<std::sync::Arc<dyn spark_comm::CommBackend>>> {
    if world_size > 1 {
        anyhow::bail!(
            "multi-rank NCCL is not available on Apple Silicon (metal feature); \
             single-device only"
        );
    }
    Ok(None)
}

#[cfg(all(test, feature = "nccl"))]
mod tests {
    use clap::Parser as _;

    /// The names of the settings two launches would disagree on.
    fn differing(a: (&[&str], usize, usize), b: (&[&str], usize, usize)) -> Vec<&'static str> {
        let settings = |(argv, prefill_budget, max_batch_tokens): (&[&str], usize, usize)| {
            let argv = ["spark", "some/model"].iter().chain(argv);
            let args = crate::cli::ServeArgs::parse_from(argv);
            super::rank_settings(&args, false, prefill_budget, max_batch_tokens)
        };
        let (a, b) = (settings(a), settings(b));
        a.iter()
            .zip(b)
            .filter(|(a, b)| a != &b)
            .map(|(a, _)| a.0)
            .collect()
    }

    #[test]
    fn the_snapshot_slots_are_compared_as_the_pool_is_built() {
        let pool = ["--ssm-checkpoint-interval=256", "--block-size=16"];
        let launch = |more: [&'static str; 2]| [&pool[..], &more[..]].concat();
        // One requested count, two contexts: pools of 136 and 72 slots.
        let long = launch(["--ssm-cache-slots=16", "--max-seq-len=524288"]);
        let short = launch(["--ssm-cache-slots=16", "--max-seq-len=262144"]);
        assert_eq!(
            differing((&long, 8192, 8196), (&short, 8192, 8196)),
            ["--max-seq-len", "--ssm-cache-slots"]
        );
        // Two requested counts that are raised to one pool.
        let fewer = launch(["--ssm-cache-slots=8", "--max-seq-len=524288"]);
        assert!(differing((&long, 8192, 8196), (&fewer, 8192, 8196)).is_empty());
    }

    #[test]
    fn one_capacity_does_not_hide_another_chunk_or_batch() {
        // `ATLAS_MAX_BATCH_TOKENS` gives both ranks one arena.
        assert_eq!(
            differing(
                (&["--max-batch-size=4"], 8192, 16384),
                (&["--max-batch-size=2"], 4096, 16384)
            ),
            ["prefill chunk (--max-prefill-tokens)", "--max-batch-size"]
        );
    }

    #[test]
    fn a_worker_without_the_heads_speculative_lane_is_named() {
        assert_eq!(
            differing((&["--speculative"], 8192, 8196), (&[], 8192, 8196)),
            ["--speculative or --dflash"]
        );
    }
}
