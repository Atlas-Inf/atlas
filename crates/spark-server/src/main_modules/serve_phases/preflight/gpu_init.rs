// SPDX-License-Identifier: AGPL-3.0-only

//! GPU backend init + post-load OOM check.

use anyhow::{Context, Result};

use atlas_core::config::ModelConfig;

use crate::cli;

/// Initialize the GPU backend for the active feature.
///
/// Compile-time dispatch:
/// - `cuda` feature → `AtlasCudaBackend` loading PTX modules from `ptx_set`.
/// - `metal` feature → `MetalGpuBackend` loading metallib modules from
///   `ptx_set` as well. Both arms register the RESOLVED target's modules;
///   `metallib_modules()` is a plain alias of target 0, so registering from
///   it served another model's kernels in a multi-target build.
#[cfg(feature = "cuda")]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    spark_runtime::cuda_backend::configure_pdl(ptx_set.target.model);
    let backend =
        spark_runtime::cuda_backend::AtlasCudaBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize CUDA backend")?;

    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(backend);
    // HIP targets link a cuBLASLt stub (every call fails), which would strand
    // the ~20 BF16 projection sites that route through
    // `cublaslt::bf16_gemm_act_weight_t` with no fallback. Install one
    // process-global Atlas-GEMM fallback now, while the kernel registry is
    // live: it only ever fires when `ctx()` fails, so cuBLASLt-capable
    // backends never consult it. `pipelined` may be unresolved on targets
    // that lack the tensor-core variant — the fallback then uses the scalar
    // kernel for every shape.
    let pipelined = gpu
        .kernel("gemm", "dense_gemm_bf16_pipelined")
        .unwrap_or(spark_runtime::gpu::KernelHandle(0));
    let scalar = gpu
        .kernel("gemm", "dense_gemm_bf16")
        .unwrap_or(spark_runtime::gpu::KernelHandle(0));
    // Optional small-M arm: `dense_gemv_bf16_batchm` serves 1 <= M <= 8
    // decode projections (e.g. Flash-Next's BF16 `in_proj_qkvz`) in one
    // bandwidth-bound pass instead of a 128-row WMMA tile. Handle 0 -> the
    // fallback skips the arm.
    let gemv_batchm = gpu
        .kernel("dense_gemv_bf16_batchm", "dense_gemv_bf16_batchm")
        .unwrap_or(spark_runtime::gpu::KernelHandle(0));
    if spark_runtime::cublaslt::install_bf16_fallback(spark_runtime::cublaslt::make_bf16_fallback(
        pipelined,
        scalar,
        gemv_batchm,
    )) {
        tracing::info!(
            "BF16 GEMM fallback installed (gemv<=8 / pipelined WMMA / scalar) for backends without cuBLASLt"
        );
    }
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    // Baseline for self-relative KV budgeting: free memory now (post context +
    // PTX modules, pre weights) minus free-at-build = this process's own
    // footprint, co-tenants excluded. See gpu::baseline_free_bytes.
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "GPU {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub(crate) fn init_gpu_backend(
    args: &cli::ServeArgs,
    ptx_set: &atlas_kernels::TargetPtxSet,
) -> Result<(Box<dyn spark_runtime::gpu::GpuBackend>, usize)> {
    // The RESOLVED target's modules, exactly like the CUDA arm above.
    // `metallib_modules()` is an alias of `ptx_modules()`, which build-codegen
    // emits as a plain alias of TARGET 0 in a multi-target build — so this
    // registered another model's kernels and every lookup for the model
    // actually being served failed.
    let gpu: Box<dyn spark_runtime::gpu::GpuBackend> = Box::new(
        spark_runtime::metal_backend::MetalGpuBackend::new(args.gpu_ordinal, &ptx_set.modules)
            .context("Failed to initialize Metal backend")?,
    );
    let total_mem = gpu.total_memory()?;
    let free_mem = gpu.free_memory()?;
    spark_runtime::gpu::set_baseline_free_bytes(free_mem);
    tracing::info!(
        "Metal device {}: {:.1} GB total, {:.1} GB free",
        args.gpu_ordinal,
        total_mem as f64 / (1024.0 * 1024.0 * 1024.0),
        free_mem as f64 / (1024.0 * 1024.0 * 1024.0),
    );
    Ok((gpu, free_mem))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn post_load_memory_audit(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    weight_bytes: usize,
    free_mem: usize,
    inference_reserve: usize,
    total_reserve: usize,
    gdn_two_phase_bytes: usize,
    max_batch_tokens_pre: usize,
) -> Result<()> {
    let estimated_free = free_mem.saturating_sub(weight_bytes);
    let actual_free = gpu.free_memory().unwrap_or(estimated_free);
    let available_free = if actual_free > 0 {
        actual_free
    } else {
        estimated_free
    };
    if available_free < total_reserve {
        let avail_gb = available_free as f64 / (1024.0 * 1024.0 * 1024.0);
        let need_gb = total_reserve as f64 / (1024.0 * 1024.0 * 1024.0);
        let hint = if args.max_batch_size > 1 {
            format!(
                " Reduce --max-batch-size (currently {}) or --max-seq-len (currently {}).",
                args.max_batch_size, args.max_seq_len
            )
        } else {
            format!(
                " Reduce --max-seq-len (currently {}) or use a smaller model.",
                args.max_seq_len
            )
        };
        anyhow::bail!(
            "Insufficient GPU memory for inference buffers. \
             After loading {:.2} GB of weights, only {:.2} GB remains \
             but {:.2} GB is needed for SSM state pool ({} slots × {} layers) + scratch buffers.{}",
            weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            avail_gb,
            need_gb,
            args.max_batch_size,
            config.num_ssm_layers(),
            hint,
        );
    }
    if gdn_two_phase_bytes > 0 {
        tracing::info!(
            "GDN chunked prefill reserve: {} MB (chunk_size={}, max_seq_len={})",
            gdn_two_phase_bytes / (1024 * 1024),
            max_batch_tokens_pre,
            args.max_seq_len,
        );
    }
    tracing::info!(
        "Weights: {:.2} GB, estimated free: {:.1} GB, actual free: {:.1} GB (reserve: {} MB)",
        weight_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        estimated_free as f64 / (1024.0 * 1024.0 * 1024.0),
        actual_free as f64 / (1024.0 * 1024.0 * 1024.0),
        inference_reserve / (1024 * 1024),
    );
    Ok(())
}
