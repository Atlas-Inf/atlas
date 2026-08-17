// SPDX-License-Identifier: AGPL-3.0-only

//! Nemotron-H standalone MoE FFN layer.
//!
//! Supports two variants:
//!   - **Nano 30B**: Direct MoE — experts operate on full hidden_size.
//!   - **Super 120B**: LatentMoE — routed experts operate in latent space `[moe_latent_size]`,
//!     with fc1/fc2 latent projections bridging hidden↔latent.
//!
//! Forward: RMS norm → gate → sigmoid topK routing → (fc1_latent if latent) →
//!          batched up GEMV → fused relu²+down → weighted_sum → (fc2_latent if latent) →
//!          shared expert up+relu²+down → sum routed+shared → residual add.
//!
//! All expert dispatch is device-side (pointer tables) — zero D2H sync.

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layers::{nemotron_decode_policy, ops};
use crate::weight_map::{DenseWeight, NemotronMoeWeights, QuantizedWeight};

/// Device-side pointer table for one projection across all experts.
struct ExpertPtrTable {
    packed_ptrs: DevicePtr,
    scale_ptrs: DevicePtr,
    scale2_vals: DevicePtr,
}

/// Nemotron-H standalone MoE FFN layer.
pub struct NemotronMoeLayer {
    weights: NemotronMoeWeights,
    input_norm: DenseWeight,
    /// LatentMoE dimension (0 = direct, >0 = latent).
    moe_latent_size: usize,
    /// Routed expert intermediate size for this layer (Puzzle: per-block).
    moe_inter: usize,
    /// Top-K experts activated per token for this layer (Puzzle: per-block).
    top_k: usize,
    // Kernel handles — decode (single token)
    rms_norm_residual_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    topk_sigmoid_k: KernelHandle,
    moe_expert_gemv_k: KernelHandle,
    moe_expert_gemv_wide_k: KernelHandle,
    moe_expert_gemv_wide_grouped_k: KernelHandle,
    w4a16_gemv_k: KernelHandle,
    w4a16_gemv_batch4_k: KernelHandle,
    w4a16_gemv_batch16_k: KernelHandle,
    /// Native-FP8 decode GEMV for the shared-expert up_proj (see
    /// `NemotronMoeWeights::shared_up_fp8`). 0 when unavailable.
    w8a16_gemv_k: KernelHandle,
    /// Native-FP8 prefill GEMM for the shared expert. 0 when unavailable.
    w8a16_gemm_k: KernelHandle,
    w8a16_gemm_pipelined_k: KernelHandle,
    relu2_down_shared_k: KernelHandle,
    relu2_down_wide_k: KernelHandle,
    weighted_sum_scale_k: KernelHandle,
    residual_add_k: KernelHandle,
    // Kernel handles — prefill (batched GEMM)
    dense_gemm_k: KernelHandle,
    /// Pipelined tensor-core BF16 GEMM (mma.sync.m16n8k16 + cp.async 2-stage,
    /// 128x128 tile). `dense_gemm_bf16` is a SCALAR 16x16 kernel — on the
    /// large-M prefill shapes it is ~40x slower, and the three dense GEMMs of a
    /// LatentMoE layer (gate, fc1_latent, fc2_latent) were the single largest
    /// prefill cost on Puzzle (34% of all GPU time). Same math (cosine=1.0).
    dense_gemm_pipelined_k: KernelHandle,
    w4a16_gemm_k: KernelHandle,
    // Batched N-token MoE prefill kernels
    topk_sigmoid_batched_k: KernelHandle,
    moe_up_prefill_k: KernelHandle,
    moe_relu2_down_prefill_k: KernelHandle,
    moe_weighted_sum_prefill_k: KernelHandle,
    // Sorted grouped GEMM (Qwen pattern — proven to work)
    moe_sort_k: KernelHandle,
    moe_grouped_gemm_k: KernelHandle,
    moe_relu2_elementwise_k: KernelHandle,
    moe_grouped_gemm_relu2_k: KernelHandle,
    moe_w4a4_grouped_k: KernelHandle,
    moe_unpermute_reduce_k: KernelHandle,
    moe_grouped_gemm_n128_k: KernelHandle,
    up_ptrs: ExpertPtrTable,
    down_ptrs: ExpertPtrTable,
    // Transposed expert pointer tables (for N128 grouped GEMM)
    up_ptrs_t: Option<ExpertPtrTable>,
    down_ptrs_t: Option<ExpertPtrTable>,
    // Transposed shared expert weights
    shared_up_t: Option<QuantizedWeight>,
    shared_down_t: Option<QuantizedWeight>,
    // Pre-dequantized FP8 E4M3 [N, K] copies of the shared-expert projections.
    // Consumed by `fp8_gemm_t_m128_mfast` (no dequant phase); see the SSM layer.
    shared_up_pd_fp8: Option<DevicePtr>,
    shared_down_pd_fp8: Option<DevicePtr>,
    // FP8 E4M3 copies of the BF16 latent projections, so prefill runs the tuned
    // FP8 GEMM instead of dense_gemm_bf16_pipelined (and halves their bytes).
    fc1_pd_fp8: Option<DevicePtr>,
    fc2_pd_fp8: Option<DevicePtr>,
    // Transposed SSM GEMM kernel handle (for shared expert)
    w4a16_gemm_t_k: KernelHandle,
    w4a16_gemm_t_m128_k: KernelHandle,
    fp8_gemm_m128_k: KernelHandle,
    w4a4_gemm_k: KernelHandle,
    quantize_nvfp4_k: KernelHandle,
    marlin: Option<marlin_sidecar::MarlinSidecar>,
}

impl NemotronMoeLayer {
    pub fn new(
        weights: NemotronMoeWeights,
        input_norm: DenseWeight,
        config: &ModelConfig,
        gpu: &dyn GpuBackend,
        moe_inter: usize,
        top_k: usize,
    ) -> Result<Self> {
        let up_ptrs = build_ptr_table(&weights.experts, |e| &e.up_proj, gpu)?;
        let down_ptrs = build_ptr_table(&weights.experts, |e| &e.down_proj, gpu)?;
        let moe_inter = if moe_inter > 0 {
            moe_inter
        } else {
            config.moe_intermediate_size
        };
        let top_k = if top_k > 0 {
            top_k
        } else {
            config.num_experts_per_tok
        };
        // Nemotron's `top_k` is PER LAYER (`num_experts_per_tok_for`), so this
        // runs once per MoE layer and a single outlying block config cannot
        // slip through on the model-wide value. `MoeLayer::new` carries the
        // same pair of bounds; `NemotronMoeLayer` had no check at all and its
        // decode kernel is the one whose shadows were capped at 24.
        let num_experts = weights.experts.len();
        anyhow::ensure!(
            top_k > 0
                && top_k <= num_experts
                && top_k <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K
                && num_experts <= crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
            "Nemotron MoE config invalid: top_k={} must be in 1..={} and within \
             the routing kernels' bounds (top_k max {}, num_experts={} max {})",
            top_k,
            num_experts,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_TOP_K,
            num_experts,
            crate::layers::ops::MOE_TOPK_SIGMOID_MAX_EXPERTS,
        );

        let marlin = marlin_sidecar::MarlinSidecar::try_build(
            gpu,
            &weights.experts,
            moe_inter,
            config.hidden_size,
            config.hidden_size,
            moe_inter,
        )?;

        Ok(Self {
            weights,
            input_norm,
            moe_latent_size: config.moe_latent_size,
            moe_inter,
            top_k,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            topk_sigmoid_k: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid")?,
            moe_expert_gemv_k: gpu.kernel("moe_expert_gemv", "moe_expert_gemv")?,
            moe_expert_gemv_wide_k: super::try_kernel(
                gpu,
                "moe_expert_gemv_wide",
                "moe_expert_gemv_wide",
            ),
            moe_expert_gemv_wide_grouped_k: super::try_kernel(
                gpu,
                "moe_expert_gemv_wide_grouped",
                "moe_expert_gemv_wide_grouped",
            ),
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_batch4_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch4"),
            w4a16_gemv_batch16_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16"),
            w8a16_gemv_k: super::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            w8a16_gemm_k: super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            relu2_down_shared_k: gpu.kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")?,
            relu2_down_wide_k: super::try_kernel(
                gpu,
                "moe_expert_relu2_down_wide",
                "moe_expert_relu2_down_wide",
            ),
            weighted_sum_scale_k: gpu.kernel("relu2", "moe_weighted_sum_scale")?,
            residual_add_k: gpu.kernel("residual_add", "bf16_residual_add")?,
            dense_gemm_k: gpu.kernel("gemm", "dense_gemm_bf16")?,
            dense_gemm_pipelined_k: super::try_kernel(gpu, "gemm", "dense_gemm_bf16_pipelined"),
            w4a16_gemm_k: gpu.kernel("w4a16", "w4a16_gemm")?,
            topk_sigmoid_batched_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_topk_sigmoid_batched",
            ),
            moe_up_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_up_prefill",
            ),
            moe_relu2_down_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_relu2_down_prefill",
            ),
            moe_weighted_sum_prefill_k: super::try_kernel(
                gpu,
                "nemotron_moe_prefill",
                "nemotron_moe_weighted_sum_prefill",
            ),
            moe_sort_k: super::try_kernel(gpu, "moe", "moe_sort_by_expert"),
            moe_grouped_gemm_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable",
            ),
            moe_relu2_elementwise_k: super::try_kernel(gpu, "relu2", "relu_squared_inplace"),
            moe_grouped_gemm_relu2_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_relu2",
            ),
            moe_w4a4_grouped_k: super::try_kernel(gpu, "moe_w4a4", "moe_w4a4_grouped_gemm_relu2"),
            moe_unpermute_reduce_k: super::try_kernel(gpu, "moe", "moe_unpermute_reduce_indexed"),
            moe_grouped_gemm_n128_k: super::try_kernel(
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_t",
            ),
            up_ptrs,
            down_ptrs,
            up_ptrs_t: None,
            down_ptrs_t: None,
            shared_up_t: None,
            shared_down_t: None,
            shared_up_pd_fp8: None,
            shared_down_pd_fp8: None,
            fc1_pd_fp8: None,
            fc2_pd_fp8: None,
            w4a16_gemm_t_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t"),
            w4a16_gemm_t_m128_k: super::try_kernel(gpu, "w4a16", "w4a16_gemm_t_m128"),
            fp8_gemm_m128_k: super::try_kernel(gpu, "w4a16", "fp8_gemm_t_m128_mfast"),
            w4a4_gemm_k: super::try_kernel(gpu, "w4a4", "w4a4_gemm_mfast"),
            quantize_nvfp4_k: super::try_kernel(gpu, "quantize_nvfp4", "quantize_bf16_to_nvfp4"),
            marlin,
        })
    }
}

mod decode_batched;
mod decode_helpers;
mod marlin_linear;
mod marlin_sidecar;
mod marlin_slots;
mod prefill_fallback;
mod prefill_marlin;
mod prefill_shared_up;
mod prefill_sorted;
mod prefill_weights;
mod ptr_tables;
mod transformer_layer;

use ptr_tables::{build_ptr_table, build_ptr_table_from_weights};
