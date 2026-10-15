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
    /// This layer's index — used only by the `ATLAS_N35_L0_DUMP` /
    /// `ATLAS_N35_STAGE_DUMP_LAYER` diagnostic.
    layer_idx: usize,
    // Kernel handles — decode (single token)
    rms_norm_residual_k: KernelHandle,
    dense_gemv_k: KernelHandle,
    /// Bit-identical M-row sibling of `dense_gemv_k` (M<=8). K-row verify
    /// must use this for router logits: the tiled prefill GEMM changes BF16
    /// rounding and can select different experts than sequential decode.
    dense_gemv_batchm_k: KernelHandle,
    topk_sigmoid_k: KernelHandle,
    moe_expert_gemv_k: KernelHandle,
    moe_expert_gemv_pf_k: KernelHandle,
    moe_expert_gemv_wide_k: KernelHandle,
    moe_expert_gemv_wide_grouped_k: KernelHandle,
    /// Union-table variant of `moe_expert_gemv_wide_grouped` (omni overlay):
    /// the prepass builds unique-expert match tables once, so the GEMV
    /// launches only `u_count` experts instead of scanning all 512.
    /// `KernelHandle(0)` on miss → grouped path stays.
    moe_union_match_k: KernelHandle,
    moe_expert_gemv_wide_grouped_u_k: KernelHandle,
    /// Device scratch for the union table (u_count | u_experts | u_nmatch |
    /// u_tok | u_slot): (1 + 2*512 + 2*512*32) u32 = 265 KB. Allocated once
    /// at init when the overlay resolves; reused every verify.
    moe_union_scratch: Option<DevicePtr>,
    w4a16_gemv_k: KernelHandle,
    /// Single-warp `w4a16_gemv_sw`. `KernelHandle(0)` on miss → base GEMV.
    w4a16_gemv_sw_k: KernelHandle,
    w4a16_gemv_batch4_k: KernelHandle,
    w4a16_gemv_batch16_k: KernelHandle,
    w4a16_gemv_batch32_k: KernelHandle,
    /// Native-FP8 decode GEMV for the shared-expert up_proj (see
    /// `NemotronMoeWeights::shared_up_fp8`). 0 when unavailable.
    w8a16_gemv_k: KernelHandle,
    /// Bit-identical M<=16 twin of `w8a16_gemv` — the exact-verify shared
    /// expert's batched up/down projections. 0 when the module is absent.
    w8a16_gemv_batch16_k: KernelHandle,
    /// Native-FP8 prefill GEMM for the shared expert. 0 when unavailable.
    w8a16_gemm_k: KernelHandle,
    w8a16_gemm_pipelined_k: KernelHandle,
    relu2_down_shared_k: KernelHandle,
    /// Shared-slot-only twin of `relu2_down_shared` for the batched verify —
    /// separate entry point so the fused kernel's PTX is untouched.
    relu2_down_shared_slot_k: KernelHandle,
    /// Omni-only: split the fused relu²+down launch — routed slots at
    /// inter*4 smem, shared slot via `moe_relu2_down_shared_slot` at
    /// shared_inter*4. Other targets keep the fused launch.
    split_shared_down: bool,
    /// Omni-only: read back the sorted expert_offsets per prefill chunk and
    /// launch the grouped GEMMs at the exact per-expert tile count instead
    /// of the total_expanded/64 worst case (grid was ~60M CTAs of early-exit).
    exact_prefill_tiles: bool,
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
    moe_grouped_gemm_mloop_k: KernelHandle,
    /// KS=64 staged variant of the mloop grouped kernel (strix-hip n35vl);
    /// byte-identical outputs, deeper staging. Preferred when it resolves.
    moe_grouped_gemm_mloop2_k: KernelHandle,
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
        layer_idx: usize,
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

        // Omni-only lever: the fused relu²+down launch would reserve
        // shared_inter*4 smem for every slot; routed slots need only inter*4.
        // Splitting is byte-identical (same kernel, same slot math) but only
        // measured on omni, so other Nemotron targets keep the fused launch.
        let is_omni = config.is_nemotron_h_omni();
        let union_match_k = super::try_kernel(gpu, "moe_union_gemv", "moe_union_match");
        let union_gemv_k =
            super::try_kernel(gpu, "moe_union_gemv", "moe_expert_gemv_wide_grouped_u");
        // Union table scratch: (1 + 2*512 + 2*512*32) u32 = 265 KB, reused
        // for every verify's up and down grouped launches.
        // One thread per expert — the prepass caps at UG_MAXU=512.
        let moe_union_scratch =
            if union_match_k.0 != 0 && union_gemv_k.0 != 0 && weights.experts.len() <= 512 {
                Some(gpu.alloc(4 + 4 * (2 * 512 + 2 * 512 * 32))?)
            } else {
                None
            };
        let slot_k = super::try_kernel(
            gpu,
            "moe_relu2_down_shared_slot",
            "moe_relu2_down_shared_slot",
        );
        // Split when the shared slot's wider smem would reserve for routed
        // slots too: omni (as measured) or any latent MoE whose routed out N
        // is smaller than hidden (grid.x is also wrong for routed slots
        // without the split — they only write `moe_latent_size` columns).
        let split_shared_down = slot_k.0 != 0
            && (is_omni || config.moe_latent_size > 0)
            && config.shared_expert_intermediate_size > moe_inter;
        let exact_prefill_tiles = is_omni;
        Ok(Self {
            weights,
            input_norm,
            moe_latent_size: config.moe_latent_size,
            moe_inter,
            top_k,
            rms_norm_residual_k: gpu.kernel("norm", "rms_norm_residual")?,
            dense_gemv_k: gpu.kernel("gemv", "dense_gemv_bf16")?,
            dense_gemv_batchm_k: super::try_kernel(
                gpu,
                "dense_gemv_bf16_batchm",
                "dense_gemv_bf16_batchm",
            ),
            topk_sigmoid_k: gpu.kernel("moe_topk_sig", "moe_topk_sigmoid")?,
            moe_expert_gemv_k: gpu.kernel("moe_expert_gemv", "moe_expert_gemv")?,
            moe_expert_gemv_pf_k: super::try_kernel(
                gpu,
                "moe_expert_gemv_pf",
                "moe_expert_gemv_pf",
            ),
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
            moe_union_match_k: union_match_k,
            moe_expert_gemv_wide_grouped_u_k: union_gemv_k,
            moe_union_scratch,
            w4a16_gemv_k: gpu.kernel("w4a16_gemv", "w4a16_gemv")?,
            w4a16_gemv_sw_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_sw"),
            w4a16_gemv_batch4_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch4"),
            w4a16_gemv_batch16_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch16"),
            w4a16_gemv_batch32_k: super::try_kernel(gpu, "w4a16_gemv", "w4a16_gemv_batch32"),
            w8a16_gemv_k: super::try_kernel(gpu, "w8a16_gemv", "w8a16_gemv"),
            w8a16_gemv_batch16_k: super::try_kernel(gpu, "w8a16_gemv_batch4", "w8a16_gemv_batch16"),
            w8a16_gemm_k: super::try_kernel(gpu, "w8a16_gemm", "w8a16_gemm"),
            w8a16_gemm_pipelined_k: super::try_kernel(
                gpu,
                "w8a16_gemm_pipelined",
                "w8a16_gemm_pipelined",
            ),
            relu2_down_shared_k: gpu.kernel("moe_relu2_fused", "moe_expert_relu2_down_shared")?,
            relu2_down_shared_slot_k: slot_k,
            split_shared_down,
            exact_prefill_tiles,
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
            moe_grouped_gemm_mloop_k: super::try_kernel_gated(
                cfg!(atlas_hip),
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_mloop",
            ),
            moe_grouped_gemm_mloop2_k: super::try_kernel_gated(
                cfg!(atlas_hip),
                gpu,
                "moe_w4a16",
                "moe_w4a16_grouped_gemm_ptrtable_mloop2",
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
            layer_idx,
            marlin,
        })
    }
}

mod decode_batched;
mod decode_helpers;
mod decode_verify_exact;
mod marlin_linear;
mod marlin_sidecar;
mod marlin_slots;
#[cfg(all(test, target_os = "linux"))]
mod prefetch_verify_gpu_tests;
mod prefill_fallback;
mod prefill_marlin;
mod prefill_shared_up;
mod prefill_sorted;
mod prefill_weights;
mod ptr_tables;
mod transformer_layer;
mod union_verify_gpu_tests;

use ptr_tables::{build_ptr_table, build_ptr_table_from_weights};

/// Launch plan for the fused relu²+down split (omni lever):
/// returns (fused grid.y slots, fused dynamic smem bytes, run shared slot
/// separately via `moe_relu2_down_shared_slot`). When the shared slot stays
/// fused it needs `shared_inter.max(inter)*4`; routed-only launches need
/// `inter*4`. The shared slot kernel then launches with `shared_inter*4`.
pub(crate) fn down_launch_plan(
    top_k: u32,
    inter: u32,
    shared_inter: u32,
    split: bool,
    native_shared: bool,
) -> (u32, u32, bool) {
    let shared_separate = split && !native_shared;
    let fused_slots = if native_shared || shared_separate {
        top_k
    } else {
        top_k + 1
    };
    let fused_smem = if fused_slots == top_k + 1 {
        shared_inter.max(inter) * 4
    } else {
        inter * 4
    };
    (fused_slots, fused_smem, shared_separate)
}

#[cfg(test)]
mod down_launch_plan_tests {
    use super::down_launch_plan;

    #[test]
    fn split_plans_smem() {
        // fused fallback (no split, no native): top_k+1 slots, full smem
        assert_eq!(
            down_launch_plan(22, 2688, 5376, false, false),
            (23, 5376 * 4, false)
        );
        // omni split: routed slots at inter*4, shared runs separately
        assert_eq!(
            down_launch_plan(22, 2688, 5376, true, false),
            (22, 2688 * 4, true)
        );
        // native-FP8 shared down: routed slots only, inter*4, no slot launch
        assert_eq!(
            down_launch_plan(22, 2688, 5376, true, true),
            (22, 2688 * 4, false)
        );
        // latent MoE on a non-omni build: routed slots are N=latent (1024),
        // not N=hidden — the fused grid must be `div_ceil(latent,8)` and the
        // shared slot splits out at smem = shared_inter*4.
        let (fused_slots, fused_smem, split_shared) = down_launch_plan(22, 2688, 5376, true, false);
        assert_eq!(
            (fused_slots, fused_smem, split_shared),
            (22, 2688 * 4, true)
        );
        // grid.x the caller derives: shared-in-fused -> max_n, else latent.
        let routed_n = 1024u32;
        let hidden_n = 4096u32;
        let fused_n = if fused_slots == 22 + 1 {
            hidden_n
        } else {
            routed_n
        };
        assert_eq!(fused_n.div_ceil(8), 128);
        assert_eq!(
            down_launch_plan(22, 2688, 5376, false, true),
            (22, 2688 * 4, false)
        );
        // shared_inter <= inter (hypothetical): fused smem is still the max
        assert_eq!(
            down_launch_plan(22, 5376, 2688, false, false),
            (23, 5376 * 4, false)
        );
    }
}
