// SPDX-License-Identifier: AGPL-3.0-only

//! MoE (Mixture of Experts) FFN component.
//!
//! Batched expert dispatch: top-K experts run in 2 fused kernel launches
//! (gate+up, silu+down) instead of 10 × 5 individual launches. Expert indices
//! and weights stay on device — zero D2H synchronization.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};

use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::layers::w4a16_gemv_tiers::W4a16BatchmTiers;
use crate::weight_map::{DenseWeight, Fp8ExpertWeight, MoeWeights, Mxfp8Proj, QuantizedWeight};
mod btile_model_owners;
pub(crate) use btile_model_owners::{
    bind as bind_resident_btile_arenas, invalidate as invalidate_resident_btile_readers,
};
#[cfg(test)]
pub(crate) use gate_up_repack::model_fixture as btile_model_fixture;

/// Device-side pointer table for one projection across all experts.
///
/// Enables GPU-side expert dispatch: the batched GEMV kernel reads
/// expert_id from device memory, then indexes these tables to find
/// the correct weight pointers — no CPU involvement needed.
pub(crate) struct ExpertPtrTable {
    allocation: Option<ptr_table_build::receipt::TableAllocation>,
    /// `[num_experts]` u64 device pointers to each expert's B_packed.
    pub(crate) packed_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's B_scale.
    pub(crate) scale_ptrs: DevicePtr,
    /// `[num_experts]` f32 per-expert scale2 values.
    pub(crate) scale2_vals: DevicePtr,
}

/// Device-resident top-k routing results that can be sliced across fused
/// small-M expert waves without recomputing the router projection.
#[derive(Clone, Copy)]
pub(super) struct PrecomputedRoutes {
    indices: DevicePtr,
    weights: DevicePtr,
}

/// Device-side pointer table for FP8 expert dispatch (one projection).
///
/// FP8 experts use 2 pointer arrays (weight + block_scale) instead of
/// NVFP4's 3 (packed + scale + scale2). The fused FP8 MoE kernel indexes
/// these tables by expert_id to load the correct FP8 weight matrix.
pub(crate) struct Fp8ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
}

/// Checkpoint-native BF16 weights for a shared expert.
///
/// This is intentionally independent of routed-expert precision. Models such
/// as Laguna ship NVFP4 routed experts but explicitly exempt the shared expert
/// from quantization, so coupling these pointers to the all-BF16 routed path
/// silently changes model numerics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bf16SharedExpert {
    gate_proj: DenseWeight,
    up_proj: DenseWeight,
    down_proj: DenseWeight,
}

impl Bf16SharedExpert {
    fn new(gate_proj: DenseWeight, up_proj: DenseWeight, down_proj: DenseWeight) -> Result<Self> {
        anyhow::ensure!(
            !gate_proj.weight.is_null() && !up_proj.weight.is_null() && !down_proj.weight.is_null(),
            "BF16 shared expert requires non-null gate/up/down weights"
        );
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// A shared expert kept outside the fused routed-expert kernels and run as
/// its own gate/up → silu_mul → down chain over the same scratch contract.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SeparateSharedExpert {
    /// Checkpoint-native BF16 (Laguna).
    Bf16(Bf16SharedExpert),
    /// FP4-Local's checkpoint-native MXFP8: E4M3 weights + E8M0 scales,
    /// projected by `mxfp8_proj.cu` — no BF16 expansion, no NVFP4 requant.
    Mxfp8 {
        gate_proj: Mxfp8Proj,
        up_proj: Mxfp8Proj,
        down_proj: Mxfp8Proj,
        /// Gate‖up row-concatenated `[2*inter, h]` twin plus its
        /// `[32, 2*inter]` BF16 landing buffer: one `mxfp8_proj` launch then
        /// `fused_silu_mul` per row-band ≤ 32 (the batchm arm's span).
        gate_up: Option<Mxfp8GateUp>,
        kernels: ops::Mxfp8ProjKernels,
        scratch: ops::Mxfp8Scratch,
    },
}

/// The fused small-row arm of a [`SeparateSharedExpert::Mxfp8`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct Mxfp8GateUp {
    /// `[2*inter, h]` — gate rows then up rows.
    pub proj: Mxfp8Proj,
    /// `[32, 2*inter]` BF16 — one batchm-chunk's gate|up activations.
    pub gate_up_out: DevicePtr,
    /// `fused_silu_mul` (`dense_gemm_bf16.cu`); `0` resolves to the unfused path.
    pub silu_mul: KernelHandle,
}

/// Unified expert pointer table for any quantization format.
///
/// Replaces the separate `ExpertPtrTable` (NVFP4) and `Fp8ExpertPtrTable` (FP8)
/// with a single enum. The MoE forward path matches on this to select the
/// correct fused kernel (moe_shared_expert_fused vs moe_shared_expert_fused_fp8).
#[allow(dead_code)]
pub(crate) enum ExpertPtrSet {
    /// NVFP4: 3 pointer arrays (packed_ptrs, scale_ptrs, per-expert scale2 f32).
    Nvfp4 {
        packed_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
        scale2_vals: DevicePtr,
    },
    /// FP8: 2 pointer arrays (weight_ptrs, block_scale_ptrs).
    Fp8 {
        weight_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
    },
}

impl MoeLayer {
    /// ARM-2 Phase-K routed-expert kernel-handle select. Returns the E8M0
    /// variant when the routed experts are native MXFP4 (`Mxfp4E8m0`), else the
    /// NVFP4 handle. Panics if E8M0 is selected but the `_e8m0` kernel is
    /// absent from this target (`try_kernel` gave 0) — that means a native
    /// checkpoint reached a build that never compiled the variant, which must
    /// be loud, not silent NVFP4-on-E8M0 garbage (the straggler net).
    #[inline]
    fn e8m0_or(
        &self,
        nvfp4: spark_runtime::gpu::KernelHandle,
        e8m0: spark_runtime::gpu::KernelHandle,
        site: &str,
    ) -> spark_runtime::gpu::KernelHandle {
        if self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0 {
            assert!(
                e8m0.0 != 0,
                "ARM-2 Phase-K: routed experts tagged Mxfp4E8m0 at {site}, but the \
                 _e8m0 kernel handle is unresolved (not compiled into this target)."
            );
            e8m0
        } else {
            nvfp4
        }
    }
}

// ── Sub-files (split for ≤500 LoC) ────────────────────────────────────────
mod types;
pub use types::MoeLayer;
mod compact_layout;
mod decode_m16;
mod dump;
mod ep_prefill;
mod forward;
mod lora;
mod lora_gateup;
mod lora_router;
pub(crate) use lora::MoeLoraWeights;
mod forward_atomic_c4;
mod forward_batched;
mod forward_batched_guards;
mod forward_batched_gate;
mod forward_c2;
mod forward_c4;
mod forward_independent;
pub(crate) use forward_independent::validate_independent_environment;
mod forward_ep;
mod forward_k2;
mod forward_k3;
mod forward_k4;
mod forward_k5;
mod forward_pair_shared;
pub(crate) use forward_c2::c2_compact_requested;
pub(crate) use forward_c4::c4_grouped_requested;
pub(crate) use forward_k5::{k5_fused_moe_hc_requested, k5_grouped_moe_requested};
pub(crate) use forward_pair_shared::{shared_reduce_overlap_requested, shared_tp_split_requested};
mod forward_pair_verify;
mod forward_phase;
mod forward_prefill;
mod forward_prefill_bf16;
mod forward_prefill_finish;
mod forward_prefill_fp8;
mod forward_prefill_phase;
mod forward_prefill_q38;
pub(crate) use forward_prefill_q38::{q38_requested, sp_shared_requested, w2_requested};
mod forward_prefill_q38_nodup;
pub(crate) use forward_prefill_q38_nodup::{nodup_requested, skips_routed_transpose};
mod forward_prefill_q38_rs;
mod forward_prefill_route;
pub(crate) mod forward_prefill_route_sp;
mod forward_prefill_routed;
pub(crate) use forward_prefill_routed::{grouped_cutlass_gate_up_enabled, prefill_fp8_down};
mod forward_prefill_router;
mod forward_prefill_tcp;
pub(crate) use forward_prefill_tcp::tcp_requested;
mod forward_rows;
mod forward_token_major;
mod gate_up_m16;
mod gate_up_repack;
#[cfg(test)]
mod gate_up_repack_test_gpu;
mod helpers_a;
mod helpers_b;
mod helpers_c;
mod helpers_shared;
mod route_dump;
mod shared_fp8_cache;
mod shared_fp8_cache_load;
mod shared_fp8_cache_output;
#[cfg(test)]
mod shared_fp8_cache_test_gpu;
mod shared_fp8_origin;
pub(crate) use gate_up_m16::validate_m16_gate_up_graphs;
pub(crate) use shared_fp8_cache::SharedFp8Reserve;
pub(crate) use shared_fp8_cache::validate_shared_fp8_cache_factory_reserve;
pub use shared_fp8_cache::{validate_shared_fp8_cache_graphs, validate_shared_fp8_cache_profile};
pub(crate) use shared_fp8_origin::load_glm_shared_fp8_weight;
mod init;
mod init_ctor;
mod init_optional;
mod inplace_transpose;
mod m5_projection_oracle;
mod m5_projections;
mod mmq_layout;
mod router_bn4;
mod router_prefill_bn32;
mod shared_fp8_cache_bytes;
mod shared_m16;
pub(crate) use m5_projections::validate_m5_projection_graphs;
#[cfg(test)]
mod mod_tests;
mod prequant_fp4;
mod prequant_fp4_c3;
mod prequant_fp4_down;
pub(crate) use prequant_fp4::with_owner_rows;
mod ptr_table_build;
mod qwen4exp_fast;
pub(crate) mod union_stats;
pub(crate) use ptr_table_build::*;

/// Exact `max_m_tiles` for the grouped GEMM grid: the largest per-expert row
/// count (adjacent diffs of the sorted `expert_offsets` prefix-sum), tiled by
/// 64 and clamped to `worst_case` for safety. `offsets_le` is the D2H copy of
/// the `(num_experts + 1)` u32 LE prefix-sum buffer.
pub(crate) fn exact_m_tiles(offsets_le: &[u8], worst_case: u32) -> u32 {
    let mut prev = 0u32;
    let mut max_rows = 0u32;
    for raw in offsets_le.chunks_exact(4).skip(1) {
        let cur = u32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
        max_rows = max_rows.max(cur.saturating_sub(prev));
        prev = cur;
    }
    max_rows.div_ceil(64).max(1).min(worst_case)
}

#[cfg(test)]
mod exact_m_tiles_tests {
    use super::exact_m_tiles;

    fn offsets(counts: &[u32]) -> Vec<u8> {
        let mut o = vec![0u8; 4];
        let mut acc = 0u32;
        for &c in counts {
            acc += c;
            o.extend_from_slice(&acc.to_le_bytes());
        }
        o
    }

    #[test]
    fn tiles_cases() {
        // all empty experts -> max diff 0 -> 1 tile
        assert_eq!(exact_m_tiles(&offsets(&[0, 0, 0]), 100), 1);
        // one hot expert takes all 512*22 routed rows
        assert_eq!(
            exact_m_tiles(&offsets(&[0, 512 * 22, 0]), 99999),
            (512 * 22) / 64
        );
        // balanced: 2816 rows spread evenly over 8 experts
        assert_eq!(exact_m_tiles(&offsets(&[352; 8]), 2816 / 64 + 1), 6);
        // clamp to worst_case when the real max exceeds it
        assert_eq!(exact_m_tiles(&offsets(&[0, 9000, 0]), 10), 10);
        // empty slice -> 1
        assert_eq!(exact_m_tiles(&[], 5), 1);
    }
}
