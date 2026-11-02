// SPDX-License-Identifier: AGPL-3.0-only

//! Batched-verify / alternative-shape kernel handles for `MoeLayer`: the
//! batch2/batch3/batch5 fused decode kernels, the token-major prefill trio,
//! and the atomic/sorted decode path. Resolved for `MoeLayer::new_with_hash`.

use super::*;

pub(super) struct BatchedKernels {
    pub(super) moe_expert_gate_up_shared_batch2: KernelHandle,
    pub(super) moe_expert_silu_down_shared_batch2: KernelHandle,
    pub(super) moe_weighted_sum_blend_batch2: KernelHandle,
    pub(super) w4a16_gemv_batch2: KernelHandle,
    pub(super) moe_expert_gate_up_shared_batch3: KernelHandle,
    pub(super) moe_expert_silu_down_shared_batch3: KernelHandle,
    pub(super) moe_weighted_sum_blend_batch3: KernelHandle,
    pub(super) w4a16_gemv_batch3: KernelHandle,
    pub(super) w4a16_batchm: W4a16BatchmTiers,
    pub(super) w4a16_batch5_dual_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_token_major: KernelHandle,
    pub(super) moe_expert_silu_down_shared_token_major: KernelHandle,
    pub(super) moe_weighted_sum_blend_token_major: KernelHandle,
    pub(super) moe_decode_atomic_c4_silu_down_accum_k: KernelHandle,
    pub(super) moe_decode_atomic_c4_finalize_k: KernelHandle,
}

impl BatchedKernels {
    pub(super) fn resolve(gpu: &dyn GpuBackend, is_glm5_next: bool) -> Result<Self> {
        Ok(Self {
            moe_expert_gate_up_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_gate_up_shared_batch2")?,
            moe_expert_silu_down_shared_batch2: gpu
                .kernel("moe_fused_batch2", "moe_expert_silu_down_shared_batch2")?,
            moe_weighted_sum_blend_batch2: gpu
                .kernel("moe_fused_batch2", "moe_weighted_sum_blend_batch2")?,
            w4a16_gemv_batch2: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch2")?,
            moe_expert_gate_up_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_gate_up_shared_batch3")?,
            moe_expert_silu_down_shared_batch3: gpu
                .kernel("moe_fused_batch3", "moe_expert_silu_down_shared_batch3")?,
            moe_weighted_sum_blend_batch3: gpu
                .kernel("moe_fused_batch3", "moe_weighted_sum_blend_batch3")?,
            w4a16_gemv_batch3: gpu.kernel("w4a16_gemv", "w4a16_gemv_batch3")?,
            w4a16_batchm: W4a16BatchmTiers::resolve(gpu, is_glm5_next),
            w4a16_batch5_dual_k: super::super::try_kernel_gated(
                is_glm5_next,
                gpu,
                "w4a16_gemv",
                "w4a16_gemv_batch5_dual",
            ),
            moe_expert_gate_up_shared_token_major: gpu
                .kernel("moe_prefill", "moe_expert_gate_up_shared_prefill")?,
            moe_expert_silu_down_shared_token_major: gpu
                .kernel("moe_prefill", "moe_expert_silu_down_shared_prefill")?,
            moe_weighted_sum_blend_token_major: gpu
                .kernel("moe_prefill", "moe_weighted_sum_blend_prefill")?,
            moe_decode_atomic_c4_silu_down_accum_k: super::super::try_kernel(
                gpu,
                "moe_decode_atomic_c4",
                "moe_decode_atomic_c4_silu_down_accum",
            ),
            moe_decode_atomic_c4_finalize_k: super::super::try_kernel(
                gpu,
                "moe_decode_atomic_c4",
                "moe_decode_atomic_c4_finalize",
            ),
        })
    }
}
