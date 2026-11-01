// SPDX-License-Identifier: AGPL-3.0-only

//! S7 verify-row dedup (`forward_rows_t`) kernel handles, from
//! `moe_shared_expert_fused_t`.
//!
//! gfx1151 builds the same file (strix-hip/common links the gb10 source,
//! which is portable), and every entry below is in its code object, so the
//! lookups are no longer gated off there. They were, and batched decode on
//! gfx1151 (MTP verify rows, C>1) fell back to one MoE launch chain per row:
//! framework f113 counted 97,920 `moe_expert_gate_up_shared_t` launches for
//! 2,040 tokens x 48 layers at C=8, and aggregate decode stayed flat from C4
//! to C8. The rows kernels are byte-identical per row by construction (see
//! `forward_rows_t`); `ATLAS_QWEN4EXP_BATCH_FAST_BISECT` bit 128 still turns
//! them off.

use super::*;

pub(super) struct TRowsKernels {
    pub(super) moe_expert_union_t_k: KernelHandle,
    pub(super) moe_expert_gate_up_shared_t_rows_k: [KernelHandle; 9],
    pub(super) moe_expert_silu_down_shared_t_rows_k: [KernelHandle; 9],
}

impl TRowsKernels {
    pub(super) fn resolve(gpu: &dyn GpuBackend) -> Self {
        Self {
            moe_expert_union_t_k: super::super::try_kernel_gated(
                true,
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_union_t",
            ),
            moe_expert_gate_up_shared_t_rows_k: std::array::from_fn(|r| {
                super::super::try_kernel_gated(
                    (2..=8).contains(&r),
                    gpu,
                    "moe_shared_expert_fused_t",
                    &format!("moe_expert_gate_up_shared_t_rows_v3_r{r}_b256"),
                )
            }),
            moe_expert_silu_down_shared_t_rows_k: std::array::from_fn(|r| {
                super::super::try_kernel_gated(
                    (2..=8).contains(&r),
                    gpu,
                    "moe_shared_expert_fused_t",
                    &format!("moe_expert_silu_down_shared_t_rows_v3_r{r}_b256"),
                )
            }),
        }
    }
}
