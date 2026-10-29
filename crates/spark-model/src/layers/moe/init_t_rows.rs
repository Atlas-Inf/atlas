// SPDX-License-Identifier: AGPL-3.0-only

//! S7 verify-row dedup (`forward_rows_t`) kernel handles: gb10-only entries
//! in `moe_shared_expert_fused_t`, gated so other targets get
//! `KernelHandle(0)`.

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
                !cfg!(atlas_hip),
                gpu,
                "moe_shared_expert_fused_t",
                "moe_expert_union_t",
            ),
            moe_expert_gate_up_shared_t_rows_k: std::array::from_fn(|r| {
                super::super::try_kernel_gated(
                    !cfg!(atlas_hip) && (2..=8).contains(&r),
                    gpu,
                    "moe_shared_expert_fused_t",
                    &format!("moe_expert_gate_up_shared_t_rows_v3_r{r}_b256"),
                )
            }),
            moe_expert_silu_down_shared_t_rows_k: std::array::from_fn(|r| {
                super::super::try_kernel_gated(
                    !cfg!(atlas_hip) && (2..=8).contains(&r),
                    gpu,
                    "moe_shared_expert_fused_t",
                    &format!("moe_expert_silu_down_shared_t_rows_v3_r{r}_b256"),
                )
            }),
        }
    }
}
