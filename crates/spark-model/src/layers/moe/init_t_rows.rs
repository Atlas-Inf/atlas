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
                    &format!(
                        "moe_expert_gate_up_shared_t_rows_v3_r{r}_b{}",
                        t_rows_block()
                    ),
                )
            }),
            moe_expert_silu_down_shared_t_rows_k: std::array::from_fn(|r| {
                super::super::try_kernel_gated(
                    (2..=8).contains(&r),
                    gpu,
                    "moe_shared_expert_fused_t",
                    &format!(
                        "moe_expert_silu_down_shared_t_rows_v3_r{r}_b{}",
                        t_rows_block()
                    ),
                )
            }),
        }
    }
}

/// Columns per CTA (= threads per block) of the verify-row dedup kernels:
/// 256 everywhere by default. On gfx1151, `ATLAS_MOE_T_ROWS_BS=128` selects
/// the `_b128` instantiations (same per-column arithmetic; see
/// moe_shared_expert_fused_t.cu). Read once.
pub(super) fn t_rows_block() -> u32 {
    static BS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *BS.get_or_init(|| {
        if cfg!(atlas_hip) && std::env::var("ATLAS_MOE_T_ROWS_BS").as_deref() == Ok("128") {
            128
        } else {
            256
        }
    })
}
