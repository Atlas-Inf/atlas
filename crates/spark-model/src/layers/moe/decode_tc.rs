// SPDX-License-Identifier: AGPL-3.0-only

//! Tensor-core MXFP4 decode SiLU+down dispatch (mimo-v2.6-flash target).
//!
//! Opt-in via `ATLAS_MOE_MXFP4_TC_SD=1`; falls back to the scalar dual-format
//! `_shared_t` kernel when the env is unset, the experts are not Mxfp4E8m0,
//! the kernel is not compiled into this target, or the tile geometry does
//! not fit (N % 16, K % 64). Single-token decode only. The gate+up twin
//! lives in `mod.rs` next to `e8m0_or`.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::{ExpertPtrTable, MoeLayer};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::QuantizedWeight;

impl MoeLayer {
    fn silu_down_tc_handle(&self, n: usize, k: usize) -> Option<KernelHandle> {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let enabled =
            *ON.get_or_init(|| std::env::var("ATLAS_MOE_MXFP4_TC_SD").is_ok_and(|v| v == "1"));
        if enabled
            && self.experts_scale_kind == crate::weight_map::WeightQuantFormat::Mxfp4E8m0
            && self.moe_expert_silu_down_tc_k.0 != 0
            && n.is_multiple_of(16)
            && k.is_multiple_of(64)
        {
            Some(self.moe_expert_silu_down_tc_k)
        } else {
            None
        }
    }

    /// Decode SiLU+down over routed experts: the tensor-core MXFP4 kernel
    /// when opted in and compatible, else the scalar dual-format kernel.
    #[allow(clippy::too_many_arguments)]
    pub(in crate::layers::moe) fn decode_silu_down_shared_t(
        &self,
        ctx: &ForwardContext,
        gate_out: DevicePtr,
        up_out: DevicePtr,
        down_t: &ExpertPtrTable,
        output: DevicePtr,
        expert_indices: DevicePtr,
        sh_gate_in: DevicePtr,
        sh_up_in: DevicePtr,
        sh_down_t: &QuantizedWeight,
        sh_down_out: DevicePtr,
        n: u32,
        k: u32,
        top_k: u32,
        stream: u64,
        site: &str,
    ) -> Result<()> {
        if let Some(tc_k) = self.silu_down_tc_handle(n as usize, k as usize) {
            return ops::moe_expert_silu_down_tc(
                ctx.gpu,
                tc_k,
                gate_out,
                up_out,
                down_t.packed_ptrs,
                down_t.scale_ptrs,
                output,
                expert_indices,
                sh_gate_in,
                sh_up_in,
                sh_down_out,
                n,
                k,
                top_k,
                stream,
            );
        }
        ops::moe_expert_silu_down_shared_t(
            ctx.gpu,
            self.e8m0_or(
                self.moe_expert_silu_down_shared_t_k,
                self.moe_expert_silu_down_shared_t_e8m0_k,
                site,
            ),
            gate_out,
            up_out,
            down_t.packed_ptrs,
            down_t.scale_ptrs,
            down_t.scale2_vals,
            output,
            expert_indices,
            sh_gate_in,
            sh_up_in,
            sh_down_t,
            sh_down_out,
            n,
            k,
            top_k,
            stream,
        )
    }
}
