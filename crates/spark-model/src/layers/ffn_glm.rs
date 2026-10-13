// SPDX-License-Identifier: AGPL-3.0-only

//! `FfnComponent` dispatch for the fixed K=4/K=5 verifier and SP prefill.

use super::{FfnComponent, glm_sp};
use crate::layer::ForwardContext;
use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

impl FfnComponent {
    /// Fixed four-row speculative-verifier FFN. Returns the actual output
    /// buffer because GLM's MoE composition safely stages over its norm input,
    /// while dense batchm writes the conventional `moe_output` scratch.
    pub fn forward_k4(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward_k4(input, ctx, stream),
            Self::Dense(d) if d.can_forward_km(4) => {
                d.forward_km(input, 4, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::Dense(d) => {
                d.forward_prefill(input, 4, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::None => Ok(input),
        }
    }

    /// Fixed five-row speculative-verifier FFN. GLM MoE layers use one M5
    /// shared-expert pass plus fused K2/K3 routed dispatch.
    pub fn forward_k5(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        match self {
            Self::Moe(m) => m.forward_k5(input, ctx, stream),
            Self::Dense(d) if d.can_forward_km(5) => {
                d.forward_km(input, 5, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::Dense(d) => {
                d.forward_prefill(input, 5, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::None => Ok(input),
        }
    }

    /// Fixed K=5 FFN for an mHC caller capable of fusing GLM's EP shared
    /// expert blend into its post-step. `Some(gate)` means the returned MoE
    /// output is routed-only and already globally reduced; the shared output
    /// remains in `buffers.attn_output()`.
    pub fn forward_k5_for_hc(
        &self,
        input: DevicePtr,
        allow_deferred_shared_hc: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<(DevicePtr, Option<DevicePtr>)> {
        match self {
            Self::Moe(m) => m.forward_k5_for_hc(input, allow_deferred_shared_hc, ctx, stream),
            _ => Ok((self.forward_k5(input, ctx, stream)?, None)),
        }
    }

    /// Execute the unfused shared-expert blend for the GLM K=5 exactness
    /// oracle after [`Self::forward_k5_for_hc`] returned `Some(gate)`.
    #[allow(clippy::too_many_arguments)]
    pub fn finish_k5_deferred_shared_blend(
        &self,
        routed: DevicePtr,
        shared: DevicePtr,
        input: DevicePtr,
        gate_weight: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => {
                m.finish_k5_deferred_shared_blend(routed, shared, input, gate_weight, ctx, stream)
            }
            _ => anyhow::bail!("deferred K=5 shared blend requires a MoE FFN"),
        }
    }

    /// Sequence-parallel prefill FFN over a normed `[2 * sp.rows, H]` input
    /// (all rows gathered). Returns this rank's `[sp.rows, H]` output rows:
    /// the MoE runs every row and reduce-scatters (`layers::glm_sp`), the
    /// replicated dense FFN runs only the local rows.
    pub fn forward_prefill_sp(
        &self,
        normed: DevicePtr,
        sp: glm_sp::SpRows,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size;
        match self {
            Self::Moe(m) => {
                m.forward_prefill(normed, 2 * sp.rows, ctx, stream)?;
                Ok(sp.local(ctx.buffers.moe_output(), h))
            }
            Self::Dense(d) => {
                d.forward_prefill(sp.local(normed, h), sp.rows, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::None => anyhow::bail!("SP prefill FFN on a layer without an FFN"),
        }
    }
}
