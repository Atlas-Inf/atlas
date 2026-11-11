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

    /// Whether the SP prefill FFN over `sp` reads only this rank's rows of
    /// its normed input, gathering what it needs of the others itself
    /// (`moe::sp_quant_gather`): the seam before it then gathers no
    /// BF16 rows.
    pub fn sp_gathers_own_input(&self, sp: glm_sp::SpRows, ctx: &ForwardContext) -> bool {
        matches!(self, Self::Moe(m) if m.sp_quant_gather(sp, ctx))
    }

    /// The routed MoE's quantized input as a seam's per-block gather
    /// (`qwen3_attention::SeamGather`): its staged rows `(ptr, width)`, when
    /// [`Self::sp_gathers_own_input`].
    pub fn sp_route_staging(
        &self,
        sp: glm_sp::SpRows,
        ctx: &ForwardContext,
    ) -> Result<Option<(DevicePtr, usize)>> {
        match self {
            Self::Moe(m) if m.sp_quant_gather(sp, ctx) => m.sp_route_staging(sp, ctx).map(Some),
            _ => Ok(None),
        }
    }

    /// Route, quantize and stage this rank's rows `block` of `normed` for
    /// the seam's per-block gather ([`Self::sp_route_staging`]).
    pub fn sp_route_rows(
        &self,
        normed: DevicePtr,
        sp: glm_sp::SpRows,
        block: (usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.sp_route_rows(normed, sp, block, ctx, stream),
            _ => anyhow::bail!("per-block routing on a layer without a routed MoE"),
        }
    }

    /// Sequence-parallel prefill FFN over a normed `[sp.total(), H]` input
    /// (all rows gathered, or only this rank's when [`Self::sp_gathers_own_input`]).
    /// Returns this rank's `[sp.rows, H]` output rows:
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
                let ran = m.forward_prefill(normed, sp.total(), ctx, stream);
                super::moe::forward_prefill_route_sp::clear_local_routes();
                ran?;
                Ok(sp.local(ctx.buffers.moe_output(), h))
            }
            Self::Dense(d) => {
                d.forward_prefill(sp.local(normed, h), sp.rows, ctx, stream)?;
                Ok(ctx.buffers.moe_output())
            }
            Self::None => anyhow::bail!("SP prefill FFN on a layer without an FFN"),
        }
    }

    /// `ATLAS_QWEN4EXP_PREFILL_SP_ROUTE`: route this rank's rows of the SP
    /// MoE input (`[sp.total(), H]`, local rows set) ahead of
    /// [`Self::forward_prefill_sp`], before the gather joins
    /// (`moe::forward_prefill_route_sp`). Nothing off a MoE FFN.
    pub fn route_local_rows(
        &self,
        normed: DevicePtr,
        sp: glm_sp::SpRows,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        match self {
            Self::Moe(m) => m.route_local_rows(normed, sp, ctx, stream),
            _ => Ok(()),
        }
    }
}

/// Seam blocks one per-block gather of a quantized seam input covers (the
/// four-rank chain's MoE, KDA and MLA quant gathers; the pair has none): one,
/// the four-rank profile's. pf10.sqlite measured four one-block gathers at
/// ~0.65 ms each against ~0.55 ms of mHC a block, so a gather a block queued
/// behind the last and ran ~1 ms past the seam; the quantized gathers since
/// move far fewer bytes.
pub(crate) const SEAM_GATHER_GROUP: usize = 1;
