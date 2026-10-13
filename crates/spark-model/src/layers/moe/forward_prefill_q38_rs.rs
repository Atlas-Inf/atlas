// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE=1`: the q38 local-route unpermute of a
//! qwen4_exp SP chunk, row range by row range, with the MoE reduce-scatter
//! pipelined behind it (`layers::qwen4exp_sp_pipe::compute_and_reduce_scatter`).
//!
//! `moe_q38_unpermute_local` is one CTA per token reading only that token's
//! routes, so a row range launched apart (pointers advanced to its first
//! token) writes exactly the bytes the whole-chunk launch writes there.

use super::*;

impl MoeLayer {
    /// Unpermute this chunk into `ptrs[1]` and reduce-scatter it, the peer's
    /// window first. `ptrs` = [expert_down_out, output, token_to_perm,
    /// topk_ids, topk_weights]; `dims` = [hidden, top_k]. `Ok(false)`: not
    /// taken (switch off, the q38 unpermute does not serve, or the pair cannot
    /// pipeline); nothing was launched.
    pub(super) fn q38_unpermute_reduce_scatter(
        &self,
        sp: crate::layers::glm_sp::SpRows,
        ptrs: [DevicePtr; 5],
        dims: [u32; 2],
        ctx: &ForwardContext,
        stream: u64,
        on_wire: impl FnOnce(u64) -> Result<()>,
    ) -> Result<bool> {
        let [expert_out, output, token_to_perm, topk_ids, topk_weights] = ptrs;
        let [hidden, topk] = dims;
        if topk_ids.is_null() || !self.q38_rs_pipe_serves(sp, hidden, ctx) {
            return Ok(false);
        }
        let route = |r0: usize| (r0 * topk as usize * 4) as u64;
        crate::layers::qwen4exp_sp_pipe::compute_and_reduce_scatter(
            sp,
            output,
            hidden as usize,
            ctx,
            stream,
            |r0, n, s| {
                let served = self.try_q38_unpermute(
                    expert_out,
                    output.offset(r0 * hidden as usize * 2),
                    DevicePtr(token_to_perm.0 + route(r0)),
                    DevicePtr(topk_ids.0 + route(r0)),
                    DevicePtr(topk_weights.0 + route(r0)),
                    [hidden, n as u32, topk],
                    ctx,
                    s,
                )?;
                anyhow::ensure!(served, "q38 unpermute refused a row range it served");
                Ok(())
            },
            on_wire,
        )
    }

    /// Whether [`Self::q38_unpermute_reduce_scatter`] runs for this layer's
    /// split (the q38 unpermute serves and the pipe is available).
    pub(super) fn q38_rs_pipe_serves(
        &self,
        sp: crate::layers::glm_sp::SpRows,
        hidden: u32,
        ctx: &ForwardContext,
    ) -> bool {
        let inter = ctx.config.routed_inter_local() as u32;
        hidden.is_multiple_of(8)
            && hidden / 8 <= 1024
            && self.lora.is_none()
            && self.q38_routed_serves(hidden, inter, ctx)
            && crate::layers::qwen4exp_sp_pipe::rs_available(sp, hidden as usize, ctx)
    }

    /// ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE with `_SP_SHARED`: the shared
    /// expert of this rank's rows runs while the MoE reduce-scatter is on the
    /// wire (after the unpermute) instead of before the routed experts. It
    /// reads `input` and writes its own scratch and `attn_output`, which the
    /// routed path and the unpermute never touch, so its bytes are unchanged.
    pub(super) fn defer_shared_to_rs(
        &self,
        sp: Option<crate::layers::glm_sp::SpRows>,
        has_shared: bool,
        overlap_shared_reduce: bool,
        hidden: u32,
        ctx: &ForwardContext,
    ) -> bool {
        has_shared
            && !overlap_shared_reduce
            && sp.is_some_and(|sp| !sp.full_shared && self.q38_rs_pipe_serves(sp, hidden, ctx))
    }
}
