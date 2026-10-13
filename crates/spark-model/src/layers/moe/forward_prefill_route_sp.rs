// SPDX-License-Identifier: AGPL-3.0-only

//! `ATLAS_QWEN4EXP_PREFILL_SP_ROUTE=1` (with the qwen4_exp SP prefill): each
//! rank routes only its own rows, the router GEMM and the top-k, then the
//! pair all-gathers the routes (ids and weights, 80 bytes a row) instead of
//! both ranks routing every row. The local half runs at the MoE input's
//! gather site, after the collapse and before the join
//! (`qwen4exp_sp_pipe::collapse_and_gather_then`), so under
//! `ATLAS_QWEN4EXP_PREFILL_SP_PIPE` it overlaps the gather's last piece.
//!
//! Exact by construction: the q38 router GEMM (`dense_gemm_bf16_pipelined`,
//! 128-row tiles, one in-order chain per output) and `moe_topk_softmax_batched`
//! (one row at a time) give a row the same bytes at any row count, as long as
//! both halves take the q38 arm (32 rows or more). The GPU test checks the
//! halves against the whole chunk.

use std::cell::Cell;

use super::*;
use crate::layers::glm_sp::SpRows;
use spark_runtime::kernel_args::KernelLaunch;

/// `ATLAS_QWEN4EXP_PREFILL_SP_ROUTE=1`.
pub(crate) fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_SP_ROUTE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

thread_local! {
    /// `(ptr, bytes)` of this thread's route buffer: `[total, top_k]` ids,
    /// then `[total, top_k]` weights.
    static ROUTES: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
    /// The input whose local rows were routed: (input, total rows).
    static ROUTED: Cell<Option<(u64, usize)>> = const { Cell::new(None) };
}

impl MoeLayer {
    /// Route this rank's rows of `input` (`[sp.total(), H]`, local rows
    /// final on `stream`) into the route buffer, for
    /// [`MoeLayer::take_local_routes`]. Launches nothing when not served.
    pub(crate) fn route_local_rows(
        &self,
        input: DevicePtr,
        sp: SpRows,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        ROUTED.with(|c| c.set(None));
        let h = ctx.config.hidden_size;
        let serves = requested()
            && ctx.config.model_type == "qwen4_exp"
            && !ctx.graph_capture
            && ctx.comm.is_some_and(|c| c.world_size() == 2)
            && ctx.config.ep_world_size > 1
            && sp.rows >= 32
            && sp.peer_rows >= 32
            && self.gate_fp8.is_none()
            && self.tid2eid_dev.is_none()
            && self.correction_bias_dev.is_none()
            && self.weights.router_pre_norm.is_none()
            && self.lora.is_none();
        let Some(router) = self.gate_nvfp4.as_ref().filter(|_| serves) else {
            return Ok(());
        };
        let logits = ctx.buffers.gate_logits();
        let rows = sp.rows as u32;
        let local_in = sp.local(input, h);
        let n_out = self.router_logits_n;
        if !self.try_q38_router(local_in, router, logits, rows, n_out, h as u32, ctx, stream)? {
            return Ok(());
        }
        let top_k = ctx.config.num_experts_per_tok;
        let route = top_k * 4;
        let most = crate::layers::qwen4exp_sp_pipe::prealloc_rows(ctx).max(sp.total());
        let buf = routes(ctx.gpu, 2 * most * route, stream)?;
        let weights = buf.offset(sp.total() * route);
        self.prefill_topk(
            logits,
            buf.offset(sp.row0 * route),
            weights.offset(sp.row0 * route),
            rows,
            ctx.config.num_experts as u32,
            top_k as u32,
            ctx,
            stream,
        )?;
        ROUTED.with(|c| c.set(Some((input.0, sp.total()))));
        Ok(())
    }

    /// When [`MoeLayer::route_local_rows`] routed this `input`, all-gather
    /// the routes and land them at `indices` / `weights` (the routing steps'
    /// outputs); `Ok(false)` otherwise (route as usual).
    pub(super) fn take_local_routes(
        &self,
        input: DevicePtr,
        n: usize,
        [indices, weights]: [DevicePtr; 2],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let Some(sp) = ROUTED
            .with(Cell::take)
            .filter(|&r| r == (input.0, n))
            .and_then(|_| crate::layers::glm_sp::current())
            .filter(|sp| sp.total() == n)
        else {
            return Ok(false);
        };
        let top_k = ctx.config.num_experts_per_tok;
        let bytes = n * top_k * 4;
        let buf = DevicePtr(ROUTES.with(Cell::get).0);
        for (i, dst) in [indices, weights].into_iter().enumerate() {
            let src = buf.offset(i * bytes);
            sp.all_gather(src, top_k * 2, ctx, stream)?;
            ctx.gpu.copy_d2d_async(src, dst, bytes, stream)?;
        }
        Ok(true)
    }
}

/// The q38 router GEMM: `dense_gemm_bf16_pipelined` of `[n, h]` rows by the
/// dequantised `[n_out, h]` weight. `ptrs` = [input, weight, logits].
pub(super) fn q38_router_gemm(
    gpu: &dyn GpuBackend,
    k_gemm: KernelHandle,
    [input, weight, logits]: [DevicePtr; 3],
    [n, n_out, h]: [u32; 3],
    stream: u64,
) -> Result<()> {
    KernelLaunch::new(gpu, k_gemm)
        .grid([n_out.div_ceil(128), n.div_ceil(128), 1])
        .block([256, 1, 1])
        .arg_ptr(input)
        .arg_ptr(weight)
        .arg_ptr(logits)
        .arg_u32(n)
        .arg_u32(n_out)
        .arg_u32(h)
        .launch(stream)
}

/// Drop an untaken local routing (the MoE took a body that routes itself),
/// so no later chunk can take it.
pub(crate) fn clear_local_routes() {
    ROUTED.with(|c| c.set(None));
}

/// This thread's route buffer, at least `bytes`, grown once `stream` drains
/// (its gathers run on `stream`).
fn routes(gpu: &dyn GpuBackend, bytes: usize, stream: u64) -> Result<DevicePtr> {
    let (ptr, size) = ROUTES.with(Cell::get);
    if size >= bytes {
        return Ok(DevicePtr(ptr));
    }
    if ptr != 0 {
        gpu.synchronize(stream)?;
        gpu.free(DevicePtr(ptr))?;
    }
    let p = gpu.alloc(bytes)?;
    ROUTES.with(|s| s.set((p.0, bytes)));
    Ok(p)
}

#[cfg(test)]
#[path = "forward_prefill_route_sp_gpu_tests.rs"]
mod gpu_tests;
