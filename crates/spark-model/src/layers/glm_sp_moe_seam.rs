// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_GLM_SP_GATHER_OVERLAP` across a layer boundary, on the four-rank
//! chain: a GLM MoE layer's output seam (the reduce-scatter of its routed
//! partials, the shared-expert blend and the FFN site's mHC post) runs block
//! by block inside the next layer's attention seam
//! (`qwen3_attention::hc_seam_overlapped`).
//!
//! The MoE unpermute-reduces one row block at a time, each block's
//! reduce-scatter behind it on the seam's side stream
//! (`glm_sp_overlap::reduce_ahead`), and leaves its blend pending here
//! ([`begin`]); the layer leaves its post pending too ([`defer_post`]). The
//! next attention seam then runs, per block once it is reduced, the blend,
//! the post and its own pre, norm and gather ([`Pending::run_rows`]). Each of
//! those kernels computes a row on its own, so every row gets the bytes of
//! the whole seam, which ran the same kernels over all of this rank's rows.
//!
//! A pending seam is never lost: whatever needs the reduced output or the
//! posted highway before the next attention seam runs it whole first
//! ([`flush`]) -- a capture layer, an attention seam that does not overlap, a
//! layer that cannot defer its post (the last). Opt-in,
//! `ATLAS_GLM_SP_MOE_SEAM=1`: on four nodes (integ27, 32K, four overlap
//! blocks) it measured 3521 tok/s against 3633 off, exact either way -- the
//! blocked reduce-scatter took ~8 ms where the whole one took ~5.6, more
//! than the unpermute it hid; with its unpermute on a 32-block grid
//! (`ATLAS_GLM_SP_MOE_SEAM_CTAS=32`) it measured 4255 against 4200 off
//! (job 51, 32K).
//!
//! With `ATLAS_RDMA_RS_FUSED_SEG` the chain runs the seam's reduce-scatter
//! as one fused op instead of one a block: the unpermute
//! (`moe_unpermute_rs_fused_vec8`) writes every row straight to where the op
//! reads it and publishes it piece by piece, so the network starts on the
//! first finished piece, and each block of the next attention seam runs once
//! its own pieces are final. Thread-local: a forward runs its layers in
//! order on one thread.

use std::cell::Cell;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, KernelHandle};

use super::glm_sp::SpRows;
use super::glm_sp_overlap;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// The shared-expert blend of the reduced rows: `out += sigmoid(input .
/// gate) * shared` (`ops::moe_batched_blend`), after `join` when given. The
/// pointers are at this rank's first row.
#[derive(Clone, Copy)]
pub struct Blend {
    pub kernel: KernelHandle,
    pub shared: DevicePtr,
    pub input: DevicePtr,
    pub gate: DevicePtr,
    pub join: Option<u64>,
}

/// A MoE output seam left to the next attention seam.
#[derive(Clone, Copy)]
pub struct Pending {
    /// `[total, hidden]` routed partials, reduced ahead.
    pub out: DevicePtr,
    sp: SpRows,
    blend: Option<Blend>,
    /// The FFN site's `hc_post` kernel, once the layer deferred it.
    post: Option<KernelHandle>,
}

thread_local! {
    static PENDING: Cell<Option<Pending>> = const { Cell::new(None) };
}

/// Whether a MoE layer on `sp` may leave its output seam pending: the
/// four-rank chain (the pair keeps its own seam), when opted in.
pub fn enabled(sp: SpRows) -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    sp.ranks > 2 && *ON.get_or_init(|| std::env::var("ATLAS_GLM_SP_MOE_SEAM").as_deref() == Ok("1"))
}

/// The grid of each unpermute launch of a seam reduced ahead
/// (`ATLAS_GLM_SP_MOE_SEAM_CTAS`; unset or 0: one block a token). A capped
/// grid leaves SMs to the reduce-scatter running beside it: seam1.sqlite's
/// first two reduce-scatter ops took ~3 ms each beside the whole-grid
/// unpermute, against ~1.3 ms alone.
pub fn unpermute_ctas() -> Option<u32> {
    static CTAS: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *CTAS.get_or_init(|| {
        std::env::var("ATLAS_GLM_SP_MOE_SEAM_CTAS")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &u32| n > 0)
    })
}

/// What [`begin`] asks its producer for: the unpermute of a row block of
/// every rank's part, or of every row as the producer of a fused chain
/// reduce-scatter (`ops::moe_unpermute_rs_fused`, with this plan).
pub enum Produce<'a> {
    Block((usize, usize)),
    Fused(&'a spark_comm::RsFusedPlan),
}

/// Reduce `out` ahead over `blocks` and leave the blend and the post
/// pending. When `fusable` (the producer has a fused kernel) and the chain
/// offers it (`ATLAS_RDMA_RS_FUSED_SEG`), one fused reduce-scatter whose
/// producer writes every row (`Produce::Fused`); else a reduce-scatter a
/// block, `produce(Produce::Block(block))` writing each block of every
/// rank's part first.
#[allow(clippy::too_many_arguments)]
pub fn begin(
    sp: SpRows,
    blocks: &[(usize, usize)],
    out: DevicePtr,
    blend: Option<Blend>,
    fusable: bool,
    ctx: &ForwardContext,
    stream: u64,
    mut produce: impl FnMut(Produce<'_>) -> Result<()>,
) -> Result<()> {
    anyhow::ensure!(
        PENDING.with(|p| p.get().is_none()),
        "a MoE output seam is already pending"
    );
    let h = ctx.config.hidden_size;
    let fused = match ctx.comm.filter(|_| fusable) {
        Some(comm) => {
            let (seg0, stride, bytes) = sp.chain_segments(out, h, (0, sp.rows));
            comm.rs_fused_begin(seg0, stride, bytes)?
        }
        None => None,
    };
    if ctx.gpu.op_cache().once("glm:sp_moe_seam") {
        tracing::info!(
            "ATLAS_GLM_SP_MOE_SEAM: MoE output seam reduced ahead in {} blocks (unpermute grid {:?}, {})",
            blocks.len(),
            unpermute_ctas(),
            match &fused {
                Some(f) => format!(
                    "one fused reduce-scatter, {} pieces of {} B",
                    f.plan.npieces, f.piece_bytes
                ),
                None => "a reduce-scatter a block".into(),
            }
        );
    }
    match fused {
        Some(f) => {
            let row = h * 2;
            anyhow::ensure!(
                f.piece_bytes % row == 0 && f.pieces * f.piece_bytes == sp.rows * row,
                "fused reduce-scatter pieces of {} B do not tile {} rows of {row} B",
                f.piece_bytes,
                sp.rows
            );
            glm_sp_overlap::reduce_ahead_fused(sp, blocks, out, f, ctx, stream, || {
                produce(Produce::Fused(&f.plan))
            })?
        }
        None => glm_sp_overlap::reduce_ahead(sp, blocks, out, h, ctx, stream, |b| {
            produce(Produce::Block(b))
        })?,
    }
    PENDING.with(|p| {
        p.set(Some(Pending {
            out,
            sp,
            blend,
            post: None,
        }))
    });
    Ok(())
}

/// The unpermute of [`ops::moe_unpermute_reduce_indexed_ep_grid`] as the producer
/// of a fused chain reduce-scatter (`moe_unpermute_rs_fused_vec8{,_e4m3}`):
/// it computes the rows of every piece of `plan` (`plan` words,
/// `spark_comm::RsFusedPlan::words`), token indices counted from `output`,
/// and writes and publishes them where the plan says.
#[allow(clippy::too_many_arguments)]
pub fn unpermute_rs_fused(
    gpu: &dyn spark_runtime::gpu::GpuBackend,
    kernel: KernelHandle,
    expert_output: DevicePtr,
    output: DevicePtr,
    (token_to_perm, topk_ids, topk_weights): (DevicePtr, DevicePtr, DevicePtr),
    hidden_size: u32,
    topk: u32,
    (local_expert_start, local_expert_end): (u32, u32),
    ctas: u32,
    e4m3_scales: Option<DevicePtr>,
    plan: &[u64],
    stream: u64,
) -> Result<()> {
    let launch = spark_runtime::kernel_args::KernelLaunch::new(gpu, kernel)
        .grid([ctas.max(1), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(expert_output)
        .arg_ptr(output)
        .arg_ptr(token_to_perm)
        .arg_ptr(topk_ids)
        .arg_ptr(topk_weights)
        .arg_u32(hidden_size)
        .arg_u32(topk)
        .arg_u32(local_expert_start)
        .arg_u32(local_expert_end);
    match e4m3_scales {
        Some(p) => launch.arg_ptr(p),
        None => launch,
    }
    .arg_words(plan)
    .launch(stream)
}

/// Leave the FFN site's `hc_post` of the pending seam's output to the next
/// attention seam. `false` (nothing pending): the caller posts now.
pub fn defer_post(kernel: KernelHandle) -> bool {
    PENDING.with(|p| match p.get() {
        Some(pending) if pending.post.is_none() => {
            p.set(Some(Pending {
                post: Some(kernel),
                ..pending
            }));
            true
        }
        _ => false,
    })
}

/// The pending seam, for the attention seam that runs it.
pub fn take() -> Option<Pending> {
    PENDING.with(Cell::take)
}

/// Run a pending seam whole now: wait for its reduce-scatter, then blend
/// and post every local row.
pub fn flush(ctx: &ForwardContext, stream: u64) -> Result<()> {
    match take() {
        Some(p) => p.run_whole(ctx, stream),
        None => Ok(()),
    }
}

impl Pending {
    /// [`flush`] of this taken seam.
    pub fn run_whole(&self, ctx: &ForwardContext, stream: u64) -> Result<()> {
        glm_sp_overlap::join_ahead(ctx, stream)?;
        self.run_rows((0, self.sp.rows), ctx, stream)
    }

    /// The blend and the post of local rows `[r0, r0 + rows)`, once reduced.
    pub fn run_rows(
        &self,
        (r0, rows): (usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let at = |ptr: DevicePtr| ptr.offset(r0 * h * 2);
        let out = at(self.sp.local(self.out, h));
        if let Some(b) = self.blend {
            if let Some(event) = b.join {
                ctx.gpu.stream_wait_event(stream, event)?;
            }
            // Both pending: one pass that writes the post's bytes (the
            // blended rows are read by nothing else).
            if let Some(kernel) = self.post {
                let (streams, post, comb) = crate::layers::qwen3_attention::hc_rows(ctx, r0);
                let ptrs = [out, at(b.shared), at(b.input), b.gate, streams, post, comb];
                let dims = [rows as u32, h as u32, ctx.config.hc_mult as u32];
                if ops::glm_decode_fuse::moe_blend_post(
                    ctx.gpu,
                    [b.kernel, kernel],
                    ptrs,
                    dims,
                    stream,
                )? {
                    return Ok(());
                }
            }
            ops::moe_batched_blend(
                ctx.gpu,
                b.kernel,
                out,
                at(b.shared),
                at(b.input),
                b.gate,
                h as u32,
                rows as u32,
                stream,
            )?;
        }
        if let Some(kernel) = self.post {
            let (streams, post, comb) = crate::layers::qwen3_attention::hc_rows(ctx, r0);
            ops::hc_post(
                ctx.gpu,
                kernel,
                out,
                streams,
                post,
                comb,
                streams,
                rows as u32,
                h as u32,
                ctx.config.hc_mult as u32,
                stream,
            )?;
        }
        Ok(())
    }
}
