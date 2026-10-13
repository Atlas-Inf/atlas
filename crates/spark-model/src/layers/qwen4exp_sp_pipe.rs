// SPDX-License-Identifier: AGPL-3.0-only
//! Slab-pipelined SP all-gather for qwen4_exp prefill
//! (`ATLAS_QWEN4EXP_PREFILL_SP_PIPE=1`, with `ATLAS_QWEN4EXP_PREFILL_SP=1`).
//!
//! Under SP each rank collapses its own rows of a block input (the mHC
//! `hc_pre`, in 2048-row slabs) and then all-gathers them before the block.
//! The gather ran after the whole collapse, and on the pair it was the
//! largest idle in a 16K prefill: 3-4 ms a site, ~340 ms a prefill on rank 0
//! (nsys, 2026-10-06) -- copy-engine staging and the 200G wire while the GPU
//! waits.
//!
//! Here the gather goes out in window pieces of one slab each, on a side
//! stream, every piece as soon as the slabs it reads are collapsed
//! (`hc_pre_gemm` reports each finished slab through [`slab_done`]), so the
//! wire time of all but the last piece hides under the rest of the collapse.
//! The main stream joins the side stream only before the block reads the
//! gathered rows.
//!
//! Exact by construction: the same bytes move to the same places. Each rank
//! sends the same window (`glm_sp_uneven::plan`) and lands the peer's the
//! same way (in place, or staged and the peer's region copied out), in
//! slab-sized pieces instead of one payload. Both ranks issue the same piece
//! sizes in the same order, as the pair exchange requires, and every
//! exchange stays totally ordered by stream dependencies (the side stream
//! starts after the main stream's previous exchange; the main stream's next
//! one follows the join), which the pair's two-slot reuse relies on. On an
//! attention layer the collapse writes compacted rows to a scratch that is
//! copied into the gathered buffer; that copy moves per slab too.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::cell::{Cell, RefCell};

use super::glm_sp::SpRows;
use crate::layer::ForwardContext;

#[path = "qwen4exp_sp_side.rs"]
mod side_ops;
pub use side_ops::{RsOffer, SideExchanges, prealloc_rows, rs_offered, rs_took};

/// `ATLAS_QWEN4EXP_PREFILL_SP_PIPE=1`.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_SP_PIPE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Rows a piece carries: the collapse slab.
const PIECE: usize = crate::layers::ops::HC_PREFILL_SLAB as usize;

/// Piece bookkeeping of one gather (plain data, unit-testable).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Pieces {
    /// Window rows; this rank's row 0 sits `lead` rows into its window.
    m: usize,
    lead: usize,
    /// This rank's rows.
    rows: usize,
    /// Pieces sent; local rows finished.
    sent: usize,
    done: usize,
}

impl Pieces {
    fn count(&self) -> usize {
        self.m.div_ceil(PIECE)
    }
    /// Local rows piece `i` reads.
    fn need(&self, i: usize) -> usize {
        ((i + 1) * PIECE).saturating_sub(self.lead).min(self.rows)
    }
    /// The next piece to send, as (first window row, rows), once its rows
    /// are done.
    fn next_ready(&self) -> Option<(usize, usize)> {
        let i = self.sent;
        (i < self.count() && self.need(i) <= self.done)
            .then(|| (i * PIECE, PIECE.min(self.m - i * PIECE)))
    }
}

/// The side stream and its two events, made once per thread.
#[derive(Clone, Copy)]
struct Side {
    stream: u64,
    to_side: u64,
    to_main: u64,
}

thread_local! {
    static SIDE: Cell<Option<Side>> = const { Cell::new(None) };
    /// `(ptr, bytes)` of this thread's landing stage.
    static STAGE: Cell<(u64, usize)> = const { Cell::new((0, 0)) };
    /// The [`Gather`] whose collapse is running ([`Gather::during`]).
    static ACTIVE: Cell<Option<*const ()>> = const { Cell::new(None) };
}

/// A gather begun by [`begin`]: run the collapse in [`Gather::during`],
/// then [`Gather::finish`] before the gathered buffer is read.
pub struct Gather<'a> {
    ctx: &'a ForwardContext<'a>,
    side: Side,
    /// `[total, width]` buffer the rows are gathered in; row bytes.
    buf: DevicePtr,
    row: usize,
    /// This rank's first row, and the first row of the window it sends.
    row0: usize,
    send0: usize,
    /// This rank's rows compacted at row 0 elsewhere (attention layers).
    copy_from: Option<DevicePtr>,
    /// Where the peer's window lands; the region copy-out when staged.
    land: DevicePtr,
    copy_out: Option<(DevicePtr, DevicePtr, usize)>,
    pieces: RefCell<Pieces>,
}

/// Begin a slab-pipelined all-gather of `buf` (`[total, width]` BF16) for a
/// collapse about to run on `stream`; `None` when not requested, not split,
/// or the pair cannot take a piece -- the caller then gathers as before.
/// `copy_from`: this rank's rows compacted at row 0 of another buffer, copied
/// into `buf` slab by slab before they are sent.
pub fn begin<'a>(
    sp: Option<SpRows>,
    buf: DevicePtr,
    copy_from: Option<DevicePtr>,
    width: usize,
    ctx: &'a ForwardContext<'a>,
    stream: u64,
) -> Result<Option<Gather<'a>>> {
    match sp {
        Some(sp) if requested() => begin_split(sp, buf, copy_from, width, ctx, stream),
        _ => Ok(None),
    }
}

/// Run `collapse`, which sets this rank's `rows` rows of `buf` (`[total,
/// width]` BF16, rows at their chunk positions) -- or, with `copy_from`, the
/// same rows compacted at row 0 there, which are then copied into `buf` --
/// and all-gather `buf` when the chunk is split: slab-pipelined under the
/// switch, else after the collapse as before.
#[allow(clippy::too_many_arguments)]
pub fn collapse_and_gather(
    sp: Option<SpRows>,
    buf: DevicePtr,
    copy_from: Option<DevicePtr>,
    rows: usize,
    width: usize,
    ctx: &ForwardContext<'_>,
    stream: u64,
    collapse: impl FnOnce() -> Result<()>,
) -> Result<()> {
    let rows_width = [rows, width];
    collapse_and_gather_then(
        sp,
        buf,
        copy_from,
        rows_width,
        ctx,
        stream,
        collapse,
        |_| Ok(()),
    )
}

/// [`collapse_and_gather`] with `local(stream)` run once this rank's rows
/// are in `buf` and before the peer's are waited for (pipelined: while the
/// last piece is on the wire).
#[allow(clippy::too_many_arguments)]
pub fn collapse_and_gather_then(
    sp: Option<SpRows>,
    buf: DevicePtr,
    copy_from: Option<DevicePtr>,
    [rows, width]: [usize; 2],
    ctx: &ForwardContext<'_>,
    stream: u64,
    collapse: impl FnOnce() -> Result<()>,
    local: impl FnOnce(u64) -> Result<()>,
) -> Result<()> {
    if let Some(g) = begin(sp, buf, copy_from, width, ctx, stream)? {
        g.during(collapse)?;
        return g.finish_after(stream, local);
    }
    collapse()?;
    if let Some(src) = copy_from {
        let at = sp.map_or(buf, |sp| sp.local(buf, width));
        ctx.gpu.copy_d2d_async(src, at, rows * width * 2, stream)?;
    }
    local(stream)?;
    match sp {
        Some(sp) => sp.all_gather(buf, width, ctx, stream),
        None => Ok(()),
    }
}

/// [`begin`] for a split chunk, the switch aside.
pub(super) fn begin_split<'a>(
    sp: SpRows,
    buf: DevicePtr,
    copy_from: Option<DevicePtr>,
    width: usize,
    ctx: &'a ForwardContext<'a>,
    stream: u64,
) -> Result<Option<Gather<'a>>> {
    let Some(comm) = ctx.comm else {
        return Ok(None);
    };
    if ctx.config.model_type != "qwen4_exp" || ctx.graph_capture {
        return Ok(None);
    }
    let row = width * 2;
    let plan = super::glm_sp_uneven::plan(sp, false);
    if !comm.supports_exchange_async(plan.m.min(PIECE) * row) {
        return Ok(None);
    }
    let gpu = ctx.gpu;
    let side = side(gpu)?;
    // The side stream starts after everything the main stream enqueued,
    // the previous exchange included.
    gpu.record_event(side.to_side, stream)?;
    gpu.stream_wait_event(side.stream, side.to_side)?;
    let dst = buf.offset(plan.recv0 * row);
    let (land, copy_out) = if plan.recv_n == plan.m {
        (dst, None)
    } else {
        let stage = stage(
            gpu,
            (plan.m * row).max(prealloc_rows(ctx) * row),
            stream,
            side.stream,
        )?;
        let src = stage.offset(plan.skip * row);
        (stage, Some((src, dst, plan.recv_n * row)))
    };
    Ok(Some(Gather {
        ctx,
        side,
        buf,
        row,
        row0: sp.row0,
        send0: plan.send0,
        copy_from,
        land,
        copy_out,
        pieces: RefCell::new(Pieces {
            m: plan.m,
            lead: sp.row0 - plan.send0,
            rows: sp.rows,
            sent: 0,
            done: 0,
        }),
    }))
}

/// The collapse finished local rows `[0, rows_done)` on `stream`. Called by
/// `hc_pre_gemm` after each slab; a no-op unless a [`Gather::during`] runs.
pub fn slab_done(rows_done: usize, stream: u64) -> Result<()> {
    let Some(g) = ACTIVE.with(Cell::get) else {
        return Ok(());
    };
    // SAFETY: `ACTIVE` points at the `Gather` whose `during` is on this
    // thread's stack: set on entry to `during`, cleared when it returns or
    // unwinds (its drop guard).
    // The pointer never leaves the thread, and `progress` takes `&self`.
    let g = unsafe { &*(g as *const Gather<'_>) };
    g.progress(rows_done, stream)
}

impl Gather<'_> {
    /// Run `collapse`, sending each piece as soon as its slabs are done.
    pub fn during<R>(&self, collapse: impl FnOnce() -> Result<R>) -> Result<R> {
        ensure!(
            ACTIVE.with(Cell::get).is_none(),
            "qwen4exp SP pipe: a gather is already running"
        );
        /// Clears `ACTIVE` when `during` leaves, a panic in `collapse` included.
        struct Clear;
        impl Drop for Clear {
            fn drop(&mut self) {
                ACTIVE.with(|c| c.set(None));
            }
        }
        ACTIVE.with(|c| c.set(Some(self as *const Gather<'_> as *const ())));
        let _clear = Clear;
        collapse()
    }

    /// Local rows `[0, rows_done)` are set on `stream`: copy them in (when
    /// compacted elsewhere) and send every piece they complete.
    fn progress(&self, rows_done: usize, stream: u64) -> Result<()> {
        let gpu = self.ctx.gpu;
        let mut p = self.pieces.borrow_mut();
        let rows_done = rows_done.min(p.rows);
        if rows_done <= p.done {
            return Ok(());
        }
        if let Some(src) = self.copy_from {
            gpu.copy_d2d_async(
                src.offset(p.done * self.row),
                self.buf.offset((self.row0 + p.done) * self.row),
                (rows_done - p.done) * self.row,
                stream,
            )?;
        }
        p.done = rows_done;
        if p.next_ready().is_none() {
            return Ok(());
        }
        gpu.record_event(self.side.to_side, stream)?;
        gpu.stream_wait_event(self.side.stream, self.side.to_side)?;
        let comm = self.ctx.comm.expect("checked in begin");
        while let Some((w0, rows)) = p.next_ready() {
            ensure!(
                comm.exchange_async(
                    self.buf.offset((self.send0 + w0) * self.row).0,
                    self.land.offset(w0 * self.row).0,
                    rows * self.row,
                    false,
                    self.side.stream,
                )?,
                "qwen4exp SP pipe: pair exchange refused ({rows} rows)"
            );
            p.sent += 1;
        }
        Ok(())
    }

    /// Every local row is set on `stream`: send what is left, copy the
    /// peer's region out of the stage, and make `stream` wait for it all.
    pub fn finish(self, stream: u64) -> Result<()> {
        self.finish_after(stream, |_| Ok(()))
    }

    /// [`Gather::finish`], running `before_join(stream)` once every piece is
    /// out and before `stream` waits for them: main-stream work on this
    /// rank's rows that overlaps the last piece's wire.
    pub fn finish_after(
        self,
        stream: u64,
        before_join: impl FnOnce(u64) -> Result<()>,
    ) -> Result<()> {
        let rows = self.pieces.borrow().rows;
        self.progress(rows, stream)?;
        let gpu = self.ctx.gpu;
        if let Some((src, dst, bytes)) = self.copy_out {
            gpu.copy_d2d_async(src, dst, bytes, self.side.stream)?;
        }
        before_join(stream)?;
        gpu.record_event(self.side.to_main, self.side.stream)?;
        gpu.stream_wait_event(stream, self.side.to_main)
    }
}

/// `ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE=1`.
pub fn rs_requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// Whether [`compute_and_reduce_scatter`] runs for this split (the switch,
/// qwen4_exp, no graph capture, and a pair that takes a piece).
pub fn rs_available(sp: SpRows, width: usize, ctx: &ForwardContext<'_>) -> bool {
    let plan = super::glm_sp_uneven::plan(sp, true);
    rs_requested()
        && ctx.config.model_type == "qwen4_exp"
        && !ctx.graph_capture
        && ctx
            .comm
            .is_some_and(|c| c.supports_exchange_async(plan.m.min(PIECE) * width * 2))
}

/// Compute a `[total, width]` BF16 partial with `rows(first, count, stream)`
/// (row-local: any row range, any order, same bytes) and reduce-scatter it,
/// with the window this rank sends computed FIRST and sent in slab-sized
/// pieces on the side stream while the rest of the rows are computed
/// (`ATLAS_QWEN4EXP_PREFILL_SP_RS_PIPE=1`). The peer's partial of this rank's
/// rows lands in the stage and is added after the join with the pair's own
/// add kernel (`bf16_add_inplace`, `own + peer`), so every byte is the one
/// `SpRows::reduce_scatter` leaves. `on_wire(stream)` runs after the rows,
/// before the join: main-stream work independent of the exchange.
/// `Ok(false)`: not taken, nothing computed or run.
pub fn compute_and_reduce_scatter(
    sp: SpRows,
    buf: DevicePtr,
    width: usize,
    ctx: &ForwardContext<'_>,
    stream: u64,
    mut rows: impl FnMut(usize, usize, u64) -> Result<()>,
    on_wire: impl FnOnce(u64) -> Result<()>,
) -> Result<bool> {
    if !rs_available(sp, width, ctx) {
        return Ok(false);
    }
    let comm = ctx.comm.expect("rs_available checked it");
    let row = width * 2;
    let plan = super::glm_sp_uneven::plan(sp, true);
    let gpu = ctx.gpu;
    let side = side(gpu)?;
    // The side stream starts after the main stream's previous exchange.
    gpu.record_event(side.to_side, stream)?;
    gpu.stream_wait_event(side.stream, side.to_side)?;
    let stage = stage(
        gpu,
        (plan.m * row).max(prealloc_rows(ctx) * row),
        stream,
        side.stream,
    )?;
    let total = sp.total();
    for w0 in (0..plan.m).step_by(PIECE) {
        let n = PIECE.min(plan.m - w0);
        rows(plan.send0 + w0, n, stream)?;
        gpu.record_event(side.to_side, stream)?;
        gpu.stream_wait_event(side.stream, side.to_side)?;
        ensure!(
            comm.exchange_async(
                buf.offset((plan.send0 + w0) * row).0,
                stage.offset(w0 * row).0,
                n * row,
                false,
                side.stream,
            )?,
            "qwen4exp SP pipe: pair exchange refused ({n} rows)"
        );
    }
    // The rows outside the window, while it is on the wire.
    if plan.send0 > 0 {
        rows(0, plan.send0, stream)?;
    }
    if plan.send0 + plan.m < total {
        rows(plan.send0 + plan.m, total - plan.send0 - plan.m, stream)?;
    }
    // Independent main-stream work that overlaps the wire too.
    on_wire(stream)?;
    gpu.record_event(side.to_main, side.stream)?;
    gpu.stream_wait_event(stream, side.to_main)?;
    let n = (plan.recv_n * width) as u32;
    spark_runtime::kernel_args::KernelLaunch::new(gpu, gpu.kernel("bf16_add", "bf16_add_inplace")?)
        .grid([n.div_ceil(256), 1, 1])
        .block([256, 1, 1])
        .arg_ptr(buf.offset(plan.recv0 * row))
        .arg_ptr(stage.offset(plan.skip * row))
        .arg_u32(n)
        .launch(stream)?;
    Ok(true)
}

/// The landing stage's address (tests: the mock GPU runs no add kernel).
#[cfg(test)]
pub(crate) fn stage_ptr() -> DevicePtr {
    DevicePtr(STAGE.with(Cell::get).0)
}

fn side(gpu: &dyn GpuBackend) -> Result<Side> {
    if let Some(s) = SIDE.with(Cell::get) {
        return Ok(s);
    }
    let s = Side {
        stream: gpu.create_stream()?,
        to_side: gpu.create_event()?,
        to_main: gpu.create_event()?,
    };
    SIDE.with(|c| c.set(Some(s)));
    Ok(s)
}

/// The landing stage, at least `bytes`, grown on demand once both streams
/// that may still use the old one drain.
fn stage(gpu: &dyn GpuBackend, bytes: usize, main: u64, side: u64) -> Result<DevicePtr> {
    let (ptr, size) = STAGE.with(Cell::get);
    if size >= bytes {
        return Ok(DevicePtr(ptr));
    }
    if ptr != 0 {
        gpu.synchronize(main)?;
        gpu.synchronize(side)?;
        gpu.free(DevicePtr(ptr))?;
        STAGE.with(|s| s.set((0, 0)));
    }
    let p = gpu.alloc(bytes)?;
    STAGE.with(|s| s.set((p.0, bytes)));
    Ok(p)
}

#[cfg(test)]
#[path = "qwen4exp_sp_pipe_tests.rs"]
mod tests;
