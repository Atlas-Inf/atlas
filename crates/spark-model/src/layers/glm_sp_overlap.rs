// SPDX-License-Identifier: AGPL-3.0-only
//! `ATLAS_GLM_SP_GATHER_OVERLAP=1` (or `=N` blocks, 2..=8; `1` = 4): the
//! sequence-parallel exchanges of a GLM prefill seam (`layers::glm_sp`)
//! overlap the seam's own compute.
//!
//! Each seam reduce-scatters the finished block's output into this rank's
//! rows, runs the row-local mHC post / pre / norm over them, then
//! all-gathers the normed rows: link time and compute in series. Here this
//! rank's rows are cut into row blocks. The exchanges run block by block on a
//! side stream (every reduce-scatter first, then each block's all-gather as
//! soon as that block is normed), and the compute stream runs block k once
//! its reduced rows landed, while later blocks are still in flight.
//!
//! Exact: the exchanges move and add the same rows as the whole ones (BF16
//! adds are elementwise), and the block compute runs the whole call's
//! kernels on a row range (the HC site eligibility check),
//! each computing a row on its own. Both ranks cut the same blocks, so they
//! issue the same exchanges in the same order; all of a seam's exchanges run
//! on the one side stream, fenced against the compute stream before the
//! first and after the last, so the pair's exchanges stay ordered on the GPU.
//! On the four-rank chain a block's exchanges are the chain's reduce-scatter
//! and gather of that block of every rank's part (`SpRows::*_block`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm_sp::SpRows;
use crate::layer::ForwardContext;

/// Smallest block: the seam's prefill form needs 512 rows, and the block
/// compute must take the whole call's kernels.
pub const MIN_BLOCK_ROWS: usize = 512;
/// Most blocks a seam cuts (one reduce event each).
const MAX_BLOCKS: usize = 8;

fn parse_blocks(value: Option<&str>) -> Result<usize> {
    match value {
        None | Some("0") => Ok(0),
        Some("1") => Ok(4),
        Some(v) => {
            let n: usize = v
                .parse()
                .ok()
                .filter(|n| (2..=MAX_BLOCKS).contains(n))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "ATLAS_GLM_SP_GATHER_OVERLAP must be 0, 1 or 2..={MAX_BLOCKS}, got {v:?}"
                    )
                })?;
            Ok(n)
        }
    }
}

pub(crate) fn requested_blocks() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        parse_blocks(std::env::var("ATLAS_GLM_SP_GATHER_OVERLAP").ok().as_deref())
            .unwrap_or_else(|e| panic!("{e:#}"))
    })
}

/// `n` near-equal blocks `(first local row, rows)` of `rows` rows, or none
/// when a block would fall under [`MIN_BLOCK_ROWS`].
fn cut(rows: usize, n: usize) -> Option<Vec<(usize, usize)>> {
    if n < 2 {
        return None;
    }
    let size = rows.div_ceil(n);
    let blocks: Vec<_> = (0..rows)
        .step_by(size)
        .map(|r0| (r0, size.min(rows - r0)))
        .collect();
    blocks
        .iter()
        .all(|&(_, r)| r >= MIN_BLOCK_ROWS)
        .then_some(blocks)
}

/// This rank's row blocks for an overlapped seam, when requested and the
/// split is even (every rank then cuts the same blocks) and, on a chain, its
/// reduce-scatter takes the blocks. Never under graph capture or the
/// determinism tracer, whose taps sit between the ops a block pipeline
/// interleaves.
pub fn blocks(sp: SpRows, ctx: &ForwardContext) -> Option<Vec<(usize, usize)>> {
    cut(sp.rows, requested_blocks()).filter(|b| eligible(sp, b, ctx))
}

/// [`blocks`], or, for a seam reduced ahead ([`reduce_ahead`]) when overlap
/// is requested but the rows are too few to cut, one block of every local
/// row: the reduce still runs beside the compute after it (a fused one
/// beside its own producer), and the next seam, which cannot cut blocks
/// either, joins it whole.
pub fn blocks_or_whole(sp: SpRows, ctx: &ForwardContext) -> Option<Vec<(usize, usize)>> {
    blocks(sp, ctx).or_else(|| {
        let whole = vec![(0, sp.rows)];
        (requested_blocks() > 0 && eligible(sp, &whole, ctx)).then_some(whole)
    })
}

fn eligible(sp: SpRows, blocks: &[(usize, usize)], ctx: &ForwardContext) -> bool {
    let even = sp.rows * (sp.ranks - 1) == sp.peer_rows;
    if !even || ctx.graph_capture || crate::det_trace::on() || super::glm_sp::pipelined() {
        return false;
    }
    let row = ctx.config.hidden_size * 2;
    let chain_ok = |b: &(usize, usize)| {
        ctx.comm.is_some_and(|c| {
            c.supports_reduce_scatter_async(b.1 * row) && c.supports_gather_async(b.1 * row)
        })
    };
    sp.ranks == 2 || blocks.iter().all(chain_ok)
}

/// The side stream and its fences, made once per process (one device).
struct Lane {
    stream: u64,
    begun: u64,
    computed: u64,
    landed: u64,
    reduced: [u64; MAX_BLOCKS],
}

fn lane(gpu: &dyn GpuBackend) -> Result<&'static Lane> {
    static LANE: std::sync::OnceLock<Lane> = std::sync::OnceLock::new();
    if let Some(lane) = LANE.get() {
        return Ok(lane);
    }
    let mut reduced = [0; MAX_BLOCKS];
    for e in &mut reduced {
        *e = gpu.create_event()?;
    }
    let lane = Lane {
        stream: gpu.create_stream()?,
        begun: gpu.create_event()?,
        computed: gpu.create_event()?,
        landed: gpu.create_event()?,
        reduced,
    };
    Ok(LANE.get_or_init(|| lane))
}

/// A seam whose reduce-scatter [`reduce_ahead`] already issued: the next
/// [`pipelined`] over the same buffer and blocks runs the rest of it.
#[derive(Clone, Copy)]
struct Ahead {
    reduce: DevicePtr,
    blocks: [(usize, usize); MAX_BLOCKS],
    len: usize,
    /// One fused reduce-scatter ([`reduce_ahead_fused`]) in place of a
    /// reduce-scatter a block: a block's rows are final once its pieces are.
    fused: Option<spark_comm::RsFused>,
}

impl Ahead {
    fn new(
        reduce: DevicePtr,
        blocks: &[(usize, usize)],
        fused: Option<spark_comm::RsFused>,
    ) -> Self {
        let mut kept = [(0, 0); MAX_BLOCKS];
        kept[..blocks.len()].copy_from_slice(blocks);
        Self {
            reduce,
            blocks: kept,
            len: blocks.len(),
            fused,
        }
    }
}

thread_local! {
    static AHEAD: std::cell::Cell<Option<Ahead>> = const { std::cell::Cell::new(None) };
}

/// Issue a seam's reduce-scatter of `reduce` now, a block at a time right
/// after `produce(block)` wrote that block of every rank's part on the
/// compute stream, and leave the rest of the seam to the next [`pipelined`]
/// over the same `reduce` and `blocks` (or to [`join_ahead`]). The compute
/// stream reads none of the reduced rows until then.
#[allow(clippy::too_many_arguments)]
pub fn reduce_ahead(
    sp: SpRows,
    blocks: &[(usize, usize)],
    reduce: DevicePtr,
    width: usize,
    ctx: &ForwardContext,
    stream: u64,
    mut produce: impl FnMut((usize, usize)) -> Result<()>,
) -> Result<()> {
    ensure!(
        (1..=MAX_BLOCKS).contains(&blocks.len())
            && sp.rows * (sp.ranks - 1) == sp.peer_rows
            && AHEAD.with(|a| a.get().is_none()),
        "a seam reduced ahead needs 1..={MAX_BLOCKS} blocks of an even split, one at a time"
    );
    let (gpu, lane) = (ctx.gpu, lane(ctx.gpu)?);
    gpu.record_event(lane.begun, stream)?;
    gpu.stream_wait_event(lane.stream, lane.begun)?;
    let mut run = || -> Result<()> {
        for (&block, &done) in blocks.iter().zip(&lane.reduced) {
            produce(block)?;
            gpu.record_event(lane.computed, stream)?;
            gpu.stream_wait_event(lane.stream, lane.computed)?;
            sp.reduce_scatter_block(reduce, width, block, ctx, lane.stream)?;
            gpu.record_event(done, lane.stream)?;
        }
        Ok(())
    };
    if let Err(e) = run() {
        gpu.record_event(lane.landed, lane.stream)?;
        gpu.stream_wait_event(stream, lane.landed)?;
        return Err(e);
    }
    AHEAD.with(|a| a.set(Some(Ahead::new(reduce, blocks, None))));
    Ok(())
}

/// [`reduce_ahead`] as one fused chain reduce-scatter (`fused`, from
/// `CommBackend::rs_fused_begin` over all of `reduce`'s rows): `produce`
/// launches the producer kernel, which writes and publishes every piece of
/// the op, on the compute stream; the op's chain kernel runs beside it on the
/// side stream from the same point. The next [`pipelined`] over `reduce` and
/// `blocks` runs each block once its own pieces are final.
#[allow(clippy::too_many_arguments)]
pub fn reduce_ahead_fused(
    sp: SpRows,
    blocks: &[(usize, usize)],
    reduce: DevicePtr,
    fused: spark_comm::RsFused,
    ctx: &ForwardContext,
    stream: u64,
    produce: impl FnOnce() -> Result<()>,
) -> Result<()> {
    ensure!(
        (1..=MAX_BLOCKS).contains(&blocks.len())
            && sp.rows * (sp.ranks - 1) == sp.peer_rows
            && AHEAD.with(|a| a.get().is_none()),
        "a seam reduced ahead needs 1..={MAX_BLOCKS} blocks of an even split, one at a time"
    );
    let comm = ctx
        .comm
        .expect("fused reduce-scatter without a communicator");
    let (gpu, lane) = (ctx.gpu, lane(ctx.gpu)?);
    gpu.record_event(lane.begun, stream)?;
    gpu.stream_wait_event(lane.stream, lane.begun)?;
    let run = || -> Result<()> {
        produce()?;
        ensure!(
            comm.reduce_scatter_fused_async(&fused, lane.stream)?,
            "fused reduce-scatter refused after its begin"
        );
        Ok(())
    };
    if let Err(e) = run() {
        gpu.record_event(lane.landed, lane.stream)?;
        gpu.stream_wait_event(stream, lane.landed)?;
        return Err(e);
    }
    AHEAD.with(|a| a.set(Some(Ahead::new(reduce, blocks, Some(fused)))));
    Ok(())
}

/// Make `stream` wait until local rows `[r0, r0 + rows)` of a fused
/// reduce-scatter are final: every piece of this rank's segment they touch.
fn wait_fused_rows(
    fused: &spark_comm::RsFused,
    (r0, rows): (usize, usize),
    row_bytes: usize,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let comm = ctx
        .comm
        .expect("fused reduce-scatter without a communicator");
    for piece in pieces_of((r0, rows), fused.piece_bytes / row_bytes) {
        comm.rs_fused_wait(fused, piece, stream)?;
    }
    Ok(())
}

/// The pieces of `piece_rows` rows that local rows `[r0, r0 + rows)` touch.
fn pieces_of((r0, rows): (usize, usize), piece_rows: usize) -> std::ops::Range<usize> {
    r0 / piece_rows..(r0 + rows).div_ceil(piece_rows)
}

/// Whether a seam reduced ahead over `reduce` and `blocks` is pending.
pub fn ahead_matches(reduce: DevicePtr, blocks: &[(usize, usize)]) -> bool {
    AHEAD.with(|a| {
        a.get()
            .is_some_and(|a| a.reduce == reduce && a.blocks[..a.len] == *blocks)
    })
}

/// End a seam reduced ahead without its [`pipelined`] rest: the compute
/// stream waits until every block is reduced. No-op when none is pending.
pub fn join_ahead(ctx: &ForwardContext, stream: u64) -> Result<()> {
    if AHEAD.with(|a| a.take()).is_some() {
        let lane = lane(ctx.gpu)?;
        ctx.gpu.record_event(lane.landed, lane.stream)?;
        ctx.gpu.stream_wait_event(stream, lane.landed)?;
    }
    Ok(())
}

/// [`join_ahead`] of a seam reduced ahead over `reduce`, for a caller that
/// would otherwise reduce `reduce` itself: `true` when one was pending (its
/// reduce is done once the compute stream gets here), else `false` and
/// nothing enqueued.
pub fn join_ahead_of(reduce: DevicePtr, ctx: &ForwardContext, stream: u64) -> Result<bool> {
    if !AHEAD.with(|a| a.get().is_some_and(|a| a.reduce == reduce)) {
        return Ok(false);
    }
    join_ahead(ctx, stream)?;
    Ok(true)
}

/// One overlapped seam over `blocks` (from [`blocks`]): reduce-scatter
/// `reduce` (`[total, width]` BF16 partials) when given, run `site(row0,
/// rows)` per block once its local rows are reduced, and all-gather `gather`
/// (`(tensors, group)`: each tensor `(ptr, its width)`, `[total, width]`,
/// local rows written by `site`) every `group` blocks (fewer, larger
/// exchanges where each costs more than the compute beside it). `site` writes
/// only its block's rows of `gather` and reads only its block's rows of
/// `reduce`. The compute stream continues after every exchange landed. A
/// `reduce` that [`reduce_ahead`] already issued over these blocks is not
/// issued again: the blocks run once it lands, as here.
#[allow(clippy::too_many_arguments)]
pub fn pipelined(
    sp: SpRows,
    blocks: &[(usize, usize)],
    reduce: Option<DevicePtr>,
    gather: Option<(&[(DevicePtr, usize)], usize)>,
    width: usize,
    ctx: &ForwardContext,
    stream: u64,
    mut site: impl FnMut((usize, usize)) -> Result<()>,
) -> Result<()> {
    ensure!(
        (1..=MAX_BLOCKS).contains(&blocks.len()) && sp.rows * (sp.ranks - 1) == sp.peer_rows,
        "overlapped GLM SP seam needs 1..={MAX_BLOCKS} blocks of an even split"
    );
    let (gpu, lane) = (ctx.gpu, lane(ctx.gpu)?);
    let ahead = reduce.is_some_and(|ptr| ahead_matches(ptr, blocks));
    ensure!(
        ahead || AHEAD.with(|a| a.get().is_none()),
        "a seam reduced ahead is pending over another buffer or other blocks"
    );
    let fused = AHEAD
        .with(|a| a.get())
        .and_then(|a| a.fused)
        .filter(|_| ahead);
    if ahead {
        AHEAD.with(|a| a.set(None));
    } else {
        gpu.record_event(lane.begun, stream)?;
        gpu.stream_wait_event(lane.stream, lane.begun)?;
    }
    let mut run = || -> Result<()> {
        if let Some(ptr) = reduce.filter(|_| !ahead) {
            for (&block, &done) in blocks.iter().zip(&lane.reduced) {
                sp.reduce_scatter_block(ptr, width, block, ctx, lane.stream)?;
                gpu.record_event(done, lane.stream)?;
            }
        }
        let mut first = 0;
        for (i, (&(r0, rows), &done)) in blocks.iter().zip(&lane.reduced).enumerate() {
            match &fused {
                Some(f) => wait_fused_rows(f, (r0, rows), width * 2, ctx, stream)?,
                None if reduce.is_some() => gpu.stream_wait_event(stream, done)?,
                None => {}
            }
            site((r0, rows))?;
            if let Some((tensors, group)) = gather
                && ((i + 1) % group.max(1) == 0 || i + 1 == blocks.len())
            {
                gpu.record_event(lane.computed, stream)?;
                gpu.stream_wait_event(lane.stream, lane.computed)?;
                for &(ptr, width) in tensors {
                    sp.all_gather_block(ptr, width, (first, r0 + rows - first), ctx, lane.stream)?;
                }
                first = r0 + rows;
            }
        }
        Ok(())
    };
    let result = run();
    // Enqueued even after a failure, so the compute stream never runs ahead
    // of an exchange already in flight.
    gpu.record_event(lane.landed, lane.stream)?;
    gpu.stream_wait_event(stream, lane.landed)?;
    result
}

thread_local! {
    static BESIDE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Issue `exchanges(side stream)` on the side stream behind the compute
/// stream's work so far; the compute stream runs on until [`join_beside`].
/// Nothing may issue another exchange on the compute stream in between, so
/// the ranks' exchanges stay in one order on the GPU.
pub fn beside(
    ctx: &ForwardContext,
    stream: u64,
    exchanges: impl FnOnce(u64) -> Result<()>,
) -> Result<()> {
    ensure!(
        AHEAD.with(|a| a.get().is_none()),
        "a side-stream exchange while a seam reduced ahead is pending"
    );
    let (gpu, lane) = (ctx.gpu, lane(ctx.gpu)?);
    gpu.record_event(lane.begun, stream)?;
    gpu.stream_wait_event(lane.stream, lane.begun)?;
    BESIDE.with(|b| b.set(true));
    let result = exchanges(lane.stream);
    if result.is_err() {
        join_beside(ctx, stream)?;
    }
    result
}

/// [`beside`]: all-gather `tensors` (`(ptr, its width)`: `[total, width]`
/// BF16-width rows, this rank's rows set) whole.
pub fn gather_beside(
    sp: SpRows,
    tensors: &[(DevicePtr, usize)],
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    beside(ctx, stream, |lane| {
        tensors
            .iter()
            .try_for_each(|&(ptr, width)| sp.all_gather_block(ptr, width, (0, sp.rows), ctx, lane))
    })
}

/// The compute stream waits for every [`beside`] exchange so far; a no-op
/// when none is pending.
pub fn join_beside(ctx: &ForwardContext, stream: u64) -> Result<()> {
    if !BESIDE.with(|b| b.replace(false)) {
        return Ok(());
    }
    let lane = lane(ctx.gpu)?;
    ctx.gpu.record_event(lane.landed, lane.stream)?;
    ctx.gpu.stream_wait_event(stream, lane.landed)
}

#[cfg(test)]
#[path = "glm_sp_overlap_tests.rs"]
mod tests;
