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

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::glm_sp::{SpRows, exchange_rows};
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
/// split is even (both ranks then cut the same blocks). Never under graph
/// capture or the determinism tracer, whose taps sit between the ops a
/// block pipeline interleaves.
pub fn blocks(sp: SpRows, ctx: &ForwardContext) -> Option<Vec<(usize, usize)>> {
    if sp.rows != sp.peer_rows || ctx.graph_capture || crate::det_trace::on() {
        return None;
    }
    cut(sp.rows, requested_blocks())
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

/// One overlapped seam over `blocks` (from [`blocks`]): reduce-scatter
/// `reduce` (`[total, width]` BF16 partials) when given, run `site(row0,
/// rows)` per block once its local rows are reduced, and all-gather `gather`
/// (`[total, width]`, local rows written by `site`) per block. `site` writes
/// only its block's rows of `gather` and reads only its block's rows of
/// `reduce`. The compute stream continues after every exchange landed.
#[allow(clippy::too_many_arguments)]
pub fn pipelined(
    sp: SpRows,
    blocks: &[(usize, usize)],
    reduce: Option<DevicePtr>,
    gather: Option<DevicePtr>,
    width: usize,
    ctx: &ForwardContext,
    stream: u64,
    mut site: impl FnMut((usize, usize)) -> Result<()>,
) -> Result<()> {
    ensure!(
        (1..=MAX_BLOCKS).contains(&blocks.len()) && sp.rows == sp.peer_rows,
        "overlapped GLM SP seam needs 1..={MAX_BLOCKS} blocks of an even split"
    );
    let (gpu, lane) = (ctx.gpu, lane(ctx.gpu)?);
    let row = width * 2;
    let at = |ptr: DevicePtr, first: usize, r0: usize| ptr.offset((first + r0) * row);
    gpu.record_event(lane.begun, stream)?;
    gpu.stream_wait_event(lane.stream, lane.begun)?;
    let mut run = || -> Result<()> {
        if let Some(ptr) = reduce {
            for (&(r0, rows), &done) in blocks.iter().zip(&lane.reduced) {
                let (peer, local) = (at(ptr, sp.peer0, r0), at(ptr, sp.row0, r0));
                exchange_rows(peer, local, rows, row, true, ctx, lane.stream)?;
                gpu.record_event(done, lane.stream)?;
            }
        }
        for (&(r0, rows), &done) in blocks.iter().zip(&lane.reduced) {
            if reduce.is_some() {
                gpu.stream_wait_event(stream, done)?;
            }
            site((r0, rows))?;
            if let Some(ptr) = gather {
                gpu.record_event(lane.computed, stream)?;
                gpu.stream_wait_event(lane.stream, lane.computed)?;
                let (local, peer) = (at(ptr, sp.row0, r0), at(ptr, sp.peer0, r0));
                exchange_rows(local, peer, rows, row, false, ctx, lane.stream)?;
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

#[cfg(test)]
#[path = "glm_sp_overlap_tests.rs"]
mod tests;
