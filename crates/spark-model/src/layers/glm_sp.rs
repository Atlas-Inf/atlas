// SPDX-License-Identifier: AGPL-3.0-only
//! Sequence-parallel GLM prefill (`ATLAS_GLM_PREFILL_SP=1`; TP2/EP2, and
//! TP4/EP4 over the RDMA chain).
//!
//! Attention (TP over heads) and the MoE (EP) need every row of a chunk, but
//! the mHC seams, norms, shared expert and dense FFN are row-local and were
//! computed redundantly on every rank. Under SP each of the `n` ranks owns
//! `1/n` of the chunk's rows: the highway (`hc_streams`), `hidden` and the
//! seam scratch hold only the local rows, compacted at row 0. Each
//! all-reduce becomes a reduce-scatter into the local rows, and the normed
//! block inputs are all-gathered before attention and the MoE. Rank `r` owns
//! the `n - 1 - r`th part, so rank 0 owns the top rows and the tail rows
//! DFlash captures (rank 0 only) are always local to it.
//!
//! On the pair both exchanges move only the rows the peer owns. On a chain
//! the reduce-scatter is the chain's (`CommBackend::reduce_scatter_async`,
//! three quarters of the rows over each link; else its all-reduce of every
//! row), whose local rows are the all-reduce's own words, and the gather the
//! chain's segment all-gather (`CommBackend::gather_async`).
//!
//! All ops keep their per-row arithmetic, so outputs are bit-identical.
//!
//! qwen4_exp (`ATLAS_QWEN4EXP_PREFILL_SP=1`, `model::qwen4exp_prefill_sp`)
//! splits UNEVENLY, at a multiple of the 2048-row mHC slab, so that each
//! rank's collapses run the same slabs as the unsplit chunk (the slab height
//! picks the GEMM, and so the bits); [`SpRows::split_at`]. Its exchanges move
//! `max(rows, peer_rows)` rows each way and land the peer's in a staging
//! buffer where the two halves differ (`glm_sp_uneven`).

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use std::cell::Cell;

use crate::layer::ForwardContext;

/// This rank's rows of the current chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpRows {
    /// First local row in chunk order.
    pub row0: usize,
    /// Local rows (half the chunk).
    pub rows: usize,
    /// First row the peer owns.
    pub peer0: usize,
    /// Rows the peer owns (`rows` for an even split).
    pub peer_rows: usize,
    /// The shared expert runs every row and only its blend the local ones,
    /// instead of running the local rows only: its GEMMs' kernel choice may
    /// depend on the row count, so a split that must keep every byte keeps
    /// its row count (qwen4_exp).
    pub full_shared: bool,
    /// Ranks the rows are split over. Beyond two, `peer0` is 0 and
    /// `peer_rows` counts every row the other ranks own.
    pub ranks: usize,
}

impl SpRows {
    /// Rank `rank` of `ranks` owns part `ranks - 1 - rank` of `total` rows
    /// (on the pair, rank 0 the upper half); `total` must be a multiple of
    /// `ranks`.
    pub fn for_rank(total: usize, rank: usize, ranks: usize) -> Self {
        let rows = total / ranks;
        let row0 = (ranks - 1 - rank) * rows;
        let peer0 = if ranks == 2 { rows - row0 } else { 0 };
        Self {
            row0,
            rows,
            peer0,
            peer_rows: total - rows,
            full_shared: false,
            ranks,
        }
    }

    /// Split `total` rows at `split`: rank 0 owns `[0, split)`, rank 1
    /// `[split, total)` (qwen4_exp: rank 0, the only drafting rank, keeps
    /// chunk row 0 at highway row 0). The shared expert keeps every row.
    pub fn split_at(total: usize, split: usize, rank: usize) -> Self {
        let (lo, hi) = (split, total - split);
        let (row0, rows, peer0, peer_rows) = if rank == 0 {
            (0, lo, split, hi)
        } else {
            (split, hi, 0, lo)
        };
        Self {
            row0,
            rows,
            peer0,
            peer_rows,
            full_shared: true,
            ranks: 2,
        }
    }

    /// Rows of the whole chunk.
    pub fn total(self) -> usize {
        self.rows + self.peer_rows
    }

    /// `ptr` advanced to the local rows of a `[total, width]` BF16 tensor.
    pub fn local(self, ptr: DevicePtr, width: usize) -> DevicePtr {
        ptr.offset(self.row0 * width * 2)
    }

    /// Reduce-scatter a `[total, width]` BF16 partial: the local rows end
    /// up holding the sum of both ranks' partials.
    pub fn reduce_scatter(
        self,
        ptr: DevicePtr,
        width: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.ranks > 2 {
            let comm = ctx.comm.expect("GLM chain reduce without a communicator");
            // A pipelined half marks its layer's reduces (`pipe_fences`).
            if let Some(event) = next_fence(&REDUCE_MARKS) {
                ctx.gpu.record_event(event, stream)?;
            }
            if !comm.supports_reduce_scatter_async(self.rows * width * 2) {
                return comm.all_reduce_async(ptr.0, self.total() * width * 2, stream);
            }
            return self.reduce_scatter_block(ptr, width, (0, self.rows), ctx, stream);
        }
        if self.rows != self.peer_rows {
            return super::glm_sp_uneven::exchange(self, ptr, width, true, ctx, stream);
        }
        exchange_rows(
            ptr.offset(self.peer0 * width * 2),
            self.local(ptr, width),
            self.rows,
            width * 2,
            true,
            ctx,
            stream,
        )
    }

    /// Reduce-scatter local rows `[r0, r0 + n)` (and every rank's same rows)
    /// of a `[total, width]` BF16 partial, on an even split: a block of
    /// [`Self::reduce_scatter`] (`layers::glm_sp_overlap`). On a chain it
    /// needs `CommBackend::reduce_scatter_async`.
    pub fn reduce_scatter_block(
        self,
        ptr: DevicePtr,
        width: usize,
        block: (usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.exchange_block(ptr, width, block, true, ctx, stream)
    }

    /// All-gather local rows `[r0, r0 + n)` (every rank's same rows) of a
    /// `[total, width]` BF16 tensor, on an even split.
    pub fn all_gather_block(
        self,
        ptr: DevicePtr,
        width: usize,
        block: (usize, usize),
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.exchange_block(ptr, width, block, false, ctx, stream)
    }

    /// A chain collective's segments for local rows `[r0, r0 + n)` of every
    /// part of a `[total, width]` BF16 tensor: rank 0's segment, the stride
    /// to rank `d`'s, and the bytes of each.
    pub fn chain_segments(
        self,
        ptr: DevicePtr,
        width: usize,
        (r0, n): (usize, usize),
    ) -> (u64, i64, usize) {
        let row = width * 2;
        // Rank 0's part is the top one, rank r's `r` parts below it.
        let rank0 = ptr.offset(((self.ranks - 1) * self.rows + r0) * row);
        (rank0.0, -((self.rows * row) as i64), n * row)
    }

    fn exchange_block(
        self,
        ptr: DevicePtr,
        width: usize,
        (r0, n): (usize, usize),
        reduce: bool,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let row = width * 2;
        if self.ranks == 2 {
            let (peer, local) = (
                ptr.offset((self.peer0 + r0) * row),
                ptr.offset((self.row0 + r0) * row),
            );
            let (send, dst) = if reduce { (peer, local) } else { (local, peer) };
            return exchange_rows(send, dst, n, row, reduce, ctx, stream);
        }
        let comm = ctx.comm.expect("GLM chain exchange without a communicator");
        let (rank0, stride, bytes) = self.chain_segments(ptr, width, (r0, n));
        let done = if reduce {
            comm.reduce_scatter_async(rank0, stride, bytes, stream)?
        } else {
            comm.gather_async(rank0, stride, bytes, stream)?
        };
        ensure!(
            done,
            "GLM chain {} refused ({n} rows x {width})",
            if reduce { "reduce-scatter" } else { "gather" }
        );
        Ok(())
    }

    /// All-gather a `[total, width]` BF16 tensor whose local rows are set.
    pub fn all_gather(
        self,
        ptr: DevicePtr,
        width: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.ranks > 2 {
            if let Some(event) = next_fence(&GATHER_WAITS) {
                ctx.gpu.stream_wait_event(stream, event)?;
            }
            return self.all_gather_block(ptr, width, (0, self.rows), ctx, stream);
        }
        if self.rows != self.peer_rows {
            return super::glm_sp_uneven::exchange(self, ptr, width, false, ctx, stream);
        }
        exchange_rows(
            self.local(ptr, width),
            ptr.offset(self.peer0 * width * 2),
            self.rows,
            width * 2,
            false,
            ctx,
            stream,
        )
    }
}

/// Two-rank copy-engine exchange of `rows` rows of `row_bytes`: send from
/// `send` and land the peer's equally sized rows in `dst`, added in place
/// (BF16) when `add`, else copied. Both ranks must call it in the same order
/// with the same size, after checking `CommBackend::supports_exchange_async`
/// from mirrored inputs, so a refusal is an error.
pub fn exchange_rows(
    send: DevicePtr,
    dst: DevicePtr,
    rows: usize,
    row_bytes: usize,
    add: bool,
    ctx: &ForwardContext,
    stream: u64,
) -> Result<()> {
    let comm = ctx.comm.expect("GLM pair exchange without a communicator");
    ensure!(
        comm.exchange_async(send.0, dst.0, rows * row_bytes, add, stream)?,
        "GLM pair exchange refused ({rows} rows x {row_bytes} bytes)"
    );
    Ok(())
}

thread_local! {
    static CURRENT: Cell<Option<SpRows>> = const { Cell::new(None) };
    static PIPELINED: Cell<bool> = const { Cell::new(false) };
    static REDUCE_MARKS: Cell<[Option<u64>; 2]> = const { Cell::new([None; 2]) };
    static GATHER_WAITS: Cell<[Option<u64>; 2]> = const { Cell::new([None; 2]) };
    static CAPTURE: Cell<Option<DevicePtr>> = const { Cell::new(None) };
    static WHOLE: Cell<bool> = const { Cell::new(false) };
}

/// An attention output that this rank's rows already hold whole (the MLA
/// o_proj after an all-to-all, `qwen3_attention::prefill::paged_glm_a2a`):
/// its seam reduces nothing. Taken once by the seam.
pub fn mark_attn_whole() {
    WHOLE.with(|w| w.set(true));
}

/// Whether the attention output just made is whole on this rank's rows
/// ([`mark_attn_whole`]); clears the mark.
pub fn take_attn_whole() -> bool {
    WHOLE.with(Cell::take)
}

type Fences = std::thread::LocalKey<Cell<[Option<u64>; 2]>>;

/// The next of a layer's fences, consumed (the slot of the next reduce or
/// gather of the layer).
fn next_fence(fences: &'static Fences) -> Option<u64> {
    fences.with(|c| {
        let [first, second] = c.get();
        c.set([second, None]);
        first
    })
}

/// The two-half pipeline's fences for this thread's next layer
/// (`model::glm_prefill_pipe`): `marks` are recorded on the stream of the
/// layer's first and second reduce-scatter, before each (a layer's first is
/// its attention output's, its second its MoE output's, so they mark the
/// attention, KDA state and K/V, and the MoE done); `waits` are waited on
/// before its first and second all-gather (the second gathers the MoE
/// input). [`take_pipe_fences`] hands back the marks no reduce took.
pub fn pipe_fences(marks: [Option<u64>; 2], waits: [Option<u64>; 2]) {
    REDUCE_MARKS.with(|c| c.set(marks));
    GATHER_WAITS.with(|c| c.set(waits));
}

/// The marks of [`pipe_fences`] no reduce took, all cleared.
pub fn take_pipe_fences() -> [Option<u64>; 2] {
    GATHER_WAITS.with(|c| c.set([None; 2]));
    REDUCE_MARKS.with(|c| c.replace([None; 2]))
}

/// Whether this thread is issuing a two-half pipelined chunk
/// (`model::glm_prefill_pipe`): its seams run whole (the halves overlap
/// instead), so no seam takes the side lane.
pub fn pipelined() -> bool {
    PIPELINED.with(Cell::get)
}

/// Clears [`pipelined`] when the pipelined chunk ends (or unwinds).
pub struct PipelineScope(());

impl Drop for PipelineScope {
    fn drop(&mut self) {
        PIPELINED.with(|c| c.set(false));
    }
}

/// Mark this thread's chunk pipelined until the scope drops.
pub fn enter_pipelined() -> PipelineScope {
    PIPELINED.with(|c| c.set(true));
    PipelineScope(())
}

/// The whole chunk's contracted `[total, hidden]` rows of the current
/// DFlash capture layer, gathered over a chain (`model::glm_prefill_sp`):
/// the capture reads them there instead of this rank's highway rows.
pub fn capture_source() -> Option<DevicePtr> {
    CAPTURE.with(Cell::get)
}

/// Clears [`capture_source`] when the layer's capture ends (or unwinds).
pub struct CaptureScope(());

impl Drop for CaptureScope {
    fn drop(&mut self) {
        CAPTURE.with(|c| c.set(None));
    }
}

/// Let this layer's DFlash capture read the gathered rows at `rows`.
pub fn capture_from(rows: DevicePtr) -> CaptureScope {
    CAPTURE.with(|c| c.set(Some(rows)));
    CaptureScope(())
}

/// The SP rows of the prefill chunk this thread is running, if any.
pub fn current() -> Option<SpRows> {
    CURRENT.with(Cell::get)
}

/// Clears the SP rows when the chunk's layer loop ends (or unwinds).
pub struct Scope(());

impl Drop for Scope {
    fn drop(&mut self) {
        CURRENT.with(|c| c.set(None));
    }
}

/// Run the rest of this chunk sequence-parallel over `rows`.
pub fn enter(rows: SpRows) -> Scope {
    CURRENT.with(|c| c.set(Some(rows)));
    Scope(())
}

/// Whether `ATLAS_GLM_PREFILL_SP=1`.
pub fn requested() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PREFILL_SP").as_deref() == Ok("1"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank0_owns_the_upper_half() {
        assert_eq!(
            SpRows::for_rank(8196, 0, 2),
            SpRows {
                row0: 4098,
                rows: 4098,
                peer0: 0,
                peer_rows: 4098,
                full_shared: false,
                ranks: 2,
            }
        );
        assert_eq!(
            SpRows::for_rank(8196, 1, 2),
            SpRows {
                row0: 0,
                rows: 4098,
                peer0: 4098,
                peer_rows: 4098,
                full_shared: false,
                ranks: 2,
            }
        );
        let r = SpRows::for_rank(8, 0, 2);
        assert_eq!(
            r.local(DevicePtr(0x1000), 4096),
            DevicePtr(0x1000 + 4 * 4096 * 2)
        );
    }

    #[test]
    fn chain_ranks_own_quarters_top_down() {
        let parts: Vec<(usize, usize)> = (0..4)
            .map(|r| SpRows::for_rank(8192, r, 4))
            .map(|sp| {
                assert_eq!((sp.total(), sp.peer0, sp.ranks), (8192, 0, 4));
                (sp.row0, sp.rows)
            })
            .collect();
        assert_eq!(parts, [(6144, 2048), (4096, 2048), (2048, 2048), (0, 2048)]);
    }

    #[test]
    fn split_at_gives_rank0_the_lower_rows() {
        let lo = SpRows::split_at(16046, 8192, 0);
        let hi = SpRows::split_at(16046, 8192, 1);
        assert_eq!(
            (lo.row0, lo.rows, lo.peer0, lo.peer_rows),
            (0, 8192, 8192, 7854)
        );
        assert_eq!(
            (hi.row0, hi.rows, hi.peer0, hi.peer_rows),
            (8192, 7854, 0, 8192)
        );
        assert_eq!((lo.total(), hi.total()), (16046, 16046));
        assert!(lo.full_shared && hi.full_shared);
    }

    #[test]
    fn scope_clears_on_drop() {
        assert_eq!(current(), None);
        {
            let _s = enter(SpRows::for_rank(16, 1, 2));
            assert_eq!(current().map(|r| r.rows), Some(8));
        }
        assert_eq!(current(), None);
    }
}
