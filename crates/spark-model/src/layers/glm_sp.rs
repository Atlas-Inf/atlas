// SPDX-License-Identifier: AGPL-3.0-only
//! Sequence-parallel GLM prefill (`ATLAS_GLM_PREFILL_SP=1`, TP2/EP2).
//!
//! Attention (TP over heads) and the MoE (EP) need every row of a chunk, but
//! the mHC seams, norms, shared expert and dense FFN are row-local and were
//! computed redundantly on both ranks. Under SP each rank owns half of the
//! chunk's rows: the highway (`hc_streams`), `hidden` and the seam scratch
//! hold only the local rows, compacted at row 0. Each all-reduce becomes a
//! reduce-scatter into the local rows, and the normed block inputs are
//! all-gathered before attention and the MoE. Rank 0 owns the upper half, so
//! the tail rows DFlash captures (rank 0 only) are always local to it.
//!
//! All ops keep their per-row arithmetic, so outputs are bit-identical.

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
}

impl SpRows {
    /// Rank 0 owns the upper half; `total` must be even.
    pub fn for_rank(total: usize, rank: usize) -> Self {
        let rows = total / 2;
        let (row0, peer0) = if rank == 0 { (rows, 0) } else { (0, rows) };
        Self { row0, rows, peer0 }
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

    /// All-gather a `[total, width]` BF16 tensor whose local rows are set.
    pub fn all_gather(
        self,
        ptr: DevicePtr,
        width: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
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
            SpRows::for_rank(8196, 0),
            SpRows {
                row0: 4098,
                rows: 4098,
                peer0: 0
            }
        );
        assert_eq!(
            SpRows::for_rank(8196, 1),
            SpRows {
                row0: 0,
                rows: 4098,
                peer0: 4098
            }
        );
        let r = SpRows::for_rank(8, 0);
        assert_eq!(
            r.local(DevicePtr(0x1000), 4096),
            DevicePtr(0x1000 + 4 * 4096 * 2)
        );
    }

    #[test]
    fn scope_clears_on_drop() {
        assert_eq!(current(), None);
        {
            let _s = enter(SpRows::for_rank(16, 1));
            assert_eq!(current().map(|r| r.rows), Some(8));
        }
        assert_eq!(current(), None);
    }
}
