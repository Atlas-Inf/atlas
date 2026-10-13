// SPDX-License-Identifier: AGPL-3.0-only

//! Staged decode ingest: the QSA indexer's per-step work split into a part a
//! CUDA graph can replay and a host part that runs after it
//! (`ATLAS_QWEN4EXP_DECODE_GRAPH_WIDE=1`, `model/decode_pieces.rs`).
//!
//! Inside the inert bound `decode_select` selects nothing: it projects the
//! row's qk, copies the raw key to `raw_keys[pos]`, advances the host
//! counters and, once a 4-token block completes, pools it. Only the
//! projection is a function of the step's activations. The copy destination
//! (a per-SEQUENCE buffer at a per-POSITION offset), the counters and the
//! every-fourth-token pool launch are not a function of anything a graph key
//! holds, which is why an indexer vetoes capture.
//!
//! Staged, a row's ingest is two halves:
//!
//! * [`QsaIndexer::stage_row`] (in the graph): the SAME `[1, hidden]` qk
//!   projection into the SAME scratch row 0 `decode_select` uses, then a
//!   copy of the raw-key columns into staging row `1 + row` of the
//!   layer-owned `qk_scratch` (fixed addresses only).
//! * [`QsaIndexer::commit_staged_row`] (eager, after the run): staging row
//!   -> `raw_keys[pos]`, the counters, the block pool -- `decode_select`'s
//!   ingest tail verbatim.
//!
//! The raw keys are byte copies of the same projection output, so the
//! indexer state after a staged step is the eager step's, and the first
//! ACTIVE `decode_select` finds `pos == ingested` as always. Staging rows
//! `1..` of `qk_scratch` are otherwise used only by prefill ingest, which
//! never runs between a step's forward and its commit.
//!
//! The mode is a scoped thread flag ([`StagedIngest`]): the model sets it for
//! exactly the layer runs it captures, and the layers that consult it
//! (`attention_forward`, the multi-row ingest) run on the calling thread.

use std::cell::Cell;

use super::*;

thread_local! {
    static STAGED: Cell<bool> = const { Cell::new(false) };
}

/// Whether the indexer layers called on this thread stage their ingest.
pub fn staged_ingest() -> bool {
    STAGED.with(Cell::get)
}

/// Scope of a staged run; restores the previous mode on drop.
pub struct StagedIngest(bool);

impl StagedIngest {
    pub fn enter(on: bool) -> Self {
        Self(STAGED.with(|s| s.replace(on)))
    }
}

impl Drop for StagedIngest {
    fn drop(&mut self) {
        STAGED.with(|s| s.set(self.0));
    }
}

impl QsaIndexer {
    /// Staging rows a step may use (`qk_scratch` minus the projection row).
    pub fn max_staged_rows(&self) -> usize {
        INGEST_SLAB - 1
    }

    fn staged_key(&self, row: usize) -> DevicePtr {
        self.qk_scratch.offset((1 + row) * self.qk_width() * 2)
    }

    /// The graph half of row `row`'s ingest: `normed` (`[1, hidden]` BF16)
    /// projected exactly as `decode_select` projects it, raw key parked in
    /// staging row `row`.
    pub fn stage_row(
        &self,
        normed: DevicePtr,
        row: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            row < self.max_staged_rows(),
            "QSA staged ingest: row {row} past the {} staging rows",
            self.max_staged_rows()
        );
        let hd = self.hd as usize;
        ops::cublas_bf16_proj_dense(
            normed,
            self.qk_proj_w,
            self.qk_scratch,
            1,
            self.qk_width() as u32,
            self.hidden,
            stream,
        )
        .context("QSA qk projection (staged decode)")?;
        gpu.copy_d2d_async(
            self.qk_scratch.offset(self.n_heads as usize * hd * 2),
            self.staged_key(row),
            hd * 2,
            stream,
        )
    }

    /// The host half: staging row `row` becomes the raw key at `pos`, as
    /// `decode_select`'s ingest leaves it. Inert positions only -- an active
    /// row selects, which no staged step does.
    pub fn commit_staged_row(
        &self,
        st: &mut QsaSeqState,
        row: usize,
        pos: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            pos == st.ingested,
            "QSA staged commit at pos {pos} but {} tokens ingested",
            st.ingested
        );
        anyhow::ensure!(
            !self.is_active_at(pos),
            "QSA staged commit at pos {pos}: only inert rows stage (bound {})",
            self.inert_bound(),
        );
        anyhow::ensure!(
            row < self.max_staged_rows(),
            "QSA staged commit: row {row} past the staging rows"
        );
        self.reserve(st, pos + 1, gpu, stream)?;
        self.raw_room(st, 1, gpu, stream)?;
        let hd = self.hd as usize;
        gpu.copy_d2d_async(self.staged_key(row), self.raw_slot(st, pos), hd * 2, stream)?;
        st.ingested = pos + 1;
        self.pool_new_blocks(st, gpu, stream)
    }
}

#[cfg(test)]
mod tests {
    use super::{StagedIngest, staged_ingest};

    #[test]
    fn scope_sets_and_restores() {
        assert!(!staged_ingest());
        {
            let _outer = StagedIngest::enter(true);
            assert!(staged_ingest());
            {
                let _inner = StagedIngest::enter(false);
                assert!(!staged_ingest());
            }
            assert!(staged_ingest(), "the inner scope restores the outer mode");
        }
        assert!(!staged_ingest());
    }
}
