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
    static TABLE: std::cell::RefCell<Option<Vec<CommitEntry>>> =
        const { std::cell::RefCell::new(None) };
}

/// One staged commit of [`QsaIndexer::commit_staged_rows`] for
/// `qsa_commit_table` (`ATLAS_QWEN4EXP_QSA_COMMIT_TABLE`): the pitched copy
/// and the block pool it would have launched, with their arguments.
#[derive(Clone, Copy, Debug)]
pub struct CommitEntry {
    pub kernel: KernelHandle,
    pub src: DevicePtr,
    pub dst: DevicePtr,
    pub raw_origin: DevicePtr,
    pub k_norm_w: DevicePtr,
    pub block_keys: DevicePtr,
    pub src_pitch: u32,
    pub count: u32,
    pub first_block: u32,
    pub n_new: u32,
    /// `(ratio, hd, rot, theta bits, eps bits)`: one launch needs them equal.
    pub shape: (u32, u32, u32, u32, u32),
}

/// Scope in which pitched staged commits are collected, not launched; the
/// model launches them as one table (`model/qsa_commit_table.rs`).
pub struct CommitTable;

impl CommitTable {
    pub fn enter() -> Self {
        TABLE.with(|t| *t.borrow_mut() = Some(Vec::new()));
        Self
    }

    /// The collected entries; collection ends.
    pub fn take(self) -> Vec<CommitEntry> {
        TABLE.with(|t| t.borrow_mut().take()).unwrap_or_default()
    }
}

impl Drop for CommitTable {
    fn drop(&mut self) {
        TABLE.with(|t| *t.borrow_mut() = None);
    }
}

fn collecting() -> bool {
    TABLE.with(|t| t.borrow().is_some())
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
        // ATLAS_QWEN4EXP_QSA_KEY_ONLY: the key rows only, straight into the
        // staging row, once a probe has shown the bytes equal (`qsa_key_only.rs`).
        if self.key_only(gpu, stream)? {
            return ops::cublas_bf16_proj_dense(
                normed,
                self.key_proj_w(),
                self.staged_key(row),
                1,
                self.hd,
                self.hidden,
                stream,
            )
            .context("QSA key projection (staged decode)");
        }
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

impl QsaIndexer {
    /// [`Self::commit_staged_row`] over `count` rows of one sequence,
    /// staging rows `first_row..` at positions `pos..`. With `pitched`
    /// (`ATLAS_QWEN4EXP_QSA_COMMIT_ROWS`), the raw keys land as ONE pitched
    /// copy (staging rows `qk_width` apart -> window rows `hd` apart, the
    /// window holding `pos..pos + count` contiguously after `raw_room`) and
    /// the blocks they complete pool in ONE launch; otherwise the per-row
    /// commit, verbatim. The bytes are the same: each raw key is the same
    /// copy, and `qsa_block_pool` computes each block from its own four raw
    /// keys whatever the launch's block count.
    #[allow(clippy::too_many_arguments)]
    pub fn commit_staged_rows(
        &self,
        st: &mut QsaSeqState,
        first_row: usize,
        pos: usize,
        count: usize,
        pitched: bool,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        if !pitched {
            for j in 0..count {
                self.commit_staged_row(st, first_row + j, pos + j, gpu, stream)?;
            }
            return Ok(());
        }
        if count == 0 {
            return Ok(());
        }
        let last = pos + count - 1;
        anyhow::ensure!(
            pos == st.ingested,
            "QSA staged commit at pos {pos} but {} tokens ingested",
            st.ingested
        );
        anyhow::ensure!(
            !self.is_active_at(last),
            "QSA staged commit through pos {last}: only inert rows stage (bound {})",
            self.inert_bound(),
        );
        anyhow::ensure!(
            first_row + count <= self.max_staged_rows(),
            "QSA staged commit: rows {first_row}..{} past the staging rows",
            first_row + count
        );
        self.reserve(st, pos + count, gpu, stream)?;
        self.raw_room(st, count, gpu, stream)?;
        if collecting() && self.k_commit_table_k.0 != 0 {
            let complete = (pos + count) / self.ratio as usize;
            let entry = CommitEntry {
                kernel: self.k_commit_table_k,
                src: self.staged_key(first_row),
                dst: self.raw_slot(st, pos),
                raw_origin: self.raw_origin(st),
                k_norm_w: self.k_norm_w,
                block_keys: st.block_keys,
                src_pitch: self.qk_width() as u32,
                count: count as u32,
                first_block: st.pooled as u32,
                n_new: complete.saturating_sub(st.pooled) as u32,
                shape: (
                    self.ratio,
                    self.hd,
                    self.rot,
                    self.theta.to_bits(),
                    self.eps.to_bits(),
                ),
            };
            TABLE.with(|t| t.borrow_mut().as_mut().map(|v| v.push(entry)));
            st.ingested = pos + count;
            st.pooled = st.pooled.max(complete);
            return Ok(());
        }
        let row = self.hd as usize * 2;
        crate::model::qwen4exp_step_copies::copy_rows(
            gpu,
            self.staged_key(first_row),
            self.qk_width() * 2,
            self.raw_slot(st, pos),
            row,
            row,
            count,
            true,
            stream,
        )?;
        st.ingested = pos + count;
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

#[cfg(test)]
#[path = "qsa_staged_rows_tests.rs"]
mod rows_tests;
