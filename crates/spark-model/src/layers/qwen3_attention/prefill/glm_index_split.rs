// SPDX-License-Identifier: AGPL-3.0-only

//! Query-row split of the prefill semantic selection across the TP pair
//! (`ATLAS_GLM_INDEX_SPLIT=1`).
//!
//! Both ranks hold the same pooled index keys and project the same queries.
//! Each row's logits and top-k are independent of the other rows in its
//! launch: a row group only decides which key tiles are wholly future, and
//! those score -inf for each of its rows either way. So each rank selects a
//! subset of an owner's rows and swaps the finished token-id rows with its
//! peer, and the result is bit-identical to the replicated selection.
//!
//! A row's work grows with its causal extent, so the rows are cut into
//! zigzag quarters: rank 0 selects Q0 and Q3, rank 1 selects Q1 and Q2
//! (equal work), then the equal-sized pairs Q0<->Q1 and Q3<->Q2 are
//! exchanged on the copy-engine pair. Both ranks select the 0-3 rows past
//! the last whole quarter. The projections and the index-cache update stay
//! replicated. Every eligibility input is mirrored on both ranks, since one
//! rank splitting alone would deadlock the pair; the settings themselves
//! are compared across the ranks at startup (`model::startup_parity`).
//!
//! `ATLAS_GLM_INDEX_SPLIT_CHECK=1` also selects every row into scratch and
//! fails the request on both ranks on any difference (`check`).
//!
//! Prior art: splitting the prefill indexer's rows across ranks (with a
//! re-select-every-row check) follows RiNGSiDE's `GLM53_INDEXER_ROW_SPLIT`
//! (othexmr, <https://github.com/othexmr/GLM-5.3-Flash-NVFP4-2x-4x-DGX-Sparks-RiNGSiDE>
//! `src/tp4/glm53_indexer_rowsplit.py`; Apache-2.0), itself after rhys101's
//! SG18 prefill TP split and knapcio's DeepSeek-V4.1 TP4 adaptation. Ideas
//! only, no code. The zigzag quarters and the pair swap are ours
//! (docs/glm-prior-art.md).

use std::ops::Range;

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use crate::layer::ForwardContext;
use crate::layers::glm_sp;

#[path = "glm_index_split_check.rs"]
mod check;

/// Owners below this many rows stay replicated (verify, short appends): the
/// two exchanges' fixed cost outweighs the halved selection.
const MIN_ROWS: usize = 256;

/// `ATLAS_GLM_INDEX_SPLIT_MIN_CTX` default: the history per row at which
/// the halved selection starts to outweigh the row's exchange.
const DEFAULT_MIN_CTX: usize = 4096;

/// This process's split settings, when `ATLAS_GLM_INDEX_SPLIT=1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Settings {
    /// History (`seq_len_start`) an owner needs to split,
    /// `ATLAS_GLM_INDEX_SPLIT_MIN_CTX`.
    min_ctx: usize,
    /// `ATLAS_GLM_INDEX_SPLIT_CHECK=1`.
    check: bool,
}

impl Settings {
    /// What the ranks must agree on: `[on, min_ctx, check]`.
    fn words(settings: Option<Self>) -> [u64; 3] {
        settings.map_or([0; 3], |s| [1, s.min_ctx as u64, s.check as u64])
    }
}

/// The settings `var` gives, or why its `ATLAS_GLM_INDEX_SPLIT_MIN_CTX` is
/// junk (ignored while the split is off).
fn parse_settings(var: impl Fn(&str) -> Option<String>) -> Result<Option<Settings>, String> {
    let on = |name| var(name).as_deref() == Some("1");
    if !on("ATLAS_GLM_INDEX_SPLIT") {
        return Ok(None);
    }
    let min_ctx = match var("ATLAS_GLM_INDEX_SPLIT_MIN_CTX") {
        None => DEFAULT_MIN_CTX,
        Some(v) => v.parse().map_err(|_| {
            format!("ATLAS_GLM_INDEX_SPLIT_MIN_CTX must be a token count, got {v:?}")
        })?,
    };
    Ok(Some(Settings {
        min_ctx,
        check: on("ATLAS_GLM_INDEX_SPLIT_CHECK"),
    }))
}

/// This process's settings, read from its environment once.
fn settings() -> Result<Option<Settings>> {
    static S: std::sync::OnceLock<Result<Option<Settings>, String>> = std::sync::OnceLock::new();
    S.get_or_init(|| parse_settings(|name| std::env::var(name).ok()))
        .clone()
        .map_err(anyhow::Error::msg)
}

/// What the ranks agree on at startup (`model::startup_parity`): this
/// process's `[on, min_ctx, check]`. Fails on a junk
/// `ATLAS_GLM_INDEX_SPLIT_MIN_CTX`. A rank splitting alone would deadlock the
/// pair at the first owner whose history falls between the two.
pub(crate) fn index_split_words() -> Result<[u64; 3]> {
    settings().map(Settings::words)
}

/// Whether an owner of `rows` rows continuing at `seq_len_start` splits.
fn admits(rows: usize, seq_len_start: usize, min_ctx: usize) -> bool {
    rows >= MIN_ROWS && seq_len_start >= min_ctx
}

/// One exchanged pair of quarters: this rank's selected rows at `own` go to
/// the peer, whose rows land at `peer`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Swap {
    own: usize,
    peer: usize,
    rows: usize,
}

/// One owner's selection buffers.
#[derive(Clone, Copy, Debug)]
pub(super) struct OwnerRows {
    /// The selected token-id rows, `row_bytes` each.
    pub(super) selected: DevicePtr,
    pub(super) row_bytes: usize,
    /// Under the check, the replicated rows.
    pub(super) scratch: DevicePtr,
    /// Under the check, the `(pointer, bytes)` of the index queries and the
    /// head weights every row was selected from.
    pub(super) inputs: [(DevicePtr, usize); 2],
}

/// This rank's share of one owner's selection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct IndexSplit {
    swaps: [Swap; 2],
    rows: usize,
    check: bool,
}

impl IndexSplit {
    /// This rank's split of an owner of `rows` rows continuing at
    /// `seq_len_start` with `row_bytes`-wide selections, or `None` to select
    /// every row here. Both ranks reach the same answer.
    pub(super) fn plan(
        rows: usize,
        seq_len_start: usize,
        row_bytes: usize,
        ctx: &ForwardContext,
    ) -> Result<Option<Self>> {
        let settings = if ctx.comm.is_some() {
            settings()?
        } else {
            None
        };
        Self::plan_with(settings, rows, seq_len_start, row_bytes, ctx)
    }

    fn plan_with(
        settings: Option<Settings>,
        rows: usize,
        seq_len_start: usize,
        row_bytes: usize,
        ctx: &ForwardContext,
    ) -> Result<Option<Self>> {
        let (Some(Settings { min_ctx, check }), Some(comm)) = (settings, ctx.comm) else {
            return Ok(None);
        };
        let eligible = !ctx.graph_capture
            && ctx.config.tp_world_size == 2
            && comm.world_size() == 2
            && admits(rows, seq_len_start, min_ctx)
            && comm.supports_exchange_async(rows / 4 * row_bytes);
        if !eligible {
            return Ok(None);
        }
        ensure!(
            !check || ctx.buffers.sizes().expert_down_out >= 2 * rows * row_bytes,
            "ATLAS_GLM_INDEX_SPLIT_CHECK: no scratch for {rows} replicated rows"
        );
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| {
            tracing::info!(
                "GLM index split: rank {} selects zigzag quarters of owners from {min_ctx} history tokens (check={check})",
                comm.rank()
            )
        });
        Ok(Some(Self {
            swaps: Self::zigzag(rows, comm.rank()),
            rows,
            check,
        }))
    }

    /// Zigzag quarters of an owner's first `4 * (rows / 4)` rows for `rank`:
    /// pair k is (rank 0's quarter, rank 1's quarter).
    fn zigzag(rows: usize, rank: usize) -> [Swap; 2] {
        let q = rows / 4;
        [(0, q), (3 * q, 2 * q)].map(|(r0, r1)| {
            let (own, peer) = if rank == 0 { (r0, r1) } else { (r1, r0) };
            Swap { own, peer, rows: q }
        })
    }

    /// The rows this rank selects, merged into contiguous ranges: its two
    /// quarters and the rows past the last whole quarter, which both ranks
    /// select (rank 0's Q3 runs into them, rank 1's Q1 into its Q2).
    fn own(&self) -> Vec<Range<usize>> {
        let tail = 4 * self.swaps[0].rows..self.rows;
        let mut ranges: Vec<_> = self.swaps.iter().map(|s| s.own..s.own + s.rows).collect();
        ranges.push(tail);
        ranges.sort_by_key(|r| r.start);
        let mut own: Vec<Range<usize>> = Vec::new();
        for r in ranges.into_iter().filter(|r| !r.is_empty()) {
            match own.last_mut() {
                Some(last) if last.end == r.start => last.end = r.end,
                _ => own.push(r),
            }
        }
        own
    }

    /// Swap this rank's finished rows of `rows.selected` for the peer's.
    /// Under the check, then compare every row with the replicated rows.
    pub(super) fn exchange(
        &self,
        rows: &OwnerRows,
        layer: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        for s in self.swaps {
            glm_sp::exchange_rows(
                rows.selected.offset(s.own * rows.row_bytes),
                rows.selected.offset(s.peer * rows.row_bytes),
                s.rows,
                rows.row_bytes,
                false,
                ctx,
                stream,
            )?;
        }
        if self.check {
            self.check(rows, layer, ctx, stream)?;
        }
        Ok(())
    }
}

/// The selection passes of an owner of `rows` rows, as row ranges and the
/// output each lands in: every row into `selected` when replicated; split,
/// this rank's rows, then (check) every row again into `scratch`.
pub(super) fn passes(
    split: Option<IndexSplit>,
    rows: usize,
    selected: DevicePtr,
    scratch: DevicePtr,
) -> Vec<(Range<usize>, DevicePtr)> {
    let Some(split) = split else {
        return vec![(0..rows, selected)];
    };
    let mut passes: Vec<_> = split.own().into_iter().map(|r| (r, selected)).collect();
    if split.check {
        passes.push((0..rows, scratch));
    }
    passes
}

/// `(row_start, rows, output)` tiles of at most `tile_rows` rows over
/// `passes`.
pub(super) fn tiles(
    passes: &[(Range<usize>, DevicePtr)],
    tile_rows: usize,
) -> impl Iterator<Item = (usize, usize, DevicePtr)> + '_ {
    passes.iter().flat_map(move |(range, out)| {
        range
            .clone()
            .step_by(tile_rows)
            .map(move |start| (start, tile_rows.min(range.end - start), *out))
    })
}

#[cfg(test)]
#[path = "glm_index_split_tests.rs"]
mod tests;
