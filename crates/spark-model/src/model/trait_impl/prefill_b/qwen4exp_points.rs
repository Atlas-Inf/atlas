// SPDX-License-Identifier: AGPL-3.0-only

//! Where a qwen4_exp prefill pass captures checkpoints in-pass, beyond the
//! tail one (`qwen4exp_ckpt`). Both default off; both need
//! `ATLAS_QWEN4EXP_PREFILL_MIDCHUNK_CKPT=1`, whose capture machinery they
//! reuse (`layers::qwen4exp_ckpt`).
//!
//! # Dense checkpoints (`ATLAS_QWEN4EXP_DENSE_CKPT=N`)
//!
//! Base checkpoints a prompt only at its 16K chunk boundaries and its tail,
//! so a request that diverges from a cached conversation mid-history (an
//! edited or re-rendered transcript, a memory block that changed) restores
//! up to a whole chunk below its match and replays the rest. With the
//! switch every pass also captures at each multiple of `N` tokens strictly
//! inside it (`N` rounded down to the 64-row GDN grid, at least 64): the
//! GDN state and conv window, the PLE carry and the QSA keys at that row,
//! exactly what a pass ending there would leave (the GPU tests of
//! `layers::qwen4exp_ckpt`). A row the snapshot index already holds for
//! these tokens is skipped (a replayed or shared prefix keeps its earlier
//! checkpoints), so each conversation pays one slot per `N` new tokens.
//!
//! # Branch-point checkpoints (`ATLAS_QWEN4EXP_PC_BRANCH=1`)
//!
//! The idea of `ATLAS_GLM_PC_BRANCH` (`pc_policy`, Marconi's branch-point
//! admission): a prefill whose radix match lies at least
//! `ATLAS_QWEN4EXP_PC_BRANCH_MIN` tokens (default 1024) above its restore
//! depth, at a fork of the cached KV path, checkpoints the match, so the
//! next request that shares the prefix (a new conversation on the same
//! preamble, a sibling of an edited history) restores there. GLM splits the
//! chunk into two passes for it; here the checkpoint is captured in-pass at
//! the last 64-row boundary of the pass at or below the match (a 64-row
//! multiple, which ROWINV restores require), so the prefill keeps its one
//! pass and its numerics. The plan, its fork test min-reduced across ranks,
//! is `pc_policy::pc_plan_branch`; the snapshot is marked a branch point for
//! chain-aware eviction (`ATLAS_GLM_PC_EVICT`).
//!
//! # Ranks
//!
//! The rows are a function of (tokens, pass, config) and of the agreed match
//! and restore depth, the same on every rank. What can differ per rank is
//! whether the index already holds a row (skipped) or a slot is free, and
//! a capture issues no collective, so a rank that has fewer checkpoints only
//! makes the pair restore shallower (`pc_policy::agree_restore`).

use spark_runtime::kv_cache::PagedKvCache;

use super::super::super::types::TransformerModel;
use crate::layers::qwen4exp_ckpt as ckpt;
use crate::traits::SequenceState;

/// `ATLAS_QWEN4EXP_DENSE_CKPT=N`: the dense checkpoint spacing in tokens,
/// on the 64-row grid; 0 (default) off.
pub(in crate::model) fn dense_every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("ATLAS_QWEN4EXP_DENSE_CKPT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(0);
        grid_spacing(n)
    })
}

/// `N` rounded down to the 64-row grid, at least one grid step (0 stays 0).
pub(super) fn grid_spacing(n: usize) -> usize {
    if n == 0 {
        0
    } else {
        (n / ckpt::CHUNK).max(1) * ckpt::CHUNK
    }
}

/// `ATLAS_QWEN4EXP_PC_BRANCH=1`: in-pass branch-point checkpoints.
pub(in crate::model) fn branch_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        matches!(
            std::env::var("ATLAS_QWEN4EXP_PC_BRANCH").as_deref(),
            Ok("1") | Ok("true")
        )
    })
}

/// `ATLAS_QWEN4EXP_PC_BRANCH_MIN`: the smallest match-over-restore gap worth
/// a branch checkpoint (default 1024 tokens). An in-pass capture costs one
/// snapshot slot and no pass, so the bar is lower than GLM's split.
pub(in crate::model) fn branch_min() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ATLAS_QWEN4EXP_PC_BRANCH_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1024)
    })
}

/// The pass's geometry and the rows asked of it, for [`plan_points`].
#[derive(Clone, Copy, Debug)]
pub(super) struct PointAsk {
    /// The pass covers `[start, start + count)`.
    pub start: usize,
    pub count: usize,
    /// The tail checkpoint row (`qwen4exp_ckpt::tail_ckpt_row`), if any.
    pub tail: Option<usize>,
    /// The planned branch checkpoint (the agreed radix match), if any.
    pub branch: Option<usize>,
    /// Dense spacing ([`dense_every`]), 0 = none.
    pub every: usize,
    /// Every row must be a multiple of this (the QSA pool ratio, and the
    /// block or GDN grid the restore takes).
    pub align: usize,
}

/// The rows a pass captures, ascending, as (row, branch point): the tail
/// row, the branch row (floored to the last 64-row boundary of the pass at
/// or below it) and the dense rows below the tail, each strictly inside the
/// pass, a 64-row boundary of it and an `align` multiple, minus the rows
/// `have` reports already checkpointed (never the tail, which is marked a
/// branch point when the two coincide). At most
/// [`ckpt::MAX_POINTS`]: the tail and branch rows first, then the deepest
/// dense rows.
pub(super) fn plan_points(
    ask: PointAsk,
    mut have: impl FnMut(usize) -> bool,
) -> Vec<(usize, bool)> {
    let PointAsk {
        start,
        count,
        tail,
        branch,
        every,
        align,
    } = ask;
    let end = start + count;
    let ok = |r: usize| {
        r > start
            && r < end
            && (r - start).is_multiple_of(ckpt::CHUNK)
            && r.is_multiple_of(align.max(1))
    };
    let branch = branch
        .filter(|&b| b > start)
        .map(|b| start + (b.min(end) - start) / ckpt::CHUNK * ckpt::CHUNK)
        .filter(|&b| ok(b) && (Some(b) == tail || !have(b)));
    // The tail row is `tail_ckpt_row`'s, already on its own grid.
    let mut out: Vec<(usize, bool)> = Vec::new();
    out.extend(tail.map(|t| (t, Some(t) == branch)));
    out.extend(branch.filter(|&b| Some(b) != tail).map(|b| (b, true)));
    if every > 0 {
        let top = tail.unwrap_or(end);
        let first = (start / every + 1) * every;
        let dense: Vec<usize> = (first..top)
            .step_by(every)
            .filter(|&r| ok(r) && Some(r) != branch)
            .collect();
        let room = ckpt::MAX_POINTS.saturating_sub(out.len());
        let mut picked = 0;
        for r in dense.into_iter().rev() {
            if picked == room {
                break;
            }
            if !have(r) {
                out.push((r, false));
                picked += 1;
            }
        }
    }
    out.sort_by_key(|&(r, _)| r);
    out
}

impl TransformerModel {
    /// Branch-point checkpoints go in-pass for this model
    /// (`ATLAS_QWEN4EXP_PC_BRANCH`): the planned branch never splits a chunk.
    pub(in crate::model) fn qwen4exp_pc_branch(&self) -> bool {
        branch_on() && ckpt::requested() && self.config.model_type == "qwen4_exp"
    }

    /// The checkpoint rows of a pass over `[start, start + count)` beyond
    /// what the tail asks (`tail`), with this request's branch plan and the
    /// index's existing checkpoints of `tokens`.
    pub(super) fn qwen4exp_pass_points(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        [start, count]: [usize; 2],
        tail: Option<usize>,
        bs: usize,
    ) -> Vec<(usize, bool)> {
        let ratio = self.config.indexer_compress_ratio.max(1);
        let grid = if crate::layers::ops::qwen4exp_rowinv::on() {
            crate::layers::ops::qwen4exp_rowinv::PASS_GRANULE
        } else {
            bs
        };
        let ask = PointAsk {
            start,
            count,
            tail,
            branch: seq.pc_branch_at.filter(|_| self.qwen4exp_pc_branch()),
            every: dense_every(),
            align: num_lcm(ratio, grid),
        };
        let adapter = seq.adapter_id;
        plan_points(ask, |r| {
            self.prefix_cache.snapshot_at(tokens, r, adapter).is_some()
        })
    }

    /// Reserve a snapshot slot per row (best effort: a row without a slot is
    /// dropped). Returns (row, slot, branch) for the rows that got one.
    pub(super) fn qwen4exp_reserve_points(
        &self,
        seq: &SequenceState,
        kv_cache: &mut PagedKvCache,
        rows: &[(usize, bool)],
    ) -> Vec<(usize, usize, bool)> {
        let mut out = Vec::with_capacity(rows.len());
        for &(r, branch) in rows {
            match self.reserve_snapshot_slot(seq.session_hash, kv_cache) {
                Some(slot) => out.push((r, slot, branch)),
                None => {
                    tracing::warn!("qwen4_exp in-pass checkpoint: no snapshot slot for token {r}")
                }
            }
        }
        out
    }
}

fn num_lcm(a: usize, b: usize) -> usize {
    let (mut x, mut y) = (a.max(1), b.max(1));
    while y != 0 {
        (x, y) = (y, x % y);
    }
    a.max(1) / x * b.max(1)
}

#[cfg(test)]
#[path = "qwen4exp_points_tests.rs"]
mod tests;
