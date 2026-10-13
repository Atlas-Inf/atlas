// SPDX-License-Identifier: AGPL-3.0-only

//! Prefix-cache snapshot policy for hybrid SSM models: the restore-depth rank
//! agreement and branch-point checkpoints.
//!
//! # Branch-point checkpoints (`ATLAS_GLM_PC_BRANCH=1`, default off)
//!
//! A prefill saves one SSM checkpoint per turn: the tail split, one block
//! below the last block boundary under the prompt end. That serves the next
//! turn of the SAME conversation. A NEW conversation that shares a long
//! prefix with an earlier one (agentic clients: the same 20-60K-token system
//! prompt and tool list in every session, subagent and side request) matches
//! that prefix in the KV radix but finds no snapshot at or near the match,
//! because every existing snapshot sits at some other conversation's tail. It
//! then recomputes all of it ("Prefix cache hit: N tokens but no SSM snapshot
//! — recomputing all KV"), or replays from a much shallower snapshot.
//!
//! With the flag, a prefill whose radix match `matched` lies at least
//! `ATLAS_GLM_PC_BRANCH_MIN` tokens (default 2048) above its restore depth,
//! below the tail cut, and exactly where the cached KV path forks (another
//! cached request continues past `matched` with a different block, on every
//! rank) splits the chunk that spans `matched` and saves a checkpoint there.
//! The next request with that prefix restores it and replays only its own
//! suffix. This is Marconi's branch-point admission (MLSys'25,
//! arXiv:2411.19379). The checkpoint is an ordinary block-aligned prefill
//! checkpoint, so it stays valid under `ATLAS_MARCONI_PREFILL_ONLY`. The
//! cost is one extra prefill pass, paid once per shared prefix by the first
//! request that finds the fork. The fork test keeps the pass off a
//! conversation's own next turn: its match ends where its own cached path
//! ends, which is not a fork. (A reasoning turn whose re-rendered history
//! drops the decoded thinking does fork there, but its own tail checkpoint
//! sits within two blocks of the match, far under the minimum.)
//!
//! ## Accumulation order (lossless, but cache-dependent)
//!
//! The split is the same math and the same kernels, but NOT the same
//! accumulation order as the unsplit pass. The request that plants the
//! checkpoint runs `[start, at)` and `[at, end)` as two passes where a cold
//! run of the same prompt runs one, so every row from `at` on sees different
//! GEMM M-shapes, MoE groupings, attention tiling over the cached prefix, KDA
//! pieces (`glm5_kda::flash_prefill` splits a pass into near-equal pieces by
//! its row count) and SP row halves. Its output can differ bitwise from a
//! cold run, and a near-tied greedy argmax can flip, exactly as for any
//! Marconi intermediate restore (the requests that later restore at `at` run
//! the same `[at, end)` shape). Base makes the chunk shape a pure function of
//! (tokens, config) for this reason (`prefill_chunk_dispatch`, tail split);
//! this flag deliberately trades that for the shared-prefix restore. No
//! alignment of `at` can restore the unsplit shape, since the KDA piece
//! layout depends on the pass length.
//!
//! # Restore-depth rank agreement
//!
//! TP2 ranks keep their own snapshot pools and indexes. Only the radix match
//! is min-reduced across ranks (F83), and the Marconi restore depth sets each
//! rank's processed row range, so ranks that restore at different depths run
//! mismatched collectives. The pools and eviction victims do diverge across
//! ranks, with or without either flag (seen on qwen4_exp TP2: four concurrent
//! 22.6K-token turns of shared-system-prompt conversations, one rank restored
//! at 22,528 while the other found no snapshot and recomputed all of it; both
//! ranks then waited in different collectives forever), and a spill tier (the
//! NVMe prefix tier `ATLAS_KV_NVME_DIR` or the snapshot tier `ATLAS_SSM_TIER`)
//! adds restores and fault-ins that succeed on one rank only. So every
//! multi-rank world agrees on the restore depth ([`agree_restore`]): the
//! minimum depth any rank can restore, taken only if every rank holds an
//! exact-prefix snapshot at that depth, and otherwise a full recompute
//! everywhere. That costs one 4-byte min-reduction per prefill, plus a second
//! one when there is something to restore (and a third when a branch
//! checkpoint is a candidate).
//!
//! # Rank env parity
//!
//! The flags are read per process. Both ranks MUST run with the same values
//! of `ATLAS_GLM_PC_EVICT`, `ATLAS_GLM_PC_BRANCH` and
//! `ATLAS_GLM_PC_BRANCH_MIN`; a rank with a different set issues different
//! collectives and chunk shapes and the pair deadlocks, or pairs a
//! restore-depth reduction with the peer's KV-admission vote. The ranks
//! compare them at startup (`model::startup_parity`), with
//! `ATLAS_EP_PROTOCOL` and the rest of that table.
//!
//! So are `ATLAS_GLM_PC_WRITE_FLOOR` and `ATLAS_GLM_KV_WRITE_FLOOR_LEGACY`.
//! A mismatch there does not deadlock (the cache writes they skip are not
//! collectives), but one rank then attends the cached rows of a shared block
//! and the other its own recompute of them, which is a silent TP numerics
//! fault.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::prefix_cache::PrefixMatch;

use super::super::super::types::TransformerModel;
use crate::traits::SequenceState;

/// `ATLAS_GLM_PC_BRANCH=1`: branch-point checkpoints. Read once.
pub(in crate::model) fn glm_pc_branch_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PC_BRANCH").as_deref() == Ok("1"))
}

/// Smallest shared prefix worth a branch checkpoint (`ATLAS_GLM_PC_BRANCH_MIN`,
/// default 2048 tokens). The checkpoint costs one extra prefill pass, roughly
/// one sweep of the active weights (~0.1-0.2 s on GB10 TP2), once. Each later
/// request that shares the prefix then skips at least this many tokens.
pub(in crate::model) fn glm_pc_branch_min_tokens() -> usize {
    static MIN: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MIN.get_or_init(|| {
        std::env::var("ATLAS_GLM_PC_BRANCH_MIN")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(2048)
    })
}

/// `ATLAS_GLM_PC_WRITE_FLOOR=1`: see [`layer_write_floor`]. Read once; see
/// "Rank env parity" above.
pub(in crate::model) fn glm_pc_write_floor_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_GLM_PC_WRITE_FLOOR").as_deref() == Ok("1"))
}

/// The KV write floor a prefill pass over rows `[start, start + rows)` hands
/// its layers. `replay_floor` is the base value: the rows a Marconi replay
/// spends under the radix match `matched`, and 0 for a prefix hit with
/// nothing to restore, which recomputes from token 0 and rewrites every
/// matched block under the sequences that share it. With the flag (`flag`,
/// GLM only) that pass floors its writes at the match as well: the matched
/// blocks are fully written (`kv_valid_tokens` caps what the radix holds),
/// so its attention reads them instead. Same kernels and math for every row;
/// rows under the match then attend to the cached K/V rather than this
/// pass's own recompute of it, so its output can differ bitwise from the
/// flag-off run, as a Marconi replay's does from a cold one.
///
/// The flag makes matched blocks write-once, so it relies on nothing having
/// poisoned them: run it with `ATLAS_PREFIX_SUBBLOCK=0` and without
/// `ATLAS_MARCONI_EXACT=1` (the production profile). Flag off, a recompute
/// overwrites such a block; flag on, every later request reads it until it
/// is evicted.
pub(super) fn layer_write_floor(
    flag: bool,
    model_type: &str,
    replay_floor: usize,
    matched: usize,
    start: usize,
    rows: usize,
) -> usize {
    if flag && model_type == "glm5_next" {
        replay_floor.max(matched.saturating_sub(start).min(rows))
    } else {
        replay_floor
    }
}

impl TransformerModel {
    /// Whether a recompute-all prefix hit of this model keeps its matched
    /// rows as cached (`ATLAS_GLM_PC_WRITE_FLOOR`, GLM only): what
    /// [`layer_write_floor`] gives a one-row pass under a one-row match.
    pub(in crate::model) fn pc_write_floor_keeps_matched(&self) -> bool {
        let flag = glm_pc_write_floor_enabled();
        layer_write_floor(flag, &self.config.model_type, 0, 1, 0, 1) == 1
    }

    /// [`layer_write_floor`] of one pass of this model, logged when the flag
    /// raised it.
    pub(super) fn pc_write_floor(
        &self,
        replay_floor: usize,
        matched: usize,
        start: usize,
        rows: usize,
    ) -> usize {
        let floor = layer_write_floor(
            glm_pc_write_floor_enabled(),
            &self.config.model_type,
            replay_floor,
            matched,
            start,
            rows,
        );
        if floor > replay_floor {
            tracing::info!(
                "ATLAS_GLM_PC_WRITE_FLOOR: pass at token {start} keeps {floor} of {rows} rows \
                 as cached (match {matched})"
            );
        }
        floor
    }
}

/// `ATLAS_NO_TAIL_SPLIT=1`: a last chunk runs as one pass, without the
/// tail-checkpoint split (`prefill_chunk_dispatch_with`).
pub(in crate::model) fn tail_split_disabled() -> bool {
    std::env::var("ATLAS_NO_TAIL_SPLIT").as_deref() == Ok("1")
}

/// The tail-split cut for a `total`-token prompt: one block below the last
/// block boundary strictly under `total` (`prefill_chunk_dispatch_with`).
pub(in crate::model) fn tail_cut(total: usize, bs: usize) -> usize {
    // Or, with ATLAS_GLM_TAIL_CUT_DEEP, that boundary itself.
    crate::model::warm_turn::tail_cut_at(total, bs, crate::model::warm_turn::tail_cut_deep())
}

/// Where to save a branch checkpoint, if anywhere: at the radix match
/// `matched` when it lies at least `min` tokens above the restore depth
/// `skip_to` (0 when nothing was restored) and below the tail cut (which
/// already gets a checkpoint). The caller also requires a radix fork there.
pub(super) fn branch_checkpoint_at(
    matched: usize,
    skip_to: usize,
    total: usize,
    bs: usize,
    min: usize,
) -> Option<usize> {
    (matched > skip_to && matched - skip_to >= min && matched < tail_cut(total, bs))
        .then_some(matched)
}

/// Which snapshot a rank restores after [`agree_restore`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::model) enum Agreed {
    /// No rank restores.
    None,
    /// This rank's own lookup choice, at its own depth.
    Local,
    /// The exact-prefix snapshot with this id at the (shallower) agreed depth.
    At(usize),
}

/// The cross-rank restore-depth agreement, over its inputs: this rank's
/// restorable depth (`0` = none), a cross-rank min-reduction `min`, and
/// `probe(depth)`, this rank's restorable exact-prefix snapshot at a
/// shallower depth. Returns the agreed depth and what this rank restores;
/// every rank gets the same depth, and `(0, Agreed::None)` on all ranks when
/// any rank lacks the agreed snapshot. Issues one reduction, plus a second
/// exactly when the agreed depth is non-zero, so the collective count is the
/// same on every rank whatever the local inputs.
pub(in crate::model) fn agree_restore(
    depth: usize,
    mut min: impl FnMut(u32) -> Result<u32>,
    probe: impl FnOnce(usize) -> Option<usize>,
) -> Result<(usize, Agreed)> {
    let agreed = min(u32::try_from(depth)?)? as usize;
    if agreed == 0 {
        return Ok((0, Agreed::None));
    }
    let mine = if agreed == depth {
        Some(Agreed::Local)
    } else {
        probe(agreed).map(Agreed::At)
    };
    let all = min(u32::from(mine.is_some()))? == 1;
    Ok(match mine {
        Some(r) if all => (agreed, r),
        _ => (0, Agreed::None),
    })
}

/// The split point for a chunk `[start, start + len)`: the planned branch
/// checkpoint when it lies strictly inside the chunk. A checkpoint on a chunk
/// boundary needs no split. Chunks carrying DFlash verify owners
/// (`passengers`) never split.
pub(super) fn branch_split_at(
    at: Option<usize>,
    (start, len): (usize, usize),
    passengers: bool,
) -> Option<usize> {
    at.filter(|&a| !passengers && a > start && a < start + len)
}

impl TransformerModel {
    /// Whether `snap_id` at `snap_tok` may be restored for this lookup. This
    /// is the base restore gate, unchanged; see `prefill_b_prefix_lookup` for
    /// why each term exists.
    pub(super) fn marconi_restorable(
        &self,
        snap_id: usize,
        snap_tok: usize,
        is_tail: bool,
        matched: usize,
        total: usize,
        seq: &SequenceState,
    ) -> bool {
        let exact_without_hidden =
            snap_tok == matched && matched == total && !self.ssm_snapshots.has_hidden(snap_id);
        let bypass_exact =
            snap_tok == matched && matched == total && !crate::model::mtp_carry::marconi_exact();
        snap_tok >= crate::model::mtp_carry::marconi_min_tokens()
            && snap_tok > 0
            && matched <= total
            && !exact_without_hidden
            && !bypass_exact
            && (!is_tail
                || self
                    .ssm_snapshots
                    .session_matches(snap_id, seq.session_hash))
            && (!self.requires_aux_state() || self.ssm_snapshots.has_aux(snap_id))
    }

    /// Agree on one restore `(snapshot, depth, is_tail)` across ranks (see
    /// the module docs and [`agree_restore`]). Returns the local choice
    /// unchanged in a single-rank world;
    /// `(None, 0, false)` means no rank restores.
    pub(super) fn pc_agree_restore(
        &self,
        tokens: &[u32],
        seq: &SequenceState,
        prefix_match: &PrefixMatch,
        matched: usize,
        total: usize,
        local: (Option<usize>, usize),
    ) -> Result<(Option<usize>, usize, bool)> {
        let is_tail = prefix_match.ssm_snapshot_is_tail;
        if !self.multi_rank_protocol_active() {
            return Ok((local.0, local.1, is_tail));
        }
        let restorable = |id: usize, tok: usize, tail: bool| {
            self.marconi_restorable(id, tok, tail, matched, total, seq)
        };
        let depth = match local {
            (Some(id), tok) if restorable(id, tok, is_tail) => tok,
            _ => 0,
        };
        let (agreed, mine) = agree_restore(
            depth,
            |v| self.ep_min_u32(v),
            |at| {
                self.prefix_cache
                    .snapshot_at(tokens, at, seq.adapter_id)
                    .filter(|&id| restorable(id, at, false))
            },
        )?;
        let out = match mine {
            Agreed::None => (None, 0, false),
            Agreed::Local => (local.0, agreed, is_tail),
            Agreed::At(id) => (Some(id), agreed, false),
        };
        if out.1 != depth {
            tracing::info!(
                "pc rank-agree: local restore depth {depth}, restoring at {}",
                out.1
            );
        }
        Ok(out)
    }

    /// Plan this prefill's branch checkpoint (chunk 0, after the restore
    /// decision). Every input but the fork test is rank-agreed, and the fork
    /// test is min-reduced, so every rank plans the same split and issues the
    /// same collectives.
    pub(super) fn pc_plan_branch(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        matched: usize,
        skip_to: usize,
        bs: usize,
    ) -> Result<()> {
        seq.pc_branch_at = None;
        if !glm_pc_branch_enabled()
            || self.config.num_ssm_layers() == 0
            || !self.ssm_snapshots.is_enabled()
            || !self.prefix_cache.is_active()
            || self.tokens_have_vision_pad(tokens)
        {
            return Ok(());
        }
        let min = glm_pc_branch_min_tokens();
        let Some(at) = branch_checkpoint_at(matched, skip_to, tokens.len(), bs, min) else {
            return Ok(());
        };
        let mut fork = self.prefix_cache.forks_at(tokens, at, bs, seq.adapter_id);
        if self.multi_rank_protocol_active() {
            fork = self.ep_min_u32(u32::from(fork))? == 1;
        }
        if fork {
            seq.pc_branch_at = Some(at);
            tracing::info!(
                "pc branch checkpoint planned at token {at} ({matched}-token shared prefix, \
                 restore depth {skip_to})"
            );
        }
        Ok(())
    }

    /// Run `[start, start + len)` as two chunks split at the planned branch
    /// checkpoint `at`; the first ends at `at`, where
    /// `prefill_b_save_checkpoint` saves it.
    pub(super) fn pc_branch_split(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        (start, len): (usize, usize),
        at: usize,
        is_last_chunk: bool,
        stream: u64,
    ) -> Result<DevicePtr> {
        self.prefill_chunk_dispatch(tokens, seq, start, at - start, false, stream)?;
        self.prefill_chunk_dispatch(tokens, seq, at, start + len - at, is_last_chunk, stream)
    }
}

#[cfg(test)]
#[path = "pc_policy_tests.rs"]
mod tests;
