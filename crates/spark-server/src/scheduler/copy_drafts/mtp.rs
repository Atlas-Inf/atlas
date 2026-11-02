// SPDX-License-Identifier: AGPL-3.0-only

//! Copy drafts on the MTP path (`ATLAS_QWEN_COPY_DRAFTS=1`, default off).
//!
//! The DFlash2 lane offers a copy over a block the drafter already paid
//! for. An MTP head drafts sequentially, one head pass a draft (up to seven
//! on qwen4_exp's exact lane), so here a copy stands in for the propose
//! itself: after a verify, when the context's last `match` tokens occurred
//! earlier, the tokens that followed them become the next verify's drafts
//! and the head does not run. Index, proposal and knobs are the parent
//! module's (`ATLAS_DFLASH_COPY_MATCH`, `_MAX`, `_REPLY_MATCH`,
//! `_MISS_MAX`); the drafts carry `COPY_CONF` / `COPY_REPLY_CONF` as
//! there, which is how the verify recognises a copy round.
//!
//! **Exact.** The lane runs only where every verify row is serial decode's
//! arithmetic whatever its window and whichever rows share its forward
//! (`Model::batch_verify_bit_exact`: qwen4_exp under
//! `ATLAS_QWEN4EXP_EXACT_VERIFY=1` + `ATLAS_QWEN4EXP_BATCH_FAST=1`,
//! `trait_impl/verify_rows.rs`). The accept rule is unchanged, so every
//! emitted token is the target's pick at its own row, and since that pick
//! does not depend on the window a copy leaves, a greedy run commits what
//! serial decode commits, as with the head's drafts. Copies never widen a
//! verify past what the head's drafts reach: in a batch they take the
//! step's ladder like any drafts; a LONE sequence's copy may verify up to
//! the step's ceiling (`--num-drafts`, at most 7 drafts) within its slot's
//! capacity, a window the head's own deep depth reaches ([`deep_cap`]).
//! Elsewhere the lane is inert: on GLM and on the inexact lanes nothing
//! here runs.
//!
//! - **Where.** Every propose that follows a verify of drafts ([`propose`]):
//!   the K=2/3/4 verdicts and the batched verify's propose. A bootstrap
//!   (the first drafts of a request, after a serial stretch, a decode row)
//!   always runs the head: its first propose builds the drafter's prompt
//!   context (`ensure_drafter_context`), which a skipped propose would lose.
//! - **Grammar.** None under a grammar, as on the DFlash2 lane.
//! - **Thinking.** Copies run inside `<think>` too: the exact lane
//!   speculates there (`mtp_gate::spec_think_for`), and a copy is a draft.
//! - **Drafter state.** The head wrote no rows for a copy, so a copy round's
//!   verdict skips `trim_proposer_state` ([`trim_proposer`]): the rows of the
//!   head's last propose were settled by the verdict before it, and a trim
//!   would cut rows of positions the copy never touched. The drafter's KV is
//!   append-only with sequence-space RoPE, so the copied span is a gap in
//!   its context, as a serial stretch leaves one. A TP worker trims its idle
//!   drafter state as before (it never proposes; the head runs the drafter).
//! - **Depth.** The head's depth controllers (`mtp_deep_depth`, the n=1
//!   ladders) skip copy rounds ([`settle`]): a copy says nothing about the
//!   head's per-position acceptance. Under D-Cut (off in the production
//!   profiles) a copy's confidences rank above any log-probability, so its
//!   positions are pruned last; the canonical depth assignment may still cut
//!   it, as it cuts any member.
//! - **Cost.** A copy that misses at its first draft leaves the step with no
//!   drafts the head would have kept; `ATLAS_DFLASH_COPY_MISS_MAX` narrows
//!   the copies after a miss until one is kept whole.
//! - **Stats.** Copy rounds are booked in the `COPY DRAFTS` line after the
//!   request's `Done:` line, of the prompt and of the reply apart.
//!
//! **Long copies past 7 drafts: design, not built.** The SparkGLM line's
//! `ATLAS_DFLASH_COPY_LONG_MAX` (branch `exp/copy-long-20261014`, commits
//! c3399f156 and 8d3734d3e: a lone owner's prompt copy, after a copy round
//! kept whole, proposes up to N drafts, its verify rows borrowed from idle
//! slots) does not carry over by a knob. On this lane every per-row
//! resource of the verify is sized for at most 8 rows, and in snapshot
//! form: `EXACT_LANE_MAX_ROWS` = 8 (`mtp_verify_rows_ok`), the GDN chain's
//! `GDN_VERIFY_KMAX` = 8 tokens a launch (`qwen4exp_gdn_verify_fused_rows`,
//! H and conv windows in registers), `VERIFY_SNAP_SLOTS` = 9 PLE conv-carry
//! snapshots, and per slot `K - 1` H intermediates (FP32 recurrent states,
//! `ssm_pool::verify_draft_capacity`) plus `K` conv rows. The GLM borrow
//! (that branch's `model::ssm_borrow`) relies on KDA fold records addressed
//! slot-major; it refuses the H snapshot rollback because a slot holds
//! `K - 1` H rows against `K` conv rows and the rollback reads them by
//! slot, which is exactly qwen4_exp's rollback. A long copy here therefore
//! needs, in order:
//! (1) a window rule for H snapshots (n slots lend `n(K - 1)` H rows, so
//! the window is `n(K - 1) + 1` rows, and `commit_accepted_prefix` must
//! read H row `na` and conv row `na` from the window, not by slot); (2) the
//! exact GDN chain past 8 tokens (a per-token loop over `GDN_VERIFY_KMAX`
//! chunks, the H state carried across launches, each chunk the same
//! per-token arithmetic); (3) PLE snapshots and the QSA rewind for the long
//! window (`VERIFY_SNAP_SLOTS`, `REWIND_MARGIN` 512 already covers it);
//! (4) the exact lane's row tiers past 8 rows a sequence (projections, MoE
//! and the LM head already tier to 32 rows a launch by the step's total,
//! `ops::Qwen4ExpWideRows`, so the per-sequence bound is the scheduler's,
//! `step_verify_k4_batched`'s `ks <= 8` and the worker's `(1..=32)`);
//! (5) the scheduler: the branch's `Model::lone_verify_draft_capacity` (the
//! window while the owner is alone, else its own `K - 1`) capping both the
//! proposal here and `deep_cap`, so an admission between proposal and
//! verify cuts the verify back. Each step changes the verify's rows and
//! must be proved exact against serial decode on hardware (a per-row check
//! in the style of `ATLAS_QWEN4EXP_EXACT_VERIFY_CHECK`, which covers K=2..4
//! only, and `bench/greedy_eq.py`) before the next. TODO(long-copy):
//! (1)-(5).
//!
//! Prior art: copy drafts beside an MTP head is MiaAI-Lab's TensorFold
//! recipe for this model pair, whose logs show copy rounds on structured
//! output, code and prose (head-to-head 2026-10-16); the policy is the
//! parent module's (see its prior art and docs/glm-prior-art.md). Ideas, no
//! code. Ours: copies in place of the head's propose, and the exact-lane
//! gate that keeps speculation equal to serial decode.

use spark_model::traits::Model;

use super::{Settings, copy_for, is_copy, knobs};
use crate::scheduler::sched_ctx::SchedCtx;
use crate::scheduler::types::ActiveSeq;

/// Whether this run copies on the MTP path: the lever, on a model whose
/// every verify row, alone or batched, is serial decode's.
fn lane(model: &dyn Model, sched: &SchedCtx) -> bool {
    sched.levers.qwen_copy_drafts && model.batch_verify_bit_exact()
}

/// After a verify, draft `a`'s next one from a copy in place of the MTP
/// head's propose, up to `ceiling` drafts (the step's `--num-drafts`) within
/// its slot's capacity. Returns whether it did: the caller then skips the
/// propose. No-op off the lane.
pub(in crate::scheduler) fn propose(
    model: &dyn Model,
    sched: &SchedCtx,
    a: &mut ActiveSeq,
    ceiling: usize,
) -> bool {
    lane(model, sched) && propose_with(model, a, ceiling, knobs())
}

fn propose_with(model: &dyn Model, a: &mut ActiveSeq, ceiling: usize, s: &Settings) -> bool {
    if a.grammar_state.is_some() {
        return false;
    }
    let k = s
        .max
        .min(ceiling)
        .min(model.mtp_slot_draft_capacity(a.seq.slot_idx));
    let Some((copy, copy_conf)) = copy_for(a, s, k) else {
        return false;
    };
    a.pending_draft_conf = vec![copy_conf; copy.len()];
    a.pending_drafts = copy;
    true
}

/// At the take of `a`'s pending drafts for a verify: note whether they are
/// a copy, for the verdict ([`trim_proposer`], [`settle`]). No-op off the
/// lane.
pub(in crate::scheduler) fn begin_verify(model: &dyn Model, sched: &SchedCtx, a: &mut ActiveSeq) {
    if lane(model, sched) {
        let conf = a.draft_conf();
        let copy = conf.first().copied().filter(|_| is_copy(conf));
        a.spec_adapt.copy.verifying = copy;
    }
}

/// `trim_proposer_state` for a verdict accepting `num_accepted` drafts,
/// skipped for a copy round (the head drafted none of them; module doc).
pub(in crate::scheduler) fn trim_proposer(
    model: &dyn Model,
    a: &mut ActiveSeq,
    num_accepted: usize,
) -> anyhow::Result<()> {
    if a.spec_adapt.copy.verifying.is_some() {
        return Ok(());
    }
    model.trim_proposer_state(&mut a.seq, num_accepted, 0)
}

/// Book a verify of `drafted` drafts that accepted `accepted`. Returns
/// whether it was a copy round, which the head's depth controllers skip.
pub(in crate::scheduler) fn settle(a: &mut ActiveSeq, drafted: usize, accepted: usize) -> bool {
    let state = &mut a.spec_adapt.copy;
    let Some(copy_conf) = state.verifying else {
        return false;
    };
    super::settle(state, &[copy_conf], drafted, accepted);
    true
}

/// The drafts `a` may verify in a deep step (`one_forward::deep_truncate`):
/// `own`, the head's depth under the step's ladder, or for a LONE
/// sequence's copy the step's `ceiling` within its slot's capacity.
pub(in crate::scheduler) fn deep_cap(
    model: &dyn Model,
    sched: &SchedCtx,
    a: &ActiveSeq,
    lone: bool,
    own: usize,
    ceiling: usize,
) -> usize {
    if lone && lane(model, sched) && is_copy(a.draft_conf()) {
        ceiling.min(model.mtp_slot_draft_capacity(a.seq.slot_idx))
    } else {
        own
    }
}

#[cfg(test)]
#[path = "mtp_tests.rs"]
mod tests;
