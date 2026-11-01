// SPDX-License-Identifier: AGPL-3.0-only

//! Per-step draft-depth adjustments on the MTP verify dispatch: the n=1
//! single-depth adaptation, the lone-DFlash width, and the serial ladder cap.

use super::*;

/// `ladder_nd` after the n=1 single-depth adaptation.
pub(super) fn single_depth_ladder(
    active: &[ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    ladder_nd: usize,
    dflash_verify_raw_argmax: bool,
) -> usize {
    // At n=1, GLM can choose between the measured K=3 and K=5 kernels from
    // this request's own acceptance history.  This composes after the global
    // concurrency ladder and is a carried per-run lever rather than global
    // process state (`ATLAS_MTP_SINGLE_DEPTH_ADAPT=1`).
    if active.len() == 1 && !dflash_verify_raw_argmax && sched.levers.mtp_single_depth_adapt {
        active[0]
            .mtp_acct
            .depth_drafts(ladder_nd, sched.levers.depth())
    } else {
        ladder_nd
    }
}

/// The step's draft width under the dynamic depth (`mtp_deep_depth`): the
/// deepest per-request ceiling among `active` under the configured
/// `ceiling`, so the drafter refills every sequence to at least its own;
/// `plan_verify` cuts each back to its ceiling and the step's row budget.
pub(super) fn deep_ladder(
    active: &[ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    ceiling: usize,
) -> usize {
    active
        .iter()
        .map(|a| a.mtp_acct.depth_drafts(ceiling, sched.levers.depth()))
        .max()
        .unwrap_or(ceiling)
}

/// Trim a serial DFlash verify (drafts taken with their confidences `conf`)
/// to its cost-aware width and log the width.
pub(super) fn lone_dflash_width(
    a: &ActiveSeq,
    conf: &[f32],
    drafts: &mut Vec<u32>,
    dflash_verify_raw_argmax: bool,
) {
    // A lone DFlash verify pays single-owner cost for each row it adds.
    if dflash_verify_raw_argmax
        && let Some(width) = super::dflash_width::choose(
            std::iter::once((&a.spec_adapt.survival, conf)),
            drafts.len(),
        )
    {
        drafts.truncate(width);
    }
    if dflash_verify_raw_argmax {
        super::dflash_width::log_verify(1, drafts.len());
    }
}

/// Cap a grammarless serial verify at the step's ladder depth.
pub(super) fn ladder_truncate(a: &ActiveSeq, drafts: &mut Vec<u32>, ladder_nd: usize) {
    // The n=1 path does not pass through the batched partition's depth
    // truncation above.  Honor the same ladder decision here so a K5
    // proposal can immediately step down to the cheaper K3 verifier.
    if a.grammar_state.is_none() && drafts.len() > ladder_nd {
        drafts.truncate(ladder_nd);
    }
}
