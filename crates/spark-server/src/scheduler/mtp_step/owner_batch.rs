// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched GLM verify partition of the MTP step.

use super::*;

/// Pull grammarless same-width owners out of `serial_idxs` and verify them in
/// one target traversal (`step_verify_glm_long_batched`).
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_owner_batch(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    serial_idxs: &mut Vec<usize>,
    num_drafts: usize,
    ladder_nd: usize,
    glm_repaired_narrow: bool,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    // Owner-batched GLM verify: grammarless owners with the same draft width
    // verify in ONE target traversal instead of one traversal per owner —
    // two drafts on the repaired long-context K3 lane, the most common width
    // on the DFlash lane. Everything else keeps the per-sequence path.
    let adaptive_width = if dflash_verify_raw_argmax {
        let owners: Vec<usize> = serial_idxs
            .iter()
            .copied()
            .filter(|&i| active[i].grammar_state.is_none() && !active[i].pending_drafts.is_empty())
            .collect();
        // Widest width every owner holds that the batch's row budget admits.
        let fits = (1..=owners
            .iter()
            .map(|&i| active[i].pending_drafts.len())
            .min()
            .unwrap_or(0))
            .rev()
            .find(|&w| model.can_batch_glm_long_verify_rows(owners.len(), w + 1));
        let max = fits.unwrap_or(0);
        (owners.len() >= 2)
            .then(|| {
                super::dflash_width::choose(
                    owners
                        .iter()
                        .map(|&i| (&active[i].spec_adapt.survival, active[i].draft_conf())),
                    max,
                )
            })
            .flatten()
    } else {
        None
    };
    let owner_drafts = if glm_repaired_narrow && !dflash_verify_raw_argmax && ladder_nd >= 2 {
        Some(2)
    } else if let Some(width) = adaptive_width {
        // Cost-aware width: every owner holds at least `width` drafts.
        Some(width)
    } else if dflash_verify_raw_argmax {
        // Owners with at least `w` drafts can verify together at width `w`;
        // take the width that verifies the most rows.
        let lens: Vec<usize> = serial_idxs
            .iter()
            .filter(|&&i| active[i].grammar_state.is_none())
            .map(|&i| active[i].pending_drafts.len())
            .filter(|&len| len > 0)
            .collect();
        let owners_at = |w: usize| lens.iter().filter(|&&len| len >= w).count();
        lens.iter()
            .copied()
            .filter(|&w| {
                owners_at(w) < 2 || model.can_batch_glm_long_verify_rows(owners_at(w), w + 1)
            })
            .max_by_key(|&w| (owners_at(w) * (w + 1), w))
    } else {
        None
    };
    if let Some(width) = owner_drafts {
        // DFlash owners holding more drafts than the common width join at
        // that width: the drafter's rollback keys on accepted rows only, and
        // one shared traversal beats a second per-owner one.
        let group: Vec<usize> = serial_idxs
            .iter()
            .copied()
            .filter(|&i| {
                let len = active[i].pending_drafts.len();
                active[i].grammar_state.is_none()
                    && (len == width || (dflash_verify_raw_argmax && len > width))
            })
            .collect();
        let min_group = std::env::var("ATLAS_GLM_LONG_BATCH_MIN")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);
        if group.len() >= min_group && model.can_batch_glm_long_verify_rows(group.len(), width + 1)
        {
            serial_idxs.retain(|i| !group.contains(i));
            for &i in &group {
                active[i].pending_drafts.truncate(width);
                active[i].pending_draft_conf.truncate(width);
            }
            super::dflash_width::log_verify(group.len(), width);
            let mut sorted = group.clone();
            sorted.sort_unstable();
            let mut batch: Vec<&mut ActiveSeq> = active
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| sorted.binary_search(i).is_ok())
                .map(|(_, a)| a)
                .collect();
            step_verify_glm_long_batched(
                model,
                &mut batch,
                sched,
                num_drafts,
                verify_ctx,
                dflash_verify_raw_argmax,
            );
        }
    }
}
