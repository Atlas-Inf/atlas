// SPDX-License-Identifier: AGPL-3.0-only

//! Owner-batched GLM verify partition of the MTP step.

use super::*;
use crate::scheduler::strict_spec::StrictVerify;

/// `ATLAS_GLM_STRICT_OWNER_BATCH=1`: strict structured-output owners join the
/// owner batch, each of their rows picked under its grammar mask in the
/// shared traversal, instead of taking a target traversal each. Read once per
/// process, and by the head alone: a worker learns a masked batch from the
/// E9 width word.
fn strict_owner_batch() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var("ATLAS_GLM_STRICT_OWNER_BATCH").as_deref() == Ok("1"))
}

/// Whether a sequence may join the owner batch. Grammarless always; a strict
/// grammar only where the batch masks its rows (`strict_ok`). Any other
/// grammar never: its picks come from the host pipeline, row by row.
fn owner_rides(a: &ActiveSeq, strict_ok: bool) -> bool {
    a.grammar_state.is_none() || (strict_ok && a.strict_grammar())
}

/// Strict owners' walks by `active` index (`strict_spec::prepare`): the
/// drafts each may verify and one mask per verify row.
#[derive(Default)]
struct Walks(Vec<(usize, StrictVerify)>);

impl Walks {
    /// Walk every strict owner of `serial_idxs`, once, where a batch could
    /// form at all. The walk rolls the matcher back and leaves the pending
    /// drafts alone, so an owner that ends up outside the group verifies on
    /// the serial path exactly as before; one whose walk fails is left to
    /// that path too, which reports it.
    fn walk(
        model: &dyn Model,
        active: &mut [ActiveSeq],
        serial_idxs: &[usize],
        strict_ok: bool,
    ) -> Self {
        if !strict_ok {
            return Self::default();
        }
        let candidate = |a: &ActiveSeq| owner_rides(a, strict_ok) && !a.pending_drafts.is_empty();
        let candidates = serial_idxs
            .iter()
            .filter(|&&i| candidate(&active[i]))
            .count();
        if candidates < 2 {
            return Self::default();
        }
        let vocab = model.vocab_size();
        let mut walks = Vec::new();
        for &i in serial_idxs {
            let a = &mut active[i];
            if !a.strict_grammar() || a.pending_drafts.is_empty() {
                continue;
            }
            // A sampled or scored owner the split head cannot serve is
            // refused by its own serial verify (`split_head`); in a group
            // that refusal would end every owner.
            let greedy = a.temperature <= 0.0 && a.top_logprobs.is_none();
            if !greedy && !model.verify_split_sampling() {
                continue;
            }
            let drafts = a.pending_drafts.clone();
            if let Ok(Some(walk)) = crate::scheduler::strict_spec::prepare(a, &drafts, vocab) {
                walks.push((i, walk));
            }
        }
        Self(walks)
    }

    fn get(&self, i: usize) -> Option<&StrictVerify> {
        self.0.iter().find(|(o, _)| *o == i).map(|(_, walk)| walk)
    }

    /// Whether `a` (`active[i]`) may join: grammarless, or a walked strict
    /// owner.
    fn rides(&self, a: &ActiveSeq, i: usize) -> bool {
        a.grammar_state.is_none() || self.get(i).is_some()
    }

    /// The drafts `a` (`active[i]`) can verify: the prefix a strict owner's
    /// walk kept, otherwise every draft it holds.
    fn drafts(&self, a: &ActiveSeq, i: usize) -> usize {
        self.get(i)
            .map_or(a.pending_drafts.len(), |walk| walk.drafts.len())
    }

    fn take(&mut self, i: usize) -> Option<StrictVerify> {
        let at = self.0.iter().position(|(o, _)| *o == i)?;
        Some(self.0.swap_remove(at).1)
    }
}

/// Pull same-width owners out of `serial_idxs` and verify them in one target
/// traversal (`step_verify_glm_long_batched`): grammarless owners, and with
/// `ATLAS_GLM_STRICT_OWNER_BATCH` strict structured-output owners too.
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
    // Strict owners ride only on the lane whose verify head masks rows.
    let strict_ok = strict_owner_batch()
        && crate::scheduler::strict_spec::enabled(&sched.levers, model, dflash_verify_raw_argmax);
    verify_owner_batch_with(
        model,
        active,
        sched,
        serial_idxs,
        num_drafts,
        ladder_nd,
        glm_repaired_narrow,
        verify_ctx,
        dflash_verify_raw_argmax,
        strict_ok,
    );
}

/// [`verify_owner_batch`] with the strict-owner switch supplied.
#[allow(clippy::too_many_arguments)]
fn verify_owner_batch_with(
    model: &dyn Model,
    active: &mut [ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    serial_idxs: &mut Vec<usize>,
    num_drafts: usize,
    ladder_nd: usize,
    glm_repaired_narrow: bool,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
    strict_ok: bool,
) {
    // E9 stages every owner's final normalized hidden rows and full logits;
    // EA restores them before that owner's canonical commit and private repair.
    let native = model.native_mtp_enabled();
    // A strict owner counts for the drafts its walk kept; without the switch
    // there are no walks and every owner below is grammarless, as before.
    let mut walks = Walks::walk(model, active, serial_idxs.as_slice(), strict_ok);
    // Owner-batched GLM verify: owners with the same draft width verify in
    // ONE target traversal instead of one traversal per owner —
    // two drafts on the repaired long-context K3 lane, the most common width
    // on the DFlash lane. Everything else keeps the per-sequence path.
    // With dead rows (the chain, and the pair under
    // `ATLAS_GLM_PAIR_LIVE_WIDTHS`), each chosen owner's verified drafts
    // (`(owner, drafts)`): at most `max` below, so a strict owner's stay
    // within the drafts its walk kept.
    let mut live: Vec<(usize, usize)> = Vec::new();
    let adaptive_width = if dflash_verify_raw_argmax {
        let owners: Vec<usize> = serial_idxs
            .iter()
            .copied()
            .filter(|&i| walks.rides(&active[i], i) && !active[i].pending_drafts.is_empty())
            .collect();
        // Widest width every owner holds that the batch's row budget admits.
        let fits = (1..=owners
            .iter()
            .map(|&i| walks.drafts(&active[i], i))
            .min()
            .unwrap_or(0))
            .rev()
            .find(|&w| model.can_batch_glm_long_verify_rows(owners.len(), w + 1));
        let max = fits.unwrap_or(0);
        let chosen = (owners.len() >= 2)
            .then(|| {
                super::dflash_width::choose_live(
                    owners.iter().map(|&i| active[i].draft_conf()),
                    max,
                )
            })
            .flatten();
        if let Some((width, counts)) = chosen {
            live = owners.iter().copied().zip(counts).collect();
            Some(width)
        } else {
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
        }
    } else {
        None
    };
    let owner_drafts = if glm_repaired_narrow && !dflash_verify_raw_argmax && ladder_nd >= 2 {
        Some(2)
    } else if let Some(width) = adaptive_width {
        // Cost-aware width: every owner holds at least `width` drafts.
        Some(width)
    } else if dflash_verify_raw_argmax || native {
        // Owners with at least `w` drafts can verify together at width `w`;
        // take the width that verifies the most rows.
        let lens: Vec<usize> = serial_idxs
            .iter()
            .filter(|&&i| walks.rides(&active[i], i))
            .map(|&i| walks.drafts(&active[i], i))
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
                let len = walks.drafts(&active[i], i);
                walks.rides(&active[i], i)
                    && (len == width || ((dflash_verify_raw_argmax || native) && len > width))
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
            let mut sorted = group.clone();
            sorted.sort_unstable();
            // In the batch's owner order: each strict owner's walk, cut to
            // the width, with its kept drafts now the drafts that owner
            // holds (a dead first draft is not the one the drafter gave).
            let mut strict: Vec<Option<StrictVerify>> = Vec::with_capacity(sorted.len());
            for &i in &sorted {
                let a = &mut active[i];
                let walk = walks.take(i).map(|mut walk| {
                    walk.truncate(width);
                    // Confidences that did not describe the drafts stay
                    // unmeasured, as the serial path reads them.
                    if a.draft_conf().is_empty() {
                        a.pending_draft_conf.clear();
                    }
                    a.pending_drafts.clone_from(&walk.drafts);
                    walk
                });
                a.pending_drafts.truncate(width);
                a.pending_draft_conf.truncate(width);
                strict.push(walk);
            }
            // Empty: every owner verifies every draft (no dead rows).
            let verified: Vec<usize> = if live.is_empty() {
                Vec::new()
            } else {
                sorted
                    .iter()
                    .map(|i| {
                        live.iter()
                            .find(|(o, _)| o == i)
                            .map_or(width, |&(_, d)| d.min(width))
                    })
                    .collect()
            };
            super::dflash_width::log_verify(group.len(), width, &verified);
            let mut batch: Vec<&mut ActiveSeq> = active
                .iter_mut()
                .enumerate()
                .filter(|(i, _)| sorted.binary_search(i).is_ok())
                .map(|(_, a)| a)
                .collect();
            if strict.iter().all(Option::is_none) {
                step_verify_glm_long_batched(
                    model,
                    &mut batch,
                    sched,
                    num_drafts,
                    verify_ctx,
                    dflash_verify_raw_argmax,
                    &verified,
                );
            } else {
                crate::scheduler::verify_dflash_step::step_verify_glm_long_strict(
                    model,
                    &mut batch,
                    sched,
                    num_drafts,
                    verify_ctx,
                    dflash_verify_raw_argmax,
                    &verified,
                    &strict,
                );
            }
        }
    }
}

#[cfg(test)]
#[path = "owner_batch_tests.rs"]
mod tests;
