// SPDX-License-Identifier: AGPL-3.0-only

//! The batched-verify plan of one MTP step: which sequences ride the ONE
//! batched forward, and at how many rows each.
//!
//! A step used to split the batch three ways. Sequences holding exactly the
//! ladder depth of drafts shared one batched verify; a sequence the drafter's
//! confidence stop left with fewer took a per-sequence verify forward of its
//! own; and a sequence holding none (just admitted, or its propose skipped)
//! took a separate bootstrap decode forward. At C=8 with the confidence stop
//! armed that is several weight passes a step, and the throughput gate,
//! measuring them against a single batched decode, parked the batch in serial
//! decode for most steps.
//!
//! Now every grammarless verify-ready sequence rides the batched verify at
//! its own depth (`drafts + 1` rows, D-Cut prunes within it), and when the
//! model verifies decode rows (`Model::can_batch_verify` admits `k = 1`,
//! qwen4_exp's exact lane) every draftless grammarless sequence rides it too
//! at ONE row: the target token at its position, the verify row a decode
//! step would compute. Its verdict emits that row's pick and its propose runs
//! with the batch's, exactly as the bootstrap's would. Anything the plan
//! cannot carry (grammar, DFlash's uniform-width contract, a shape the model
//! refuses) keeps its previous path.
//!
//! Kill switch `ATLAS_NO_MTP_ONE_FORWARD` (PRESENCE): draftless sequences
//! bootstrap in their own forward again.

use super::*;

/// Kill switch, PRESENCE check, read once per process.
fn decode_rows_disabled() -> bool {
    static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_ONE_FORWARD").is_some())
}

/// The batched verify of one step.
pub(super) struct VerifyPlan {
    /// Members in dispatch order (`active` indices).
    pub batch: Vec<usize>,
    /// Rows per member: drafts + 1, or 1 for a decode row.
    pub ks: Vec<usize>,
    /// Whether depths were paired canonically (`mtp_dcut::assign`); the
    /// per-chunk re-ordering must use the same arm.
    pub canonical: bool,
    /// Verify-ready sequences the per-sequence loop serves.
    pub serial: Vec<usize>,
    /// Draftless sequences carried as decode rows; the caller must not
    /// bootstrap them.
    pub decode_rows: Vec<usize>,
}

/// Plan the batched verify over `verify_idxs` (sequences holding drafts)
/// plus, where admissible, `bootstrap_idxs` (sequences holding none) as
/// decode rows. Truncates drafts to the planned depth.
///
/// `deep_ceiling` (`ATLAS_MTP_DYNAMIC_DEPTH`, `mtp_deep_depth`): each
/// sequence verifies at most its own ceiling under it, the batch at most
/// `deep_depth::row_budget` rows, and a LONE sequence holding more drafts
/// than its own verify serves rides the batched verify as a batch of one
/// where the model admits it (qwen4_exp's exact lane, 5..8 rows).
#[allow(clippy::too_many_arguments)]
pub(super) fn plan_verify(
    model: &dyn Model,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    active: &mut [ActiveSeq],
    verify_idxs: &[usize],
    bootstrap_idxs: &[usize],
    ladder_nd: usize,
    deep_ceiling: Option<usize>,
    dflash_verify_raw_argmax: bool,
) -> VerifyPlan {
    let mut plan = VerifyPlan {
        batch: Vec::new(),
        ks: Vec::new(),
        canonical: false,
        serial: Vec::new(),
        decode_rows: Vec::new(),
    };
    // DSpark is included unless ATLAS_NO_DFLASH_BATCH_VERIFY (presence). MTP
    // still uses !dflash via the model self-gate (`dflash_hidden_save`).
    let dspark_batch_ok = !dflash_verify_raw_argmax || !dspark_batch_verify_disabled();
    if spark_model::speculative::glm_repair_policy::enabled()
        || !spark_model::speculative::mtp_multi_seq_mode()
        || !dspark_batch_ok
        || batch_verify_disabled()
        || ladder_nd == 0
    {
        plan.serial = verify_idxs.to_vec();
        return plan;
    }
    // DSpark's release contract is a fixed width at every batch size, so
    // only owners holding the full ladder depth batch there. Everywhere else
    // a sequence verifies the drafts it holds, up to the ladder depth.
    let min_drafts = if dflash_verify_raw_argmax {
        ladder_nd
    } else {
        1
    };
    for &idx in verify_idxs {
        let a = &active[idx];
        if a.grammar_state.is_none() && a.pending_drafts.len() >= min_drafts {
            plan.batch.push(idx);
        } else {
            plan.serial.push(idx);
        }
    }
    if !plan.batch.is_empty()
        && !dflash_verify_raw_argmax
        && !decode_rows_disabled()
        && !crate::scheduler::mtp_bootstrap_step::dflash_ctx_serializes(sched)
        && !model.decode_logits_fp32()
    {
        plan.decode_rows = bootstrap_idxs
            .iter()
            .copied()
            .filter(|&i| active[i].grammar_state.is_none())
            .collect();
        plan.batch.extend_from_slice(&plan.decode_rows);
    }
    if let Some(ceiling) = deep_ceiling {
        deep_truncate(sched, active, &plan.batch, ceiling, ladder_nd);
    }
    if let [lone] = plan.batch[..]
        && deep_ceiling.is_some()
        && !plan.decode_rows.contains(&lone)
        && model.can_batch_verify(&[active[lone].pending_drafts.len() + 1])
    {
        // A lone deep window: its own drafts, no D-Cut (the confidence stop
        // and the depth controller already chose them).
        plan.ks = vec![active[lone].pending_drafts.len() + 1];
        active[lone]
            .pending_draft_conf
            .truncate(active[lone].pending_drafts.len());
        plan.decode_rows.clear();
        return plan;
    }
    if plan.batch.len() < 2 {
        // A lone sequence verifies on the per-sequence path, as before.
        plan.serial.extend(
            plan.batch
                .drain(..)
                .filter(|i| !plan.decode_rows.contains(i)),
        );
        plan.decode_rows.clear();
        return plan;
    }
    // Surplus drafts past the ladder depth are not verified (their
    // confidences are left as they were, as before: D-Cut then reads the
    // sequence as unmeasured).
    for &idx in &plan.batch {
        active[idx].pending_drafts.truncate(ladder_nd);
    }
    let rows = ladder_nd + 1;
    if dflash_verify_raw_argmax {
        plan.ks = vec![rows; plan.batch.len()];
        plan.canonical =
            spark_model::speculative::verify_key::canonical_assignment(plan.batch.len());
    } else {
        let planned = crate::scheduler::mtp_dcut::plan(active, &mut plan.batch, ladder_nd);
        plan.ks = planned.ks;
        plan.canonical = planned.canonical;
    }
    // Decode rows ride only a shape the model verifies in ONE forward;
    // otherwise they bootstrap as before. Dropping them keeps the drafted
    // members' order valid (a subsequence of a slot- or depth-ordered batch
    // is still ordered) and their truncated drafts are still prefixes.
    if !plan.decode_rows.is_empty()
        && (crate::scheduler::mtp_dcut::chunk_ranges(&plan.ks).len() != 1
            || !model.can_batch_verify(&plan.ks))
    {
        let (batch, ks): (Vec<usize>, Vec<usize>) = plan
            .batch
            .iter()
            .zip(&plan.ks)
            .filter(|&(_, &k)| k > 1)
            .map(|(&i, &k)| (i, k))
            .unzip();
        plan.batch = batch;
        plan.ks = ks;
        plan.decode_rows.clear();
    }
    plan
}

/// Cut each member's drafts to its own dynamic ceiling (at most the step's
/// `ladder_nd`), then the batch to the deep row budget, deepest first.
fn deep_truncate(
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    active: &mut [ActiveSeq],
    members: &[usize],
    ceiling: usize,
    ladder_nd: usize,
) {
    use crate::scheduler::mtp_deep_depth::fit_row_budget;
    let mut caps: Vec<usize> = members
        .iter()
        .map(|&i| {
            let a = &active[i];
            let own = a.mtp_acct.depth_drafts(ceiling, sched.levers.depth());
            a.pending_drafts.len().min(own).min(ladder_nd)
        })
        .collect();
    let budget = spark_model::speculative::deep_depth::row_budget(
        members.len(),
        ceiling + 1,
        crate::scheduler::mtp_dcut::VERIFY_ROW_BUDGET,
    );
    fit_row_budget(&mut caps, budget);
    for (&i, &c) in members.iter().zip(&caps) {
        active[i].pending_drafts.truncate(c);
        active[i].pending_draft_conf.truncate(c);
    }
}

#[cfg(test)]
#[path = "one_forward_tests.rs"]
mod tests;
