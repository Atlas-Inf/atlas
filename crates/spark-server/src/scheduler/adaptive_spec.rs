// SPDX-License-Identifier: AGPL-3.0-only

//! Adaptive speculation (`ATLAS_DFLASH_ADAPTIVE=1`).
//!
//! Speculation pays only when τ (= mean accepted + 1 bonus) exceeds
//! step_time / serial_time — ≈3.5 at the measured 222ms γ16 verify step vs
//! 63ms serial decode, i.e. mean accepted ≥ ~2.5. Measured 2026-07-08 on
//! coherent output: MinHeap code runs τ≈5.7 (spec wins, +30% vs serial),
//! Volvo prose runs τ≈2.3 (spec LOSES ~20% vs serial). Content decides.
//!
//! Policy: per-sequence rolling window of `accepted` over the last
//! [`WINDOW`] K=γ verify steps. Window full and mean below the threshold →
//! SUSPEND speculation for that sequence (no proposing; the scheduler's
//! bootstrap path serial-decodes it at full NOSPEC pace). After
//! `sched.levers.dflash_adaptive_reprobe` serial tokens, UN-suspend and re-probe: the window
//! must refill before suspension can re-trigger, so a probe costs WINDOW
//! spec steps (~2.7s) once per re-probe interval — a few percent on pure
//! prose, nothing on accepting content, and mixed documents (prose→code)
//! re-engage speculation automatically.
//!
//! Net posture: never materially slower than plain decode, +30% where
//! acceptance supports it. State is transient (reset on swap/restore —
//! a resumed sequence just re-measures).
//!
//! Knobs (env, read once): `ATLAS_DFLASH_ADAPTIVE=1` master switch;
//! `ATLAS_DFLASH_ADAPTIVE_MIN` mean-accepted suspend threshold (default
//! 2.0); `ATLAS_DFLASH_ADAPTIVE_REPROBE` serial tokens between probes
//! (default 256).

use crate::scheduler::ActiveSeq;

/// Rolling accept window + suspend state, embedded in [`ActiveSeq`].
#[derive(Default)]
pub(crate) struct AdaptState {
    /// Per-position draft acceptance for the verify width choice.
    pub(crate) survival: super::dflash_width::DraftSurvival,
    /// Copy drafts' index and outcomes (`ATLAS_DFLASH_COPY_DRAFTS`,
    /// `ATLAS_QWEN_COPY_DRAFTS`).
    pub(crate) copy: super::copy_drafts::CopyState,
    window: Vec<u32>,
    /// What the steps in `window` had to accept to pay ([`step_share`] of
    /// the block threshold each), summed as the window is.
    window_share: Vec<f32>,
    suspended: bool,
    serial_tokens: u32,
    /// Probes in a row that ended in a suspension: each doubles the wait
    /// before the next ([`hold_tokens`]); a window that pays clears it.
    failed_probes: u32,
    /// Accepts of the last narrow steps (fewer drafts offered than a block:
    /// two or more live sequences on a depth table), and the width they
    /// were offered at.
    narrow: Vec<u32>,
    narrow_offered: usize,
    /// Plain batched steps left before this sequence's batch drafts again
    /// ([`judge_narrow`]).
    hold: u32,
    /// The current verify span's per-position pipeline effects, applied as
    /// each picked token is committed (`emit_step::PickEffects`).
    pub(in crate::scheduler) pick_effects: crate::scheduler::emit_step::PickEffects,
}

const WINDOW: usize = 12;

/// Whether a step that offered `offered` drafts feeds the per-sequence
/// window. The suspension threshold asks how a whole block does; a narrower
/// step (two or more live sequences on a depth table) offers fewer drafts
/// than it asks to be accepted, so it is judged with its batch instead
/// ([`note_narrow`], [`judge_narrow`]).
pub(crate) fn judges_block(offered: usize) -> bool {
    offered >= super::dflash_width::MAX_DRAFTS
}

/// Record a narrow verify step's accept count for [`judge_narrow`].
pub(crate) fn note_narrow(
    a: &mut ActiveSeq,
    num_accepted: usize,
    offered: usize,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    if !sched.levers.dflash_adaptive {
        return;
    }
    let st = &mut a.spec_adapt;
    if st.narrow_offered != offered {
        st.narrow.clear();
        st.narrow_offered = offered;
    }
    st.narrow.push(num_accepted as u32);
    if st.narrow.len() > WINDOW {
        st.narrow.remove(0);
    }
}

/// Whether narrow steps pay: the sequences' pooled mean accepted against
/// the block threshold `min` scaled to the rows the step verifies
/// (`offered + 1` of a block's `MAX_DRAFTS + 1`). `None` until every
/// sequence has a full window at that width.
fn narrow_pays<'a>(
    windows: impl Iterator<Item = (&'a [u32], usize)>,
    offered: usize,
    min: f32,
) -> Option<(bool, f32)> {
    let (mut sum, mut steps) = (0u32, 0usize);
    for (window, width) in windows {
        if window.len() < WINDOW || width != offered {
            return None;
        }
        sum += window.iter().sum::<u32>();
        steps += window.len();
    }
    if steps < 2 * WINDOW {
        return None;
    }
    let mean = sum as f32 / steps as f32;
    let floor = min * (offered + 1) as f32 / (super::dflash_width::MAX_DRAFTS + 1) as f32;
    Some((mean >= floor, mean))
}

/// After a batched verify of narrow steps: when the batch's drafts stop
/// paying, hold every member to plain batched decode until the re-probe.
/// The members stop together because a batch that drafts for one sequence
/// and not the other runs two forwards a step.
pub(crate) fn judge_narrow(
    batch: &mut [&mut ActiveSeq],
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    if !sched.levers.dflash_adaptive {
        return;
    }
    let Some(offered) = batch
        .iter()
        .find(|a| !a.finished)
        .map(|a| a.spec_adapt.narrow_offered)
    else {
        return;
    };
    let live = batch.iter().filter(|a| !a.finished);
    let verdict = narrow_pays(
        live.map(|a| (a.spec_adapt.narrow.as_slice(), a.spec_adapt.narrow_offered)),
        offered,
        sched.levers.dflash_adaptive_min,
    );
    if let Some((false, mean)) = verdict {
        for a in batch.iter_mut() {
            a.spec_adapt.hold = sched.levers.dflash_adaptive_reprobe;
            a.spec_adapt.narrow.clear();
        }
        tracing::info!(
            "adaptive spec: batch HELD (mean accepted {mean:.2} of {offered} offered over \
             {WINDOW} steps) — plain decode until re-probe"
        );
    }
}

/// Whether this sequence's batch is held to plain decode.
pub(crate) fn held(a: &ActiveSeq) -> bool {
    a.spec_adapt.hold > 0
}

/// Once a scheduler step: whether a hold keeps this step's sequences on
/// plain batched decode, counting the step against it. A sequence left
/// alone drops its hold (alone it drafts whole blocks, judged by its own
/// window).
pub(crate) fn plain_hold_step(active: &mut [ActiveSeq]) -> bool {
    if active.len() < 2 {
        for a in active.iter_mut() {
            a.spec_adapt.hold = 0;
        }
        return false;
    }
    let held = active.iter().any(held);
    for a in active.iter_mut() {
        a.spec_adapt.hold = a.spec_adapt.hold.saturating_sub(1);
    }
    held
}

/// The share of the block threshold a step that offered `offered` drafts
/// must accept to pay for itself. The threshold is what a whole block
/// (`MAX_DRAFTS` or more) must accept; a step cut shorter (the verify width
/// rules, `dflash_width` and `dflash_conf_width`) verifies fewer rows but
/// still pays for its propose, about a quarter of a block's extra cost over
/// a plain step (Inkling 2 on one GB10: 13.6 ms of propose and bookkeeping
/// against 5.6-6.5 ms a verify row).
fn step_share(offered: usize) -> f32 {
    const FIXED: f32 = 0.25;
    let block = super::dflash_width::MAX_DRAFTS;
    FIXED + (1.0 - FIXED) * offered.min(block) as f32 / block as f32
}

/// Record one verify step's accept count of the `offered` drafts it
/// verified; may trip suspension. Call after `num_accepted` is known
/// (verify_dflash_step).
pub(crate) fn record_verify(
    a: &mut ActiveSeq,
    num_accepted: usize,
    offered: usize,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) {
    if !sched.levers.dflash_adaptive {
        return;
    }
    let st = &mut a.spec_adapt;
    st.window.push(num_accepted as u32);
    st.window_share.push(step_share(offered));
    if st.window.len() > WINDOW {
        st.window.remove(0);
        st.window_share.remove(0);
    }
    if st.window.len() == WINDOW {
        let mean = st.window.iter().sum::<u32>() as f32 / WINDOW as f32;
        let floor =
            sched.levers.dflash_adaptive_min * st.window_share.iter().sum::<f32>() / WINDOW as f32;
        if mean < floor {
            let hold = hold_tokens(sched.levers.dflash_adaptive_reprobe, st.failed_probes);
            st.suspended = true;
            st.serial_tokens = 0;
            st.failed_probes = st.failed_probes.saturating_add(1);
            st.window.clear();
            st.window_share.clear();
            tracing::info!(
                "adaptive spec: SUSPENDED (mean accepted {mean:.2} < {floor:.2} over {WINDOW} \
                 steps) — serial decode for {hold} tokens"
            );
        } else {
            st.failed_probes = 0;
        }
    }
}

/// Serial tokens before the re-probe after the `failed`-th failed probe in
/// a row: an eighth of the interval `reprobe` first, doubling to four times
/// it. Text that drafts well again soon (a comment inside code) is back in
/// a few dozen tokens; text that keeps failing (prose) is probed less often
/// than the fixed interval probed it.
fn hold_tokens(reprobe: u32, failed: u32) -> u32 {
    (reprobe / 8).max(1).saturating_mul(1 << failed.min(5))
}

/// May this sequence propose/speculate right now? Un-suspends (re-probe)
/// once enough serial tokens have passed.
pub(crate) fn spec_allowed(
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
) -> bool {
    if !sched.levers.dflash_adaptive {
        return true;
    }
    let st = &mut a.spec_adapt;
    if !st.suspended {
        return true;
    }
    // `failed_probes` already counts the suspension being waited out.
    let hold = hold_tokens(
        sched.levers.dflash_adaptive_reprobe,
        st.failed_probes.saturating_sub(1),
    );
    if st.serial_tokens >= hold {
        st.suspended = false;
        st.serial_tokens = 0;
        st.window.clear();
        st.window_share.clear();
        tracing::info!("adaptive spec: RE-PROBING after {hold} serial tokens");
        return true;
    }
    false
}

/// Is this sequence currently in the adaptive-suspended (serial) regime?
/// Read-only peek — unlike `spec_allowed`, never mutates re-probe state.
pub(crate) fn is_suspended(a: &ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) -> bool {
    sched.levers.dflash_adaptive && a.spec_adapt.suspended
}

/// Count a serially-decoded token toward the re-probe interval.
pub(crate) fn tick_serial(a: &mut ActiveSeq, sched: &crate::scheduler::sched_ctx::SchedCtx) {
    if sched.levers.dflash_adaptive && a.spec_adapt.suspended {
        a.spec_adapt.serial_tokens = a.spec_adapt.serial_tokens.saturating_add(1);
    }
}

// The `ATLAS_DFLASH_SERIAL_APPEND` and `ATLAS_DFLASH_UNIFIED_CTX` statics
// that lived here are now `SchedLevers::dflash_serial_append` /
// `::dflash_unified_ctx`, resolved once per run and read through `SchedCtx`.

// The `ATLAS_DFLASH_ADAPTIVE*` statics are now `SchedLevers::dflash_adaptive*`.

#[cfg(test)]
mod tests {
    use super::{WINDOW, narrow_pays};

    #[test]
    fn a_shorter_step_owes_less_but_never_under_its_propose() {
        // a whole block, or a deeper drafter's, owes the whole threshold
        assert_eq!(super::step_share(7), 1.0);
        assert_eq!(super::step_share(15), 1.0);
        // two drafts: a quarter for the propose plus two rows of seven
        assert!((super::step_share(2) - (0.25 + 0.75 * 2.0 / 7.0)).abs() < 1e-6);
        assert_eq!(super::step_share(0), 0.25);
    }

    #[test]
    fn the_wait_starts_short_and_doubles_to_four_intervals() {
        let waits: Vec<u32> = (0..7).map(|f| super::hold_tokens(256, f)).collect();
        assert_eq!(waits, [32, 64, 128, 256, 512, 1024, 1024]);
        assert_eq!(super::hold_tokens(2, 0), 1);
    }

    #[test]
    fn only_whole_blocks_feed_the_window() {
        assert!(super::judges_block(7) && super::judges_block(15));
        assert!(!super::judges_block(2) && !super::judges_block(3));
    }

    #[test]
    fn narrow_steps_are_judged_pooled_at_the_scaled_threshold() {
        let (code, prose) = ([1u32; WINDOW], [0u32; WINDOW]);
        let pays = |w: &[&[u32]], min| narrow_pays(w.iter().map(|w| (*w, 2)), 2, min);
        // 3 rows of a block's 8, at a 2.0 block threshold: 0.75 a step
        assert_eq!(pays(&[&code, &code], 2.0), Some((true, 1.0)));
        assert_eq!(pays(&[&code, &prose], 2.0), Some((false, 0.5)));
        assert_eq!(pays(&[&prose, &prose], 2.0), Some((false, 0.0)));
        // a lower block threshold (0.26 a step) keeps the mixed pair drafting
        assert_eq!(pays(&[&code, &prose], 0.7), Some((true, 0.5)));
        // no verdict before every window is full, at one width, for two or more
        assert_eq!(pays(&[&code, &prose[..WINDOW - 1]], 2.0), None);
        assert_eq!(pays(&[&code], 2.0), None);
        let widths = [(&code[..], 2), (&prose[..], 3)];
        assert_eq!(narrow_pays(widths.into_iter(), 2, 2.0), None);
    }
}
