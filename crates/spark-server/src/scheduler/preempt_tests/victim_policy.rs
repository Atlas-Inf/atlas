// SPDX-License-Identifier: AGPL-3.0-only

//! `choose_decode_victim` policy tests. Split from `preempt_tests.rs` (500-LoC cap).

use super::super::preempt::choose_decode_victim;
use super::*;

// ── choose_decode_victim ─────────────────────────────────────────────────

#[test]
fn victim_policy_least_progress_wins() {
    let model = PreemptStubModel::default();
    let (a0, _r0) = active_seq(0, 7);
    let (a1, _r1) = active_seq(1, 3);
    let (a2, _r2) = active_seq(2, 12);
    let active = vec![a0, a1, a2];
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));
}

#[test]
fn victim_policy_starvation_guard_skips_resumed_until_progress() {
    let model = PreemptStubModel::default();
    let (mut a0, _r0) = active_seq(0, 3);
    // Just resumed: 3 generated, immune until 3 + PREEMPT_IMMUNITY_TOKENS.
    a0.preempt_immune_until_tokens = a0.output_tokens.len() + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 9);
    let active = vec![a0, a1];
    // The immune least-progress seq is skipped; the other is chosen.
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));

    // Once it has generated PREEMPT_IMMUNITY_TOKENS more, immunity lapses.
    let (mut a0, _r0) = active_seq(0, 3 + PREEMPT_IMMUNITY_TOKENS);
    a0.preempt_immune_until_tokens = 3 + PREEMPT_IMMUNITY_TOKENS;
    let (a1, _r1) = active_seq(1, 200);
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&model, &active, false), Some(0));
}

#[test]
fn victim_policy_all_immune_still_yields_a_victim() {
    // Immunity must never convert a recoverable exhaustion into a
    // batch-wide error: with every candidate immune, one is chosen anyway.
    let model = PreemptStubModel::default();
    let (mut a0, _r0) = active_seq(0, 4);
    a0.preempt_immune_until_tokens = usize::MAX;
    let (mut a1, _r1) = active_seq(1, 2);
    a1.preempt_immune_until_tokens = usize::MAX;
    let active = vec![a0, a1];
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));
}

#[test]
fn victim_policy_vision_requeue_excluded_but_spill_allowed() {
    const PAD: u32 = 999;
    let model = PreemptStubModel {
        vision_pad: Some(PAD),
        ..Default::default()
    };
    let (mut a0, _r0) = active_seq(0, 2);
    a0.seq.tokens.insert(2, PAD); // image KV: not re-prefillable from tokens
    let (a1, _r1) = active_seq(1, 8);
    let active = vec![a0, a1];
    // Requeue (no spill): the vision seq is ineligible despite least progress.
    assert_eq!(choose_decode_victim(&model, &active, false), Some(1));
    // Spill saves KV verbatim: the vision seq is eligible again.
    assert_eq!(choose_decode_victim(&model, &active, true), Some(0));
}
