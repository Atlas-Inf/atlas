// SPDX-License-Identifier: AGPL-3.0-only

//! The MTP lane's pieces (`copy_drafts::mtp`) on their own; `step_mtp`
//! drives them on the exact-lane stub in `mtp_step/one_forward_tests/copy.rs`.

use super::super::{COPY_CONF, COPY_REPLY_CONF};
use super::*;
use crate::scheduler::lifecycle_tests::StubModel;

/// The knobs' defaults with a 3-token match.
fn knobs3() -> Settings {
    Settings {
        match_len: 3,
        ..Settings::knobs(|_| None)
    }
}

/// A request over `prompt` that has emitted `reply` (its last token pending).
fn request(prompt: &[u32], reply: &[u32]) -> ActiveSeq {
    let (mut a, _rx) = crate::scheduler::test_support::test_seq(reply.to_vec(), 8, None, 0);
    a.seq.tokens = prompt
        .iter()
        .chain(&reply[..reply.len() - 1])
        .copied()
        .collect();
    a
}

/// A run's context with the dynamic depth and copies on or off.
fn sched(copies: bool) -> SchedCtx {
    let mut levers = crate::scheduler::levers::SchedLevers::defaults();
    levers.mtp_dynamic_depth = true;
    levers.qwen_copy_drafts = copies;
    SchedCtx::new(
        crate::scheduler::vocab_masks::VocabMasks::default(),
        std::sync::Arc::new(levers),
        std::sync::Arc::new(crate::scheduler::snapshot::SnapshotCell::default()),
        crate::scheduler::limits::SchedLimits::NONE,
        crate::scheduler::helpers::WatchdogParams::default(),
    )
}

const PROMPT: [u32; 10] = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19];

#[test]
fn a_copy_stands_in_for_the_heads_propose_up_to_its_ceiling() {
    let model = StubModel::default();
    let mut a = request(&PROMPT, &[12, 13, 14]);
    assert!(propose_with(&model, &mut a, 3, &knobs3()));
    assert_eq!(a.pending_drafts, [15, 16, 17]);
    assert_eq!(a.draft_conf(), [COPY_CONF; 3]);
    // The knob caps it below the ceiling.
    let mut a = request(&PROMPT, &[12, 13, 14]);
    let s = Settings { max: 2, ..knobs3() };
    assert!(propose_with(&model, &mut a, 7, &s));
    assert_eq!(a.pending_drafts, [15, 16]);
    // No room, no copy: the head's propose runs.
    let mut a = request(&PROMPT, &[12, 13, 14]);
    assert!(!propose_with(&model, &mut a, 0, &knobs3()));
    assert!(a.pending_drafts.is_empty());
}

#[test]
fn a_copy_of_the_reply_carries_the_reply_confidence() {
    // `5 6 7` occurs only in the reply.
    let mut a = request(&[10, 11, 12], &[5, 6, 7, 8, 9, 5, 6, 7]);
    assert!(propose_with(&StubModel::default(), &mut a, 2, &knobs3()));
    assert_eq!(a.pending_drafts, [8, 9]);
    assert_eq!(a.draft_conf(), [COPY_REPLY_CONF; 2]);
}

#[test]
fn no_copy_under_a_grammar_or_without_an_earlier_occurrence() {
    let model = StubModel::default();
    let mut a = request(&PROMPT, &[12, 13, 14]);
    let mut engine = crate::scheduler::test_support::grammar_engine();
    let compiled = engine.compile_json_grammar().unwrap();
    a.grammar_state =
        Some(crate::grammar::GrammarState::new(&compiled, engine.vocab_size()).unwrap());
    assert!(!propose_with(&model, &mut a, 3, &knobs3()));
    assert!(a.pending_drafts.is_empty());
    let mut a = request(&PROMPT, &[1, 2, 3]);
    assert!(!propose_with(&model, &mut a, 3, &knobs3()));
}

/// The stub's verify is not serial decode's: with the lever on or off,
/// nothing proposes, marks a round or lifts a depth.
#[test]
fn the_lane_needs_the_lever_and_an_exact_verify() {
    let model = StubModel::default();
    for copies in [false, true] {
        let run = sched(copies);
        let mut a = request(&PROMPT, &[12, 13, 14]);
        assert!(!propose(&model, &run, &mut a, 3));
        assert!(a.pending_drafts.is_empty());
        a.pending_drafts = vec![15, 16, 17, 18, 19, 1, 2];
        a.pending_draft_conf = vec![COPY_CONF; 7];
        assert_eq!(deep_cap(&model, &run, &a, true, 3, 7), 3);
        begin_verify(&model, &run, &mut a);
        assert!(!settle(&mut a, 7, 7), "not booked off the lane");
    }
}

#[test]
fn only_a_copy_round_is_booked_and_kept_from_the_depth_controllers() {
    let mut a = request(&[10, 11, 12], &[13]);
    // The head's round.
    assert!(!settle(&mut a, 3, 1));
    a.spec_adapt.copy.verifying = Some(COPY_CONF);
    assert!(settle(&mut a, 3, 1));
    a.spec_adapt.copy.verifying = Some(COPY_REPLY_CONF);
    assert!(settle(&mut a, 4, 4));
    assert_eq!(a.spec_adapt.copy.tally(), ([1, 1], [3, 4], [1, 4], false));
    // A preempted request resumes with no verify running.
    let mut a = crate::scheduler::preempt::preempt_requeue(&StubModel::default(), a).a;
    assert_eq!(a.spec_adapt.copy.verifying, None);
    assert!(!settle(&mut a, 1, 1));
}
