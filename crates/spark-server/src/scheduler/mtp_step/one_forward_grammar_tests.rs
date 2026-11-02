// SPDX-License-Identifier: AGPL-3.0-only

//! Grammar sequences on the batched verify (`plan_verify` routing).

use super::*;

// ── grammar sequences on the batched verify ──────────────────────────────

/// A non-strict (tool) grammar matcher, the disengage contract.
fn lenient_grammar() -> crate::grammar::GrammarState {
    let mut engine = crate::scheduler::test_support::grammar_engine();
    let compiled = engine.compile_json_grammar().unwrap();
    crate::grammar::GrammarState::new(&compiled, engine.vocab_size()).unwrap()
}

/// A strict (response_format) grammar, armed by spec.
fn strict_grammar() -> crate::grammar::GrammarState {
    let mut sink = crate::scheduler::types::ResponseSink::Blocking(None);
    crate::scheduler::emit_step::compile_grammar_state(
        &mut Some(crate::scheduler::test_support::grammar_engine()),
        &Some(crate::api::GrammarSpec::JsonObject),
        &[crate::scheduler::test_support::GRAMMAR_EOS],
        false,
        None,
        None,
        &mut sink,
    )
    .unwrap()
    .unwrap()
}

fn plan(
    active: &mut [ActiveSeq],
    verify: &[usize],
    bootstrap: &[usize],
) -> super::super::VerifyPlan {
    let model = Stub::new(true);
    let sched = SchedCtx::for_test();
    super::super::plan_verify(&model, &sched, active, verify, bootstrap, 3, None, false)
}

/// A non-strict grammar sequence with drafts rides the batch; a strict one
/// stays serial; grammarless is unchanged; the kill switch restores serial.
#[test]
fn non_strict_grammar_rides_the_batched_verify() {
    let (mut active, _rx) = seqs(vec![(10, vec![11, 12]), (20, vec![21, 22]), (30, vec![31])]);
    active[0].grammar_state = Some(lenient_grammar());
    active[1].grammar_state = Some(strict_grammar());
    let p = plan(&mut active, &[0, 1, 2], &[]);
    assert!(p.batch.contains(&0), "non-strict grammar must ride batch");
    assert!(p.serial.contains(&1), "strict grammar stays serial");
    assert!(p.batch.contains(&2), "grammarless unchanged");

    // Kill switch: injected value flips only the non-strict member.
    assert!(!super::super::grammar_rides_batch(&active[0], true));
    assert!(super::super::grammar_rides_batch(&active[0], false));
    assert!(!super::super::grammar_rides_batch(&active[1], false));
    assert!(super::super::grammar_rides_batch(&active[2], true));
}

/// A grammar bootstrap sequence (no drafts) lands in decode_rows under the
/// same non-strict gate.
#[test]
fn a_non_strict_bootstrap_row_joins_decode_rows() {
    let (mut active, _rx) = seqs(vec![(10, vec![11, 12]), (20, vec![])]);
    active[1].grammar_state = Some(lenient_grammar());
    let p = plan(&mut active, &[0], &[1]);
    assert!(
        p.decode_rows.contains(&1),
        "grammar bootstrap joins decode rows"
    );
}
